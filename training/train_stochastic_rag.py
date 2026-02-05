"""
Stochastic RAG Training Script
==============================

Paper: "Stochastic RAG: End-to-End Retrieval-Augmented Generation
        through Expected Utility Maximization"
Authors: Zamani & Bendersky (SIGIR 2024)

Core Innovations:
-----------------
1. Models retrieval as stochastic Sampling Without Replacement (SWOR)

2. Uses Straight-Through Gumbel-Top-k for differentiable sampling

3. Maximizes expected utility with KILT-score

Key Equations:
--------------
- Equation (1) RAG Expected Utility:
    E[U] = (1/n) Σ Σ U(y, ŷ) p(ŷ|x; G_θ, R_φ)

- Equation (5) SWOR Probability:
    p(d|x) = Π p(d_i) / (1 - Σ_{j<i} p(d_j))

- Equation (7) Gumbel-Softmax:
    p̃(d_i) = exp(s_{xd_i} + G_i) / Σ exp(s_{xd} + G)
    where G ~ Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)

- Combined Loss:
    L = L_gen - E[U] × log p_SWOR(d|x)

Usage:
------
    # Train with precomputed data (recommended)
    python train_stochastic_rag.py --precomputed_path kilt_data/precomputed

    # Full training (50K steps)
    python train_stochastic_rag.py --output_dir checkpoints/stochastic_rag

    # Quick test (100 steps)
    python train_stochastic_rag.py --quick_test --steps 100

    # Initialize Generator from FiD-Light checkpoint
    python train_stochastic_rag.py --init_generator checkpoints/fidlight_paper/step_50000

Relationship with FiD-Light:
----------------------------
This script reuses FiD-Light infrastructure:
- Precomputed retrieval data (40 candidate passages)
- Temperature Sampling (T=2)
- FiD-Light compression (top-k vectors)
- Gradient Accumulation
- Checkpoint management

New Stochastic RAG components:
- StochasticReranker: Score 40 candidates
- GumbelTopKSampler: Differentiable sampling
- SWORProbability: SWOR log probability
- KILTScoreComputer: KILT-score utility function
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Adafactor,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.parallel import data_parallel

# Web Demo state reporting (optional - only used when running from web UI)
try:
    from web_demo.utils.state_io import update_step_state, StepStatus
    HAS_WEB_DEMO = True
except ImportError:
    HAS_WEB_DEMO = False


def report_training_progress(global_step: int, total_steps: int, loss: float, lr: float, utility: float = 0.0):
    """Report training progress to web demo (if available)."""
    if not HAS_WEB_DEMO:
        return
    try:
        progress = (global_step / total_steps) * 100
        message = f"Step {global_step}/{total_steps} | Loss: {loss:.4f} | E[U]: {utility:.4f}"
        update_step_state(
            step_name="train_model",
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra={
                "loss": loss,
                "lr": lr,
                "utility": utility,
                "global_step": global_step,
                "algorithm": "stochastic_rag",
                "model": "t5base"
            }
        )
    except Exception:
        pass  # Silently ignore web demo errors


# =============================================================================
# EncoderWrapper for multi-GPU data_parallel
# =============================================================================

class EncoderWrapper(nn.Module):
    """Wrapper for T5 encoder to work with data_parallel."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


# =============================================================================
# Core Module 1: Stochastic Reranker (Differentiable Reranker)
# =============================================================================

class StochasticReranker(nn.Module):
    """
    Differentiable Reranker: Scores pre-retrieved candidate documents.

    Paper Method:
    -------------
    Stochastic RAG performs reranking on N pre-retrieved candidate documents
    (e.g., N=40), rather than retrieving from all 5.9M documents. This
    significantly reduces computational cost.

    We use the first token's hidden state from T5 Encoder (similar to CLS)
    to compute scores, then map to scalar scores via a learnable linear layer.

    Architecture:
    -------------
    Input: [Batch, N_candidates, Hidden_dim] - CLS vector for each candidate
    Output: [Batch, N_candidates] - Score for each candidate

    Parameters:
        hidden_dim (int): T5 hidden dimension (T5-base=768)
        scoring_type (str): "linear" (single layer) or "mlp" (two-layer MLP)
    """

    def __init__(self, hidden_dim: int = 768, scoring_type: str = "linear"):
        """
        Initialize reranker.

        Args:
            hidden_dim: T5 model hidden dimension
                        - T5-base: 768
                        - T5-large: 1024
                        - T5-xl: 2048
            scoring_type: Scoring network type
                          - "linear": Single linear layer (fewer params, fast convergence)
                          - "mlp": Two-layer MLP (more expressive)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scoring_type = scoring_type

        if scoring_type == "linear":
            # Simple linear layer: hidden_dim -> 1
            self.scorer = nn.Linear(hidden_dim, 1)
        elif scoring_type == "mlp":
            # Two-layer MLP: hidden_dim -> hidden_dim/2 -> 1
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),  # Prevent overfitting
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown scoring_type: {scoring_type}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization to ensure initial scores are near zero."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_vectors: torch.Tensor) -> torch.Tensor:
        """
        Score candidate documents.

        Args:
            cls_vectors: [Batch, N_candidates, Hidden_dim]
                         CLS token hidden states for each candidate passage

        Returns:
            scores: [Batch, N_candidates]
                    Raw scores for each candidate (unnormalized)

        Note:
            Returns raw scores, not probabilities. The Gumbel-Top-k sampler
            will add Gumbel noise and apply softmax normalization.
        """
        # cls_vectors: [B, N, H]
        # scorer output: [B, N, 1]
        # squeeze(-1) removes last dim: [B, N]
        scores = self.scorer(cls_vectors).squeeze(-1)
        return scores


# =============================================================================
# Core Module 2: Gumbel-Top-k Sampler (Differentiable Sampler)
# =============================================================================

class GumbelTopKSampler(nn.Module):
    """
    Gumbel-Top-k Sampler: Differentiable stochastic document selection.

    Paper Method (Eq. 7):
    ---------------------
    1. Generate Gumbel noise: G ~ Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
    2. Perturb scores: s' = s + G
    3. Forward pass: Use argmax/top-k to select k documents (hard selection)
    4. Backward pass: Use softmax probabilities to compute gradients (soft gradients)

    This is the "Straight-Through Gumbel-Top-k" trick:
    - Forward: Hard selection (discrete)
    - Backward: Soft gradients (continuous)

    Why Gumbel?
    -----------
    Gumbel-Max trick: Adding Gumbel noise to logits then taking argmax is
    equivalent to sampling from a softmax distribution. This allows us to
    obtain differentiable gradients while maintaining stochasticity.

    Parameters:
        temperature (float): Softmax temperature, controls distribution "sharpness"
                             - τ → 0: Near deterministic selection (sharp)
                             - τ → ∞: Near uniform distribution (flat)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize sampler.

        Args:
            temperature: Gumbel-Softmax temperature (paper default τ=1.0)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        k: int,
        tau: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Gumbel-Top-k sampling.

        Args:
            scores: [Batch, N_candidates]
                    Raw scores from reranker
            k: Number of documents to select
            tau: Optional temperature (overrides default)

        Returns:
            selected_indices: [Batch, k]
                              Indices of selected documents (0-based)
            selected_probs: [Batch, k]
                            Soft probabilities at selected positions (for gradients)
            all_probs: [Batch, N_candidates]
                       Full softmax probability distribution

        Implementation Details:
        -----------------------
        Straight-Through Estimator:
        - Forward pass uses torch.topk (non-differentiable)
        - Backward pass uses torch.gather to extract gradients from softmax probs
        - Gradients flow back to scores through selected_probs
        """
        tau = tau if tau is not None else self.temperature

        # ========== Step 1: Generate Gumbel noise ==========
        # G = -log(-log(U)), U ~ Uniform(0, 1)
        # Add small constant 1e-9 to prevent log(0)
        u = torch.rand_like(scores)
        gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)

        # ========== Step 2: Perturb scores ==========
        # Corresponds to paper Eq. (7) numerator: exp(s_{xd_i} + G_{d_i})
        noisy_scores = scores + gumbel_noise

        # ========== Step 3: Compute soft probabilities (for backward) ==========
        # Corresponds to paper Eq. (7): p̃(d_i) = exp(s + G) / Σ exp(s + G)
        all_probs = F.softmax(noisy_scores / tau, dim=-1)

        # ========== Step 4: Hard Top-k selection (for forward) ==========
        # Hard top-k selection for forward pass
        # Note: torch.topk is non-differentiable, gradients won't flow through here
        _, selected_indices = torch.topk(noisy_scores, k, dim=-1)

        # ========== Step 5: Gather soft probabilities at selected positions ==========
        # This is key for gradient flow: gather connects hard selection with soft probs
        selected_probs = torch.gather(all_probs, dim=-1, index=selected_indices)

        return selected_indices, selected_probs, all_probs


# =============================================================================
# Core Module 3: SWOR Probability (Sampling Without Replacement Probability)
# =============================================================================

class SWORProbability(nn.Module):
    """
    Compute log probability of Sampling Without Replacement (SWOR).

    Paper Equation (5):
    -------------------
    p(d|x; R_φ) = Π_{i=1}^{k} p(d_i|x) / (1 - Σ_{j=1}^{i-1} p(d_j|x))

    Intuition:
    ----------
    Imagine drawing balls from a bag without replacement:
    - 1st ball probability = p(d_1)
    - 2nd ball conditional prob = p(d_2) / (1 - p(d_1))  [exclude already drawn]
    - 3rd ball conditional prob = p(d_3) / (1 - p(d_1) - p(d_2))
    - ...and so on

    Why SWOR?
    ---------
    RAG retrieval typically returns multiple non-repeating documents. If we
    assume independent sampling, we would overestimate the probability of
    selecting duplicate documents. SWOR more accurately models the actual
    retrieval process.

    Numerical Stability:
    --------------------
    - Use log probabilities to avoid underflow in probability products
    - Use clamp to prevent division by zero and log(0)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize SWOR probability calculator.

        Args:
            eps: Numerical stability constant
        """
        super().__init__()
        self.eps = eps

    def forward(self, selected_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute SWOR log probability.

        Args:
            selected_probs: [Batch, k]
                            Soft probabilities of selected documents

        Returns:
            log_prob: [Batch]
                      SWOR log probability for each sample

        Computation:
        ------------
        For position i:
        - Numerator: p(d_i)
        - Denominator: 1 - Σ_{j<i} p(d_j)  (sum of probs of all previously selected docs)
        - Conditional prob: p(d_i) / denominator
        - Sum log of conditional probs across all positions
        """
        batch_size, k = selected_probs.shape

        # ========== Compute cumulative probability ==========
        # cumsum: [p1, p1+p2, p1+p2+p3, ...]
        cumsum = torch.cumsum(selected_probs, dim=-1)

        # ========== Compute Σ_{j<i} p(d_j) ==========
        # Need to shift right by one: [0, p1, p1+p2, ...]
        # Use F.pad to pad 0 on the left, then remove last element
        sum_prev = F.pad(cumsum[:, :-1], (1, 0), value=0.0)

        # ========== Compute denominator: 1 - Σ_{j<i} p(d_j) ==========
        # Add clamp to prevent denominator from being too small
        denominator = torch.clamp(1.0 - sum_prev, min=self.eps)

        # ========== Compute conditional probabilities ==========
        # cond_prob[i] = selected_probs[i] / denominator[i]
        cond_probs = selected_probs / denominator

        # ========== Compute log probability ==========
        # log p(d|x) = Σ log(cond_probs)
        # Add clamp to prevent log(0)
        log_prob = torch.sum(
            torch.log(torch.clamp(cond_probs, min=self.eps)),
            dim=-1
        )

        return log_prob


# =============================================================================
# Core Module 4: KILT-Score Computer
# =============================================================================

class KILTScoreComputer:
    """
    KILT-Score Computer: Combines retrieval and generation quality metrics.

    Formula:
    --------
    KILT-Score = R-Precision × Task_Metric

    Where:
    - R-Precision: Fraction of gold passages retrieved
    - Task_Metric: Task-specific metric (EM, F1, Accuracy)

    Output Format Parsing:
    ----------------------
    FiD-Light output format: "index: 1,3 text: answer"
    - "index: 1,3" indicates model believes passages 1 and 3 contain the answer
    - "text: answer" is the generated answer

    We parse this format to:
    1. Extract predicted indices → Compute R-Precision
    2. Extract predicted answer → Compute Task_Metric

    Task Metrics (per KILT benchmark):
    ----------------------------------
    - nq, hotpotqa, triviaqa: Exact Match
    - fever: Accuracy
    - trex, structured_zeroshot: Exact Match (slot filling)
    - wow: Token F1 (dialogue)
    """

    def __init__(self, default_metric: str = "exact_match"):
        """
        Initialize KILT-Score calculator.

        Args:
            default_metric: Default task metric ("exact_match" or "f1")
        """
        self.default_metric = default_metric

        # Pre-compile regex for answer normalization
        import re
        self._article_re = re.compile(r'\b(a|an|the)\b')

        # Task -> Metric mapping
        self.task_metrics = {
            # Open-domain QA: Exact Match
            "nq": "exact_match",
            "hotpotqa": "exact_match",
            "triviaqa_support_only": "exact_match",
            # Fact Verification: Accuracy (same as EM for classification)
            "fever": "exact_match",
            # Slot Filling: Exact Match
            "trex": "exact_match",
            "structured_zeroshot": "exact_match",
            # Entity Linking: Accuracy
            "aidayago2": "exact_match",
            "cweb": "exact_match",
            "wned": "exact_match",
            # Dialogue: Token F1
            "wow": "f1",
            # Long-form QA: Token F1
            "eli5": "f1",
        }

    def parse_output(self, output_text: str) -> Tuple[List[int], str]:
        """
        Parse model output format.

        Args:
            output_text: Model-generated text
                         e.g., "index: 1,3 text: Paris"

        Returns:
            predicted_indices: Predicted document index list (1-based)
                               e.g., [1, 3]
            predicted_answer: Predicted answer text
                              e.g., "Paris"

        Format:
        -------
        Standard format: "index: {i1,i2,...} text: {answer}"
        - Indices separated by commas
        - Answer follows "text:"

        Error Handling:
        ---------------
        If format doesn't match, return empty index list and original text
        """
        # Try to match standard format: "index: 1,3 text: answer"
        pattern = r"index:\s*([\d,\s]+)\s*text:\s*(.*)"
        match = re.match(pattern, output_text.strip(), re.IGNORECASE)

        if match:
            # Parse indices
            indices_str = match.group(1)
            try:
                indices = [
                    int(x.strip())
                    for x in indices_str.split(",")
                    if x.strip().isdigit()
                ]
            except ValueError:
                indices = []

            # Extract answer
            answer = match.group(2).strip()
            return indices, answer

        # Format doesn't match, return empty indices and original text
        return [], output_text.strip()

    def compute_r_precision(
        self,
        predicted_indices: List[int],
        gold_indices: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute R-Precision.

        Formula:
        --------
        R-Precision = |predicted ∩ gold| / min(|predicted|, |gold|)

        Args:
            predicted_indices: Model-predicted document indices (1-based)
            gold_indices: Ground truth gold document indices (1-based)
            k: Optional, only consider top-k predictions

        Returns:
            r_precision: R-Precision score [0, 1]

        Edge Cases:
        -----------
        - If gold_indices is empty: Return 1.0 (no gold means any prediction is "correct")
        - If predicted_indices is empty: Return 0.0
        """
        if not gold_indices:
            # No gold passage case
            return 1.0 if not predicted_indices else 0.0

        if not predicted_indices:
            return 0.0

        # Only take top-k predictions
        pred = predicted_indices[:k] if k else predicted_indices

        # Compute intersection
        pred_set = set(pred)
        gold_set = set(gold_indices)
        intersection = len(pred_set & gold_set)

        # R-Precision denominator
        denominator = min(len(pred_set), len(gold_set))

        return intersection / denominator if denominator > 0 else 0.0

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match comparison."""
        import string
        # Lowercase
        s = s.lower()
        # Remove articles (a, an, the)
        s = self._article_re.sub(' ', s)
        # Remove punctuation
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        # Normalize whitespace
        return ' '.join(s.split())

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Compute Exact Match.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            1.0 if exact match (after normalization)
            0.0 otherwise
        """
        pred_norm = self._normalize_answer(prediction)
        gold_norm = self._normalize_answer(ground_truth)
        return 1.0 if pred_norm == gold_norm else 0.0

    def token_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute Token-level F1 score.

        Used for dialogue and other tasks requiring partial matching.

        Args:
            prediction: Predicted text
            ground_truth: Ground truth text

        Returns:
            f1_score: Token F1 score [0, 1]
        """
        # Tokenize (simple whitespace split)
        pred_tokens = set(prediction.lower().split())
        gold_tokens = set(ground_truth.lower().split())

        if not pred_tokens or not gold_tokens:
            return 0.0

        # Compute intersection
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0

        # Precision and Recall
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        # F1
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def compute_kilt_score(
        self,
        output_text: str,
        gold_answer: str,
        gold_indices: List[int],
        task: str = "nq"
    ) -> Tuple[float, float, float]:
        """
        Compute full KILT-Score.

        Args:
            output_text: Complete model output
                         e.g., "index: 1,3 text: Paris"
            gold_answer: Ground truth answer
            gold_indices: Ground truth gold document index list (1-based)
            task: Task name (determines which task metric to use)

        Returns:
            kilt_score: KILT-Score = R-Precision × Task_Metric
            r_precision: R-Precision score
            task_score: Task_Metric score

        Note:
            KILT-Score requires both retrieval and generation to be correct.
            If R-Precision = 0 or Task_Metric = 0,
            then KILT-Score = 0.
        """
        # Parse output
        pred_indices, pred_answer = self.parse_output(output_text)

        # Compute R-Precision
        r_precision = self.compute_r_precision(pred_indices, gold_indices)

        # Get task metric type
        metric_type = self.task_metrics.get(task, self.default_metric)

        # Compute task metric
        if metric_type == "f1":
            task_score = self.token_f1(pred_answer, gold_answer)
        else:  # exact_match
            task_score = self.exact_match(pred_answer, gold_answer)

        # KILT-Score = R-Precision × Task_Metric
        kilt_score = r_precision * task_score

        return kilt_score, r_precision, task_score


# =============================================================================
# Core Module 5: Utility Buffer (Offline Candidate Cache)
# =============================================================================

class UtilityBuffer:
    """
    Offline Utility Buffer: Pre-compute candidate outputs and utility values.

    Paper Method:
    -------------
    "at every N = 10,000 training steps, we run the RAG model that is being
    trained on the training inputs that will be used in the next N steps and
    use beam search to return 100 most probable outputs. We randomly sample
    m = 10 of these outputs to form Y."

    "Preparing Y for the next N training steps would also enable us to
    pre-compute utility values U(y, ŷ) : ∀ŷ ∈ Y"

    Advantages:
    -----------
    1. Very fast training - generation cost amortized over N steps
    2. Allows pre-computing utility values
    3. Ensures consistency of hard negatives

    Disadvantages:
    --------------
    1. Staleness - after model updates, cached candidates may no longer be hardest
    2. Memory usage - need to store N samples' candidates and utilities

    Parameters:
        buffer_size (int): Cache size (paper N=10,000)
        num_candidates (int): Candidates per sample (paper 100, then select m=10)
        num_samples (int): Actually used candidate count (m)
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        num_candidates: int = 100,
        num_samples: int = 10
    ):
        """
        Initialize buffer.

        Args:
            buffer_size: Number of samples to cache (N)
            num_candidates: Beam search candidate count
            num_samples: Actually used candidate count for training (m)
        """
        self.buffer_size = buffer_size
        self.num_candidates = num_candidates
        self.num_samples = num_samples

        # Cache data structure
        # data_id -> {
        #     "candidates": List[str],  # Candidate output texts
        #     "utilities": List[float],  # Pre-computed utility values
        #     "gold_answer": str,        # Ground truth answer
        #     "gold_indices": List[int], # Gold passage indices
        #     "task": str                # Task name
        # }
        self.cache: Dict[str, Dict[str, Any]] = {}

        # Cache state
        self.is_filled = False
        self.fill_step = -1  # Step when last filled

    def get(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache.

        Args:
            data_id: Data sample ID

        Returns:
            Cached candidates and utilities, or None if not exists
        """
        return self.cache.get(data_id)

    def put(
        self,
        data_id: str,
        candidates: List[str],
        utilities: List[float],
        gold_answer: str,
        gold_indices: List[int],
        task: str
    ) -> None:
        """
        Add data to cache.

        Args:
            data_id: Data sample ID
            candidates: Candidate output text list
            utilities: Pre-computed utility value list
            gold_answer: Ground truth answer
            gold_indices: Gold passage index list
            task: Task name
        """
        self.cache[data_id] = {
            "candidates": candidates,
            "utilities": utilities,
            "gold_answer": gold_answer,
            "gold_indices": gold_indices,
            "task": task
        }

    def sample_candidates(
        self,
        data_id: str,
        m: int,
        rng: np.random.Generator
    ) -> Tuple[List[str], List[float]]:
        """
        Randomly sample m candidates from cache.

        Paper: "We randomly sample m = 10 of these outputs to form Y"

        Args:
            data_id: Data sample ID
            m: Number of samples
            rng: Random number generator

        Returns:
            sampled_texts: Sampled candidate texts
            sampled_utilities: Corresponding utility values
        """
        entry = self.cache.get(data_id)
        if entry is None:
            return [], []

        candidates = entry["candidates"]
        utilities = entry["utilities"]

        if len(candidates) <= m:
            return candidates, utilities

        # Randomly sample m candidates
        indices = rng.choice(len(candidates), size=m, replace=False)
        sampled_texts = [candidates[i] for i in indices]
        sampled_utilities = [utilities[i] for i in indices]

        return sampled_texts, sampled_utilities

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.is_filled = False

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, data_id: str) -> bool:
        return data_id in self.cache


# =============================================================================
# Main Trainer: Stochastic RAG Trainer
# =============================================================================

class StochasticRAGTrainer:
    """
    Stochastic RAG Trainer.

    Extended from FiDLightTrainer, adding:
    - Differentiable reranking (StochasticReranker)
    - Gumbel-Top-k sampling (GumbelTopKSampler)
    - SWOR probability computation (SWORProbability)
    - KILT-Score utility function (KILTScoreComputer)

    Training Algorithm:
    -------------------
    for each batch:
        1. Encode 40 candidate passages → [B, 40, L, H]
        2. Re-ranker scores CLS vectors → [B, 40]
        3. Gumbel-Top-k samples k passages → [B, k]
        4. FiD-Light compression → [B, k×compression_k, H]
        5. Generator forward (teacher forcing) → L_gen
        6. Sample m outputs, compute average KILT-score → E[U]
        7. Compute SWOR log probability → log p(d|x)
        8. Combined loss: L = L_gen - E[U] × log p(d|x)
        9. Backward, update reranker + generator parameters
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize trainer.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(args.device)

        # Training state
        self.global_step = 0
        self.accumulated_steps = 0
        self.total_skipped = 0
        self.loss_history = []
        self.utility_history = []
        self.eval_history = []
        self.precomputed_val_data = {}  # task -> list of samples

        # Pre-compile regex for answer normalization
        import re
        self._article_re = re.compile(r'\b(a|an|the)\b')

        # Initialize components
        print(f"\n{'='*60}")
        print("Initializing Stochastic RAG Trainer")
        print(f"{'='*60}")

        self._init_model()
        self._init_reranker()
        self._init_stochastic_modules()
        self._init_optimizer()
        self._init_data()

        print(f"\n{'='*60}")
        print("Initialization Complete")
        print(f"{'='*60}")

    def _init_model(self) -> None:
        """Initialize T5 model and tokenizer."""
        print(f"\nLoading Generator: {self.args.model_name}...")

        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name)
        self.model.to(self.device)

        # Optional: Initialize from FiD-Light checkpoint
        if self.args.init_generator:
            print(f"  Loading pretrained weights from {self.args.init_generator}...")
            pretrained = T5ForConditionalGeneration.from_pretrained(
                self.args.init_generator
            )
            self.model.load_state_dict(pretrained.state_dict())
            del pretrained

        # Multi-GPU support with DataParallel
        self.n_gpu = torch.cuda.device_count()
        self.use_multi_gpu = getattr(self.args, 'multi_gpu', False) and self.n_gpu > 1

        if self.use_multi_gpu:
            print(f"  Using {self.n_gpu} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        else:
            print(f"  Using single GPU/CPU")

        # Parameter statistics (use base model for counting)
        base_model = self.get_base_model()
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        )
        print(f"  Generator parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

    def get_base_model(self):
        """Get the underlying model (handles DataParallel wrapper)."""
        if hasattr(self, 'use_multi_gpu') and self.use_multi_gpu:
            return self.model.module
        return self.model

    def _init_reranker(self) -> None:
        """Initialize differentiable reranker."""
        print(f"\nInitializing Stochastic Reranker...")

        # Get T5 hidden dimension (handle DataParallel)
        hidden_dim = self.get_base_model().config.d_model  # T5-base: 768

        self.reranker = StochasticReranker(
            hidden_dim=hidden_dim,
            scoring_type=self.args.scoring_type
        ).to(self.device)

        # Parameter statistics
        reranker_params = sum(p.numel() for p in self.reranker.parameters())
        print(f"  Scoring type: {self.args.scoring_type}")
        print(f"  Reranker parameters: {reranker_params:,}")

    def _init_stochastic_modules(self) -> None:
        """Initialize Stochastic RAG specific modules."""
        print(f"\nInitializing Stochastic RAG modules...")

        # Gumbel-Top-k sampler
        self.gumbel_sampler = GumbelTopKSampler(
            temperature=self.args.gumbel_tau
        )
        print(f"  Gumbel temperature (τ): {self.args.gumbel_tau}")

        # SWOR probability calculator
        self.swor = SWORProbability()

        # KILT-Score calculator
        self.kilt_scorer = KILTScoreComputer()
        print(f"  Utility function: KILT-Score")
        print(f"  Utility samples (m): {self.args.num_utility_samples}")

        # [Paper alignment] Offline Utility Buffer
        # Paper: "at every N = 10,000 training steps, we run the RAG model..."
        self.use_offline_buffer = getattr(self.args, 'use_offline_buffer', False)
        if self.use_offline_buffer:
            self.utility_buffer = UtilityBuffer(
                buffer_size=self.args.buffer_refresh_steps,
                num_candidates=self.args.buffer_num_candidates,
                num_samples=self.args.num_utility_samples
            )
            print(f"  Offline Buffer: ENABLED")
            print(f"    Buffer refresh: every {self.args.buffer_refresh_steps} steps")
            print(f"    Beam candidates: {self.args.buffer_num_candidates}")
        else:
            self.utility_buffer = None
            print(f"  Offline Buffer: DISABLED (using online generation)")

    def _init_optimizer(self) -> None:
        """
        Initialize optimizer.

        Uses Adafactor (paper setting), supports grouped learning rates:
        - Generator: lr_generator (default 1e-3)
        - Reranker: lr_reranker (default 1e-4)
        """
        print(f"\nInitializing Optimizer...")

        # Grouped learning rates
        param_groups = [
            {
                "params": self.model.parameters(),
                "lr": self.args.lr_generator,
                "name": "generator"
            },
            {
                "params": self.reranker.parameters(),
                "lr": self.args.lr_reranker,
                "name": "reranker"
            }
        ]

        # Adafactor (paper setting)
        self.optimizer = Adafactor(
            param_groups,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )

        print(f"  Optimizer: Adafactor")
        print(f"  Generator LR: {self.args.lr_generator}")
        print(f"  Reranker LR: {self.args.lr_reranker}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Effective batch size: "
              f"{self.args.batch_size * self.args.gradient_accumulation_steps}")

        # Optional: Learning rate scheduler
        if self.args.warmup_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps
            )
            print(f"  Warmup steps: {self.args.warmup_steps}")
        else:
            self.scheduler = None

    def _init_data(self) -> None:
        """
        Initialize data loading.

        Reuses FiD-Light's precomputed data format.
        """
        print(f"\nInitializing Data Pipeline...")

        if not HAS_PANDAS:
            raise ImportError("pandas is required. Install with: pip install pandas")

        path = self.args.precomputed_path

        if os.path.isdir(path):
            # Load all parquet files in directory
            files = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith('_train.parquet') and f != 'all_tasks_train.parquet'
            ])
            print(f"  Found {len(files)} task files in directory")

            dfs = []
            for fpath in files:
                print(f"    Loading {os.path.basename(fpath)}...", end=" ")
                df = pd.read_parquet(fpath, engine='fastparquet')
                dfs.append(df)
                print(f"{len(df):,} samples")

            combined_df = pd.concat(dfs, ignore_index=True)
            self.precomputed_data = {
                col: combined_df[col].tolist()
                for col in combined_df.columns
            }
        else:
            # Single file
            print(f"  Loading {path}...")
            df = pd.read_parquet(path, engine='fastparquet')
            self.precomputed_data = {
                col: df[col].tolist()
                for col in df.columns
            }

        # Organize indices by task (for temperature sampling)
        self.task_indices = defaultdict(list)
        for i, task in enumerate(self.precomputed_data["task"]):
            self.task_indices[task].append(i)

        # Compute temperature sampling probabilities
        # P_task ∝ N_task^(1/T)
        task_sizes = {
            task: len(indices)
            for task, indices in self.task_indices.items()
        }
        total_samples = sum(task_sizes.values())

        adjusted = {
            task: size ** (1 / self.args.temperature)
            for task, size in task_sizes.items()
        }
        total_adj = sum(adjusted.values())
        self.task_probs = {
            task: adj / total_adj
            for task, adj in adjusted.items()
        }

        print(f"\n  Total samples: {total_samples:,}")
        print(f"  Tasks: {len(self.task_indices)}")
        print(f"  Temperature (T): {self.args.temperature}")
        print(f"\n  Sampling probabilities:")
        for task, prob in sorted(self.task_probs.items(), key=lambda x: -x[1]):
            orig_prob = task_sizes[task] / total_samples
            print(f"    {task}: {prob:.4f} (original: {orig_prob:.4f})")

        # Random number generator
        self.rng = np.random.default_rng(
            self.args.seed if hasattr(self.args, 'seed') else 42
        )

        # Load validation data
        self._init_val_data()

    def _init_val_data(self) -> None:
        """
        Load precomputed validation data.

        If --precomputed_val_path not specified, automatically find *_dev.parquet in --precomputed_path directory.
        """
        # Determine validation data path
        if self.args.precomputed_val_path:
            path = self.args.precomputed_val_path
        elif os.path.isdir(self.args.precomputed_path):
            # Automatically find validation data in training data directory
            path = self.args.precomputed_path
        else:
            print("\n  No validation data found")
            return

        print(f"\n  Loading validation data from {path}...")

        if os.path.isdir(path):
            # Load all *_dev.parquet files
            for fname in sorted(os.listdir(path)):
                if fname.endswith('_dev.parquet'):
                    task = fname.replace('_dev.parquet', '')
                    fpath = os.path.join(path, fname)
                    df = pd.read_parquet(fpath, engine='fastparquet')
                    self.precomputed_val_data[task] = df.to_dict('records')
                    print(f"    {task}: {len(self.precomputed_val_data[task])} samples")
        else:
            # Single file
            df = pd.read_parquet(path, engine='fastparquet')
            for task in df['task'].unique():
                task_df = df[df['task'] == task]
                self.precomputed_val_data[task] = task_df.to_dict('records')
                print(f"    {task}: {len(self.precomputed_val_data[task])} samples")

        total_val = sum(len(samples) for samples in self.precomputed_val_data.values())
        print(f"  Total validation samples: {total_val:,}")

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of training data.

        Uses temperature sampling to balance different tasks.

        Args:
            batch_size: Batch size

        Returns:
            batch: List, each element contains:
                   - id: Sample ID
                   - task: Task name
                   - query: Query text
                   - answer: Answer
                   - input_texts: Formatted text for 40 candidate passages
                   - target_text: Target output
                   - matching_indices: Gold passage indices (1-based)
                   - gold_injected: Whether gold passage was injected
        """
        tasks = list(self.task_probs.keys())
        probs = [self.task_probs[t] for t in tasks]

        batch = []
        for _ in range(batch_size):
            # Sample task by probability
            task = self.rng.choice(tasks, p=probs)
            # Randomly sample index from task
            idx = self.rng.choice(self.task_indices[task])

            sample = {
                "id": self.precomputed_data["id"][idx],
                "task": self.precomputed_data["task"][idx],
                "query": self.precomputed_data["query"][idx],
                "answer": self.precomputed_data["answer"][idx],
                "input_texts": self.precomputed_data["input_texts"][idx],
                "target_text": self.precomputed_data["target_text"][idx],
                "matching_indices": self.precomputed_data["matching_indices"][idx],
                "gold_injected": self.precomputed_data.get("gold_injected", [False])[idx]
                    if "gold_injected" in self.precomputed_data else False,
            }
            batch.append(sample)

        return batch

    def fill_utility_buffer(self, num_samples: Optional[int] = None) -> None:
        """
        Fill Utility Buffer - Pre-compute candidate outputs and utilities for next N steps.

        Paper Method:
        -------------
        "at every N = 10,000 training steps, we run the RAG model that is being
        trained on the training inputs that will be used in the next N steps and
        use beam search to return 100 most probable outputs."

        Args:
            num_samples: Number of samples to pre-compute (default: buffer_refresh_steps)
        """
        if self.utility_buffer is None:
            return

        num_samples = num_samples or self.args.buffer_refresh_steps
        print(f"\n{'='*60}")
        print(f"Filling Utility Buffer (step {self.global_step})")
        print(f"{'='*60}")
        print(f"Generating {num_samples} samples with {self.args.buffer_num_candidates} candidates each...")

        self.model.eval()
        self.reranker.eval()
        self.utility_buffer.clear()

        # Pre-sample num_samples training samples
        sampled_indices = []
        tasks = list(self.task_probs.keys())
        probs = [self.task_probs[t] for t in tasks]

        for _ in range(num_samples):
            task = self.rng.choice(tasks, p=probs)
            idx = self.rng.choice(self.task_indices[task])
            sampled_indices.append(idx)

        # Process and generate candidates one by one
        processed = 0
        start_time = time.time()

        with torch.no_grad():
            for idx in tqdm(sampled_indices, desc="Filling buffer"):
                data_id = self.precomputed_data["id"][idx]

                # Skip already cached
                if data_id in self.utility_buffer:
                    continue

                sample = {
                    "id": data_id,
                    "task": self.precomputed_data["task"][idx],
                    "query": self.precomputed_data["query"][idx],
                    "answer": self.precomputed_data["answer"][idx],
                    "input_texts": self.precomputed_data["input_texts"][idx],
                    "target_text": self.precomputed_data["target_text"][idx],
                    "matching_indices": self.precomputed_data["matching_indices"][idx],
                }

                # Encode passages (simplified: only use top k)
                input_texts = sample["input_texts"][:self.args.num_passages]

                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=self.args.max_input_length,
                    truncation=True,
                    padding="max_length"
                ).to(self.device)

                # Encode (handle DataParallel)
                encoder = self.get_base_model().get_encoder()
                encoder_output = encoder(**inputs)
                hidden = encoder_output.last_hidden_state

                # FiD-Light compression
                k_comp = min(self.args.compression_k, hidden.shape[1])
                compressed = hidden[:, :k_comp, :].contiguous()
                fused = compressed.view(1, -1, compressed.shape[-1])
                fused_mask = torch.ones(1, fused.shape[1], device=self.device)

                # [Paper alignment] Beam Search to generate 100 candidates
                num_beams = self.args.buffer_num_candidates
                try:
                    outputs = self.get_base_model().generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        attention_mask=fused_mask,
                        max_length=self.args.max_output_length,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        do_sample=False,
                        early_stopping=True
                    )
                    candidates = self.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                except Exception as e:
                    # If beam search fails, use sampling as fallback
                    print(f"Warning: Beam search failed for {data_id}, using sampling")
                    outputs = self.get_base_model().generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        attention_mask=fused_mask,
                        max_length=self.args.max_output_length,
                        do_sample=True,
                        temperature=1.0,
                        num_return_sequences=min(num_beams, 20)
                    )
                    candidates = self.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )

                # Pre-compute utility
                gold_answer = sample["answer"]
                gold_indices = sample["matching_indices"]
                task = sample["task"]

                utilities = []
                for cand in candidates:
                    kilt_score, _, _ = self.kilt_scorer.compute_kilt_score(
                        output_text=cand,
                        gold_answer=gold_answer,
                        gold_indices=gold_indices,
                        task=task
                    )
                    utilities.append(kilt_score)

                # Store in buffer
                self.utility_buffer.put(
                    data_id=data_id,
                    candidates=candidates,
                    utilities=utilities,
                    gold_answer=gold_answer,
                    gold_indices=gold_indices,
                    task=task
                )

                processed += 1

        elapsed = time.time() - start_time
        self.utility_buffer.is_filled = True
        self.utility_buffer.fill_step = self.global_step

        print(f"Buffer filled: {len(self.utility_buffer)} samples")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Speed: {processed/elapsed:.1f} samples/sec")

        self.model.train()
        self.reranker.train()

    def training_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Execute one training step - Strictly following paper implementation.

        Implements complete Stochastic RAG training algorithm:
        1. Encode 40 candidate passages
        2. Re-ranker scoring
        3. Gumbel-Top-k sampling (paper Eq. 7)
        4. FiD-Light compression
        5. Generator forward (Per-sample loss with reduction='none')
        6. Sample m outputs, compute Expected Utility (force include GT=1.0)
        7. Compute per-sample SWOR Log Probability (paper Eq. 5)
        8. Per-sample combined loss: L = L_gen - U × log p(d|x)

        Args:
            batch: A batch of training samples

        Returns:
            metrics: Dictionary containing various losses and metrics
        """
        self.model.train()
        self.reranker.train()

        bsz = len(batch)
        n_candidates = self.args.n_candidates  # 40
        k = self.args.num_passages  # Number of documents to select
        compression_k = self.args.compression_k  # FiD-Light compression rate

        # ====================================================================
        # Step 1: Encode all 40 candidate passages
        # ====================================================================

        # Extract input texts: [bsz, 40] strings
        batch_input_texts = [sample["input_texts"] for sample in batch]

        # Flatten: [bsz * 40] strings
        all_texts = []
        for sample_texts in batch_input_texts:
            all_texts.extend(sample_texts[:n_candidates])  # Ensure only 40

        # Tokenize
        inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            max_length=self.args.max_input_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        seq_len = input_ids.shape[1]

        # Through T5 Encoder (handle DataParallel for multi-GPU)
        encoder = self.get_base_model().get_encoder()

        if self.use_multi_gpu and self.n_gpu > 1:
            # Use data_parallel for multi-GPU encoding
            encoder_wrapper = EncoderWrapper(encoder).to(self.device)
            device_ids = list(range(self.n_gpu))
            hidden_states = data_parallel(
                encoder_wrapper,
                (input_ids, attention_mask),
                device_ids=device_ids
            )
        else:
            encoder_output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = encoder_output.last_hidden_state  # [bsz*40, L, H]
        hidden_dim = hidden_states.shape[-1]

        # Reshape: [bsz*40, L, H] → [bsz, 40, L, H]
        hidden_states = hidden_states.view(bsz, n_candidates, seq_len, hidden_dim)
        attention_mask = attention_mask.view(bsz, n_candidates, seq_len)

        # ====================================================================
        # Step 2: Reranker scoring
        # ====================================================================

        # Extract first token of each passage (CLS-like)
        cls_vectors = hidden_states[:, :, 0, :]  # [bsz, 40, H]

        # Compute scores
        rerank_scores = self.reranker(cls_vectors)  # [bsz, 40]

        # ====================================================================
        # Step 3: Gumbel-Top-k sampling (Paper Eq. 7)
        # ====================================================================

        selected_indices, selected_probs, all_probs = self.gumbel_sampler(
            rerank_scores,
            k=k,
            tau=self.args.gumbel_tau
        )
        # selected_indices: [bsz, k] - Selected document indices (0-based, 0-39)
        # selected_probs: [bsz, k] - Soft probabilities (for gradients and SWOR)
        # all_probs: [bsz, 40] - Full softmax distribution

        # ====================================================================
        # Step 4: Select and compress passages
        # ====================================================================

        if self.args.use_full_st:
            # ========== Full Straight-Through Gumbel-Top-k ==========
            # Full ST: gradients flow to ALL N candidates weighted by softmax
            #
            # ST trick: forward uses hard selection, backward uses soft weights
            # Formula: output = soft + (hard - soft).detach()
            #          Forward: output ≈ hard (one-hot selection)
            #          Backward: ∂output/∂soft = 1 (gradients flow through soft)

            n_candidates = all_probs.shape[-1]  # N=40

            # One-hot hard selection: [bsz, k, N]
            hard_one_hot = F.one_hot(selected_indices, num_classes=n_candidates).float()

            # Soft weights from softmax: [bsz, k, N]
            # Each selected position i uses the full softmax distribution as soft weights
            soft_weights = all_probs.unsqueeze(1).expand(-1, k, -1)

            # ST trick: forward = hard, backward = soft
            st_weights = soft_weights + (hard_one_hot - soft_weights).detach()  # [bsz, k, N]

            # Weighted sum of hidden states: [bsz, k, L, H]
            # einsum: 'bkn,bnlh->bklh' = For each k position, weight N candidates' hidden states
            selected_hidden = torch.einsum('bkn,bnlh->bklh', st_weights, hidden_states)

            # Attention mask: use gather (mask has no gradients, ST not needed)
            mask_idx = selected_indices.unsqueeze(-1).expand(-1, -1, seq_len)
            selected_mask = torch.gather(
                attention_mask,
                dim=1,
                index=mask_idx
            )  # [bsz, k, L]
        else:
            # ========== Original gather-based selection ==========
            # Original: gradients only flow to k selected positions

            # Use gather to select hidden states of selected passages
            idx_expanded = selected_indices.unsqueeze(-1).unsqueeze(-1)
            idx_expanded = idx_expanded.expand(-1, -1, seq_len, hidden_dim)

            selected_hidden = torch.gather(
                hidden_states,
                dim=1,
                index=idx_expanded
            )  # [bsz, k, L, H]

            # Similarly select attention mask
            mask_idx = selected_indices.unsqueeze(-1).expand(-1, -1, seq_len)
            selected_mask = torch.gather(
                attention_mask,
                dim=1,
                index=mask_idx
            )  # [bsz, k, L]

        # FiD-Light compression: Take first compression_k vectors
        actual_k = min(compression_k, seq_len)
        compressed = selected_hidden[:, :, :actual_k, :].contiguous()  # [bsz, k, comp_k, H]
        comp_mask = selected_mask[:, :, :actual_k]  # [bsz, k, comp_k]

        # Reshape for decoder input: [bsz, k * comp_k, H]
        fused_hidden = compressed.view(bsz, k * actual_k, hidden_dim)
        fused_mask = comp_mask.reshape(bsz, k * actual_k)

        # ====================================================================
        # Step 5: Generator forward pass (Per-sample Loss)
        # ====================================================================

        target_texts = [sample["target_text"] for sample in batch]
        target_inputs = self.tokenizer(
            target_texts,
            return_tensors="pt",
            max_length=self.args.max_output_length,
            truncation=True,
            padding=True
        )
        labels = target_inputs["input_ids"].to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Create encoder_outputs object
        encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden)

        # Forward pass
        gen_outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=fused_mask,
            labels=labels
        )

        # Compute per-sample loss
        # Use reduction='none' then mean(dim=1)
        logits = gen_outputs.logits  # [bsz, seq_len, vocab]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_per_token = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss_per_token = loss_per_token.view(bsz, -1)  # [bsz, seq_len]

        # Use mean(dim=1)
        generator_loss_per_sample = loss_per_token.mean(dim=1)  # [bsz]

        # ====================================================================
        # Step 6: Sample m outputs, compute Expected Utility
        # Force include GT utility = 1.0
        # ====================================================================

        m = self.args.num_utility_samples

        # [Paper alignment] Offline Buffer mechanism
        # Paper: "at every N training steps, we run the RAG model...
        #        We randomly sample m=10 of these outputs to form Y"
        # If using Offline Buffer, prefer reading pre-computed candidates from buffer
        use_buffer_for_batch = (
            self.use_offline_buffer and
            self.utility_buffer is not None and
            self.utility_buffer.is_filled
        )

        # Check how many samples in batch are in buffer
        buffer_hits = []
        if use_buffer_for_batch:
            for sample in batch:
                data_id = sample.get("id", "")
                buffer_hits.append(data_id in self.utility_buffer)

        # Decide if online generation is needed
        need_online_generation = not use_buffer_for_batch or not all(buffer_hits)

        sampled_texts = None
        if need_online_generation:
            with torch.no_grad():
                # [Paper alignment] Hard Negative Sampling via Beam Search
                # Paper: "use beam search to return 100 most probable outputs.
                #        We randomly sample m=10 of these outputs to form Y"
                # Simplified: Directly use Beam Search to generate m highest probability candidates
                # These are the model's most "confident" outputs, most valuable Hard Negatives
                sampled_outputs = self.get_base_model().generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                    attention_mask=fused_mask,
                    max_length=self.args.max_output_length,
                    num_beams=m,              # Beam Search
                    num_return_sequences=m,    # Return m candidates
                    do_sample=False,           # Disable random sampling
                    early_stopping=True        # Stop when enough candidates found
                )
                sampled_texts = self.tokenizer.batch_decode(
                    sampled_outputs,
                    skip_special_tokens=True
                )

        # ====================================================================
        # Step 7 & 8: Per-sample SWOR and combined loss
        # ====================================================================

        total_loss = 0.0
        utilities = []

        for i in range(bsz):
            sample = batch[i]
            data_id = sample.get("id", "")
            gold_answer = sample["answer"]
            gold_indices = sample["matching_indices"]  # 1-based
            task = sample["task"]

            # ========== A. Compute Utility U(y, ŷ) ==========
            # Prefer sampling from Buffer, otherwise online generation
            if use_buffer_for_batch and buffer_hits[i]:
                # Sample m candidates and their pre-computed utilities from Buffer
                sampled_candidates, current_utils = self.utility_buffer.sample_candidates(
                    data_id=data_id,
                    m=m,
                    rng=self.rng
                )
            else:
                # Online generation: Get m sampled outputs for current sample
                start_idx = i * m
                end_idx = start_idx + m
                samples = sampled_texts[start_idx:end_idx]

                # Compute utility (KILT-score) for each sample
                current_utils = []
                for pred_text in samples:
                    kilt_score, r_prec, task_score = self.kilt_scorer.compute_kilt_score(
                        output_text=pred_text,
                        gold_answer=gold_answer,
                        gold_indices=gold_indices,
                        task=task
                    )
                    current_utils.append(kilt_score)

            # [Paper alignment] GT Sample Replacement Strategy
            # Paper: "we randomly replace one of the sampled outputs in Y with y"
            # Ensure GT is in the sample set, if not randomly replace one
            gt_in_samples = any(u >= 0.99 for u in current_utils)
            if not gt_in_samples:
                # Randomly select a position to replace with GT (utility = 1.0)
                replace_idx = self.rng.integers(0, len(current_utils))
                current_utils[replace_idx] = 1.0

            utils_arr = np.array(current_utils)
            avg_utility = torch.tensor(
                utils_arr.mean(),
                device=self.device,
                dtype=torch.float32
            )
            utilities.append(avg_utility.item())

            # ========== B. Compute SWOR Log Probability (Paper Eq. 5) ==========
            # Per-sample computation using dim=0
            probs_k = selected_probs[i]  # [k] - Soft probs of k selected docs for current sample

            # cumsum: [p1, p1+p2, p1+p2+p3, ...]
            cumsum_probs = torch.cumsum(probs_k, dim=0)

            # sum_prev: [0, p1, p1+p2, ...] - Sum of probs of all previously selected docs
            sum_prev = torch.cat([
                torch.zeros(1, device=self.device),
                cumsum_probs[:-1]
            ])

            # Denominator: 1 - sum_prev
            denominator = torch.clamp(1.0 - sum_prev, min=1e-6)

            # Conditional probability: p_i / (1 - sum_{j<i} p_j)
            cond_probs = probs_k / denominator

            # log p(d|x) = Σ log(cond_probs)
            log_p_retrieval = torch.sum(
                torch.log(torch.clamp(cond_probs, min=1e-9))
            )

            # ========== C. Per-sample Loss: L = L_gen - U * log p(d|x) ==========
            gen_loss = generator_loss_per_sample[i]
            loss_sample = gen_loss - (avg_utility * log_p_retrieval)

            total_loss += loss_sample

        # Average loss
        avg_loss = total_loss / bsz

        # Scale for gradient accumulation
        scaled_loss = avg_loss / self.args.gradient_accumulation_steps
        scaled_loss.backward()

        # Return metrics
        avg_utility = np.mean(utilities)
        return {
            "loss": avg_loss.item(),
            "gen_loss": generator_loss_per_sample.mean().item(),
            "utility": avg_utility,
            "log_swor": log_p_retrieval.item(),  # Last sample's value as reference
        }

    def optimizer_step(self) -> None:
        """Perform optimizer step."""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.reranker.parameters()),
            max_norm=self.args.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

    def train(self) -> None:
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Stochastic RAG Training")
        print(f"{'='*60}")
        print(f"Total steps: {self.args.total_steps:,}")
        print(f"Micro-batch size: {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"Effective batch size: "
              f"{self.args.batch_size * self.args.gradient_accumulation_steps}")
        print(f"Gumbel temperature (τ): {self.args.gumbel_tau}")
        print(f"Passages to select (k): {self.args.num_passages}")
        print(f"Compression rate: {self.args.compression_k}")
        print(f"Utility samples (m): {self.args.num_utility_samples}")
        print(f"Save every: {self.args.save_steps} steps")

        # [Paper alignment] Offline Buffer configuration
        if self.use_offline_buffer:
            print(f"Offline Buffer: ENABLED")
            print(f"  Buffer refresh every: {self.args.buffer_refresh_steps} steps")
            print(f"  Candidates per sample: {self.args.buffer_num_candidates}")
        else:
            print(f"Offline Buffer: DISABLED (online generation)")

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save configuration
        self._save_config()

        # Training loop
        start_time = time.time()
        accumulated_loss = 0.0
        accumulated_utility = 0.0
        pbar = tqdm(
            total=self.args.total_steps,
            desc="Training",
            initial=self.global_step
        )

        # [Paper alignment] Initial fill of Offline Buffer
        # Paper: "at every N = 10,000 training steps, we run the RAG model..."
        # Fill buffer first at step 0
        if self.use_offline_buffer:
            print("\n[Offline Buffer] Initial buffer fill at step 0...")
            self.fill_utility_buffer()

        # Track last buffer refresh step to avoid duplicate refreshes
        last_buffer_refresh_step = -1

        while self.global_step < self.args.total_steps:
            # [Paper alignment] Periodically refresh Offline Buffer
            # Refresh every buffer_refresh_steps, only once per step
            if (self.use_offline_buffer and
                self.global_step > 0 and
                self.global_step % self.args.buffer_refresh_steps == 0 and
                self.global_step != last_buffer_refresh_step):
                print(f"\n[Offline Buffer] Refreshing at step {self.global_step}...")
                self.fill_utility_buffer()
                last_buffer_refresh_step = self.global_step

            # Sample batch
            batch = self.sample_batch(self.args.batch_size)

            # Training step
            metrics = self.training_step(batch)

            accumulated_loss += metrics["loss"]
            accumulated_utility += metrics["utility"]
            self.accumulated_steps += 1

            # Gradient accumulation
            if self.accumulated_steps % self.args.gradient_accumulation_steps == 0:
                self.optimizer_step()

                # Record
                avg_loss = accumulated_loss / self.args.gradient_accumulation_steps
                avg_utility = accumulated_utility / self.args.gradient_accumulation_steps

                self.loss_history.append(avg_loss)
                self.utility_history.append(avg_utility)

                accumulated_loss = 0.0
                accumulated_utility = 0.0

                # Update progress bar (set_postfix before update to avoid double printing)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "E[U]": f"{avg_utility:.4f}",
                    "gen": f"{metrics['gen_loss']:.3f}",
                })
                pbar.update(1)

                # Report progress to web demo (every 10 steps)
                if self.global_step % 10 == 0:
                    report_training_progress(
                        self.global_step,
                        self.args.total_steps,
                        avg_loss,
                        self.args.learning_rate,
                        avg_utility
                    )

                # Evaluate (evaluate before save to ensure checkpoint contains current eval results)
                if self.global_step > 0 and self.global_step % self.args.eval_steps == 0:
                    self.evaluate()

                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()

        pbar.close()

        # Final save
        self.save_checkpoint(final=True)

        # Training summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Total steps: {self.global_step:,}")
        if self.loss_history:
            print(f"Final loss: {self.loss_history[-1]:.4f}")
        if self.utility_history:
            print(f"Final E[U]: {self.utility_history[-1]:.4f}")
        print(f"Checkpoints saved to: {self.args.output_dir}")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model.

        Uses precomputed validation data for evaluation, separated by task.
        """
        print(f"\nEvaluating at step {self.global_step}...")

        # Check if validation data exists
        if self.precomputed_val_data:
            return self._evaluate_precomputed()
        else:
            # No validation data, use quick evaluation on training data
            print("  No validation data, using training data sample...")
            return self._evaluate_training_sample()

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match comparison."""
        import string
        # Lowercase
        s = s.lower()
        # Remove articles (a, an, the)
        s = self._article_re.sub(' ', s)
        # Remove punctuation
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        # Normalize whitespace
        return ' '.join(s.split())

    def _evaluate_precomputed(self) -> Dict[str, float]:
        """Evaluate using precomputed validation data (per task)."""
        self.model.eval()
        self.reranker.eval()

        results = {}
        eval_samples_per_task = getattr(self.args, 'eval_samples', 100)

        with torch.no_grad():
            for task, samples in self.precomputed_val_data.items():
                task_samples = samples[:eval_samples_per_task]
                correct = 0
                total = 0

                for sample in tqdm(task_samples, desc=f"  {task}", leave=False):
                    input_texts = sample["input_texts"][:self.args.num_passages]
                    gold_answer = sample["answer"]

                    # Tokenize and encode
                    inputs = self.tokenizer(
                        input_texts,
                        return_tensors="pt",
                        max_length=self.args.max_input_length,
                        truncation=True,
                        padding="max_length"
                    ).to(self.device)

                    # Encode (handle DataParallel)
                    encoder = self.get_base_model().get_encoder()
                    encoder_output = encoder(**inputs)
                    hidden = encoder_output.last_hidden_state

                    # Compress
                    k_comp = min(self.args.compression_k, hidden.shape[1])
                    compressed = hidden[:, :k_comp, :].contiguous()
                    fused = compressed.view(1, -1, compressed.shape[-1])

                    # Generate
                    outputs = self.get_base_model().generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                        max_length=self.args.max_output_length,
                        num_beams=4
                    )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Extract answer (remove source pointer prefix)
                    if "text:" in generated:
                        pred_answer = generated.split("text:")[-1].strip()
                    else:
                        pred_answer = generated.strip()

                    # Exact match (normalized)
                    gold_norm = self._normalize_answer(gold_answer)
                    pred_norm = self._normalize_answer(pred_answer)
                    if gold_norm == pred_norm:
                        correct += 1
                    total += 1

                if total > 0:
                    em = correct / total
                    results[task] = em
                    print(f"  {task}: EM={em:.4f} ({correct}/{total})")

        # Compute average
        if results:
            avg_em = sum(results.values()) / len(results)
            print(f"  Average EM: {avg_em:.4f}")
            results["average"] = avg_em

        self.eval_history.append({"step": self.global_step, "results": results})

        self.model.train()
        self.reranker.train()

        return results

    def _evaluate_training_sample(self) -> Dict[str, float]:
        """When no validation data, randomly sample from training set for evaluation."""
        self.model.eval()
        self.reranker.eval()

        eval_samples = min(100, len(self.precomputed_data["id"]))

        total_kilt = 0.0
        total_r_prec = 0.0
        total_task = 0.0

        with torch.no_grad():
            for i in tqdm(range(eval_samples), desc="  Eval", leave=False):
                idx = self.rng.integers(0, len(self.precomputed_data["id"]))

                sample = {
                    "input_texts": self.precomputed_data["input_texts"][idx],
                    "answer": self.precomputed_data["answer"][idx],
                    "matching_indices": self.precomputed_data["matching_indices"][idx],
                    "task": self.precomputed_data["task"][idx],
                }

                input_texts = sample["input_texts"][:self.args.num_passages]

                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=self.args.max_input_length,
                    truncation=True,
                    padding="max_length"
                ).to(self.device)

                encoder = self.get_base_model().get_encoder()
                encoder_output = encoder(**inputs)
                hidden = encoder_output.last_hidden_state

                k_comp = min(self.args.compression_k, hidden.shape[1])
                compressed = hidden[:, :k_comp, :].contiguous()
                fused = compressed.view(1, -1, compressed.shape[-1])

                outputs = self.get_base_model().generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                    max_length=self.args.max_output_length,
                    num_beams=4
                )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                kilt, r_prec, task_score = self.kilt_scorer.compute_kilt_score(
                    generated,
                    sample["answer"],
                    sample["matching_indices"],
                    sample["task"]
                )

                total_kilt += kilt
                total_r_prec += r_prec
                total_task += task_score

        avg_kilt = total_kilt / eval_samples
        avg_r_prec = total_r_prec / eval_samples
        avg_task = total_task / eval_samples

        print(f"  KILT-Score: {avg_kilt:.4f}")
        print(f"  R-Precision: {avg_r_prec:.4f}")
        print(f"  Task Score: {avg_task:.4f}")

        self.eval_history.append({
            "step": self.global_step,
            "kilt_score": avg_kilt,
            "r_precision": avg_r_prec,
            "task_score": avg_task,
        })

        self.model.train()
        self.reranker.train()

        return {"kilt_score": avg_kilt, "r_precision": avg_r_prec}

    def save_checkpoint(self, final: bool = False) -> None:
        """Save checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        checkpoint_path = os.path.join(self.args.output_dir, suffix)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\nSaving checkpoint to {checkpoint_path}...")

        # Save model (handle DataParallel - use base model)
        self.get_base_model().save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save reranker
        torch.save(
            self.reranker.state_dict(),
            os.path.join(checkpoint_path, "reranker.pt")
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "accumulated_steps": self.accumulated_steps,
            "loss_history": self.loss_history,
            "utility_history": self.utility_history,
            "eval_history": self.eval_history,
            "args": vars(self.args),
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        # Save loss curve data
        if self.loss_history:
            np.save(
                os.path.join(checkpoint_path, "loss_history.npy"),
                np.array(self.loss_history)
            )
        if self.utility_history:
            np.save(
                os.path.join(checkpoint_path, "utility_history.npy"),
                np.array(self.utility_history)
            )

        # [Bug Fix] Save optimizer and scheduler state for proper resume
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        if self.scheduler is not None:
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(checkpoint_path, "scheduler.pt")
            )
        print(f"  Saved optimizer and scheduler state")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")

        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

        # Re-wrap with DataParallel if multi-GPU
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        # Load reranker
        reranker_path = os.path.join(checkpoint_path, "reranker.pt")
        if os.path.exists(reranker_path):
            self.reranker.load_state_dict(torch.load(reranker_path))
            self.reranker.to(self.device)

        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)

            self.global_step = state.get("global_step", 0)
            self.accumulated_steps = state.get("accumulated_steps", 0)
            self.loss_history = state.get("loss_history", [])
            self.utility_history = state.get("utility_history", [])
            self.eval_history = state.get("eval_history", [])

            print(f"  Resumed from step {self.global_step}")

        # Re-initialize optimizer (needed to create optimizer object)
        self._init_optimizer()

        # [Bug Fix] Load optimizer and scheduler state if available
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

        if os.path.exists(optimizer_path):
            # Load saved states (proper resume)
            print(f"  Loading optimizer state from {optimizer_path}...")
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            if os.path.exists(scheduler_path) and self.scheduler is not None:
                print(f"  Loading scheduler state from {scheduler_path}...")
                self.scheduler.load_state_dict(torch.load(scheduler_path))
            if self.scheduler is not None:
                print(f"  Current LR after resume: {self.scheduler.get_last_lr()[0]:.2e}")
        else:
            # Fallback: fast-forward scheduler (old checkpoints without optimizer state)
            print(f"  Warning: optimizer.pt not found, using fast-forward fallback")
            if self.global_step > 0 and self.scheduler is not None:
                print(f"  Fast-forwarding scheduler to step {self.global_step}...")
                for _ in range(self.global_step):
                    self.scheduler.step()
                print(f"  Current LR after resume: {self.scheduler.get_last_lr()[0]:.2e}")

    def _save_config(self) -> None:
        """Save training configuration."""
        config = {
            "method": "Stochastic RAG",
            "paper": "Zamani & Bendersky (SIGIR 2024)",
            "model_name": self.args.model_name,
            "n_candidates": self.args.n_candidates,
            "num_passages": self.args.num_passages,
            "compression_k": self.args.compression_k,
            "gumbel_tau": self.args.gumbel_tau,
            "num_utility_samples": self.args.num_utility_samples,
            "lr_generator": self.args.lr_generator,
            "lr_reranker": self.args.lr_reranker,
            "batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "effective_batch_size": (
                self.args.batch_size * self.args.gradient_accumulation_steps
            ),
            "total_steps": self.args.total_steps,
            "temperature": self.args.temperature,
        }

        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stochastic RAG Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===== Model =====
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="T5 model name")
    parser.add_argument("--init_generator", type=str, default=None,
                        help="Path to pretrained generator (e.g., FiD-Light checkpoint)")
    parser.add_argument("--scoring_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Reranker scoring network type")
    parser.add_argument("--use_full_st", action="store_true",
                        help="Use full Straight-Through Gumbel-Top-k (gradients flow to all N candidates)")

    # ===== Stochastic RAG Parameters =====
    parser.add_argument("--n_candidates", type=int, default=40,
                        help="Number of pre-retrieved candidates (N)")
    parser.add_argument("--num_passages", type=int, default=10,
                        help="Number of passages to select (k)")
    parser.add_argument("--compression_k", type=int, default=64,
                        help="FiD-Light compression: vectors per passage")
    parser.add_argument("--gumbel_tau", type=float, default=1.0,
                        help="Gumbel-Softmax temperature (τ)")
    parser.add_argument("--num_utility_samples", type=int, default=10,
                        help="Number of samples for expected utility (m)")

    # ===== Offline Buffer Parameters =====
    parser.add_argument("--use_offline_buffer", action="store_true",
                        help="Use offline utility buffer (paper method)")
    parser.add_argument("--buffer_refresh_steps", type=int, default=10000,
                        help="Refresh buffer every N steps (paper: 10000)")
    parser.add_argument("--buffer_num_candidates", type=int, default=100,
                        help="Beam search candidates per sample (paper: 100)")

    # ===== Input/Output =====
    parser.add_argument("--max_input_length", type=int, default=384,
                        help="Max input tokens per passage")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Max output tokens")

    # ===== Training =====
    parser.add_argument("--total_steps", type=int, default=50000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro-batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr_generator", type=float, default=1e-3,
                        help="Generator learning rate")
    parser.add_argument("--lr_reranker", type=float, default=1e-4,
                        help="Reranker learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # ===== Multi-task =====
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for task sampling (T)")

    # ===== Checkpointing =====
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/stochastic_rag",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=2500,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Samples per task for evaluation")

    # ===== Data =====
    parser.add_argument("--precomputed_path", type=str,
                        default="kilt_data/precomputed",
                        help="Path to precomputed retrieval data")
    parser.add_argument("--precomputed_val_path", type=str, default=None,
                        help="Path to precomputed validation data (directory with *_dev.parquet)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ===== Device =====
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use all available GPUs with DataParallel")

    # ===== Resume =====
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # ===== Quick Test =====
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test mode (100 steps)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps for quick test mode")

    args = parser.parse_args()

    # Quick test override
    if args.quick_test:
        args.total_steps = args.steps
        args.gradient_accumulation_steps = 4
        args.save_steps = 50
        args.eval_steps = 25
        print("Quick test mode enabled")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("Stochastic RAG Training")
    print("Paper: Zamani & Bendersky (SIGIR 2024)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Candidates (N): {args.n_candidates}")
    print(f"Selected passages (k): {args.num_passages}")
    print(f"Compression: {args.compression_k}")
    print(f"Gumbel τ: {args.gumbel_tau}")
    print(f"Utility samples (m): {args.num_utility_samples}")
    print(f"Training steps: {args.total_steps:,}")
    print(f"Device: {args.device}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print(f"Output: {args.output_dir}")

    # Initialize trainer
    trainer = StochasticRAGTrainer(args)

    # Resume training
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
