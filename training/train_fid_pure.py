"""
Pure FiD (Fusion-in-Decoder) Training Script
=============================================

Reproduces the original FiD paper.

Paper: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
Authors: Izacard & Grave (2021)

Paper Settings:
- Model: T5-base (220M) or T5-large (770M)
- Input format: "question: {q} title: {title} context: {text}"
- Output: Answer text only (e.g., "Paris")
- Optimizer: Adam, lr=1e-4, constant
- Dropout: 10%
- Training: 10k steps, batch size 64
- Passages: 100 per query, 250 word pieces each
- Decoding: Greedy (beam=1)

Key differences from FiD-Light:
- No compression (k=250 = full passage)
- No source pointer prediction (answer-only target)
- No provenance metrics (only EM evaluation)

Usage:
    # Step 1: Precompute retrieval data (FiD format)
    python precompute_retrieval_for_fid.py --tasks all --output_dir kilt_data/precomputed_fid/
    python precompute_retrieval_for_fid.py --tasks all --split validation --output_dir kilt_data/precomputed_fid/

    # Step 2: Train (with FiD-format precomputed data)
    python train_fid_pure.py \\
        --precomputed_path kilt_data/precomputed_fid/all_tasks_train.parquet \\
        --precomputed_val_path kilt_data/precomputed_fid/

    # Quick test (100 steps)
    python train_fid_pure.py --quick_test --steps 100 \\
        --precomputed_path kilt_data/precomputed_fid/all_tasks_train.parquet

    # Resume from checkpoint
    python train_fid_pure.py --resume checkpoints/fid_pure/step_5000

    # Use fewer passages (40) with existing FiD-Light data (less accurate but faster)
    python train_fid_pure.py --num_passages 40 --max_input_length 384 \\
        --precomputed_path kilt_data/precomputed_v5/all_tasks_train.parquet
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
    from datasets import Dataset, concatenate_datasets
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # FiD paper uses Adam
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.parallel import data_parallel

# Local imports
from utils.multitask_loader import MultiTaskKILTLoader, prepare_training_sample, extract_answer
from utils.gtr_retriever import GTRRetriever

# Web Demo state reporting (optional - only used when running from web UI)
try:
    from web_demo.utils.state_io import update_step_state, StepStatus
    HAS_WEB_DEMO = True
except ImportError:
    HAS_WEB_DEMO = False


def report_training_progress(global_step: int, total_steps: int, loss: float, lr: float):
    """Report training progress to web UI if available."""
    if not HAS_WEB_DEMO:
        return
    try:
        progress = (global_step / total_steps) * 100
        message = f"Step {global_step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e}"
        update_step_state(
            step_name="train_model",
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra={"loss": loss, "lr": lr, "global_step": global_step,
                   "algorithm": "fid_pure", "model": "t5base"}
        )
    except Exception:
        pass  # Silently ignore errors to not disrupt training


class EncoderWrapper(nn.Module):
    """Wrapper for T5 encoder to work with data_parallel."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


class FiDPureTrainer:
    """
    Pure FiD trainer (no source pointer, no compression).

    Key features:
    - No compression (k = max_input_length)
    - Answer-only targets (no "index: X text: Y" format)
    - Multi-task training with temperature sampling
    - Gradient accumulation for effective batch size 128
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize trainer with config."""
        self.args = args
        self.device = torch.device(args.device)

        # Training state
        self.global_step = 0
        self.accumulated_steps = 0
        self.total_skipped = 0
        self.loss_history = []
        self.eval_history = []
        self.precomputed_val_data = {}  # task -> list of samples

        # Initialize components
        self._init_model()
        self._init_optimizer()
        self._init_data()

    def _init_model(self) -> None:
        """Initialize T5 model and tokenizer with dropout."""
        print(f"\n{'='*60}")
        print("Initializing Pure FiD Model (No Compression, No Source Pointer)")
        print(f"{'='*60}")

        model_name = self.args.model_name
        print(f"Loading {model_name}...")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Load model with custom dropout (FiD paper: dropout=0.1)
        from transformers import T5Config
        config = T5Config.from_pretrained(model_name)
        config.dropout_rate = self.args.dropout
        print(f"Setting dropout_rate = {self.args.dropout}")

        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.model.to(self.device)

        # Multi-GPU support with DataParallel
        self.n_gpu = torch.cuda.device_count()
        self.use_multi_gpu = self.args.multi_gpu and self.n_gpu > 1

        if self.use_multi_gpu:
            print(f"Using {self.n_gpu} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        else:
            print(f"Using single GPU/CPU")

        # Count parameters
        base_model = self.model.module if self.use_multi_gpu else self.model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_base_model(self):
        """Get the underlying model (handles DataParallel wrapper)."""
        if self.use_multi_gpu:
            return self.model.module
        return self.model

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match comparison."""
        import re
        import string
        # Remove articles
        s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
        # Remove punctuation
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        # Normalize whitespace
        return ' '.join(s.split())

    def _init_optimizer(self) -> None:
        """
        Initialize Adam optimizer (FiD paper specification).

        Paper: "using Adam with a constant learning rate of 10^-4"
        """
        print("\nInitializing Adam optimizer (FiD paper)...")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Dropout: {self.args.dropout}")
        print(f"  Micro-batch size: {self.args.batch_size}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        # Adam optimizer (FiD paper uses constant lr=1e-4)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.0  # Paper doesn't mention weight decay
        )

        # FiD paper uses constant learning rate, no scheduler
        # But we keep the option for warmup if needed
        if self.args.warmup_steps > 0:
            total_steps = self.args.total_steps
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None

    def _init_data(self) -> None:
        """Initialize data loaders and retriever."""
        print(f"\n{'='*60}")
        print("Initializing Data Pipeline")
        print(f"{'='*60}")

        # Check if using precomputed data
        self.use_precomputed = self.args.precomputed_path is not None

        if self.use_precomputed:
            self._init_precomputed_data()
        else:
            self._init_realtime_data()

    def _init_precomputed_data(self) -> None:
        """Load precomputed retrieval data for training."""
        import pandas as pd

        print(f"\nLoading precomputed data from {self.args.precomputed_path}...")

        # Support both single file and directory of files
        path = self.args.precomputed_path
        if os.path.isdir(path):
            # Load all *_train.parquet files from directory
            files = sorted([os.path.join(path, f) for f in os.listdir(path)
                           if f.endswith('_train.parquet') and f != 'all_tasks_train.parquet'])
            print(f"Found {len(files)} task files in directory")

            # Load and concatenate using pandas with fastparquet engine
            dfs = []
            for fpath in files:
                print(f"  Loading {os.path.basename(fpath)}...")
                df = pd.read_parquet(fpath, engine='fastparquet')
                dfs.append(df)
                print(f"    -> {len(df):,} samples")

            combined_df = pd.concat(dfs, ignore_index=True)
            self.precomputed_data = {col: combined_df[col].tolist() for col in combined_df.columns}
        else:
            # Single file
            df = pd.read_parquet(path, engine='fastparquet')
            self.precomputed_data = {col: df[col].tolist() for col in df.columns}

        # Organize by task for temperature sampling
        self.task_indices = defaultdict(list)
        for i, task in enumerate(self.precomputed_data["task"]):
            self.task_indices[task].append(i)

        # Compute temperature-adjusted sampling probabilities
        task_sizes = {task: len(indices) for task, indices in self.task_indices.items()}
        total_samples = sum(task_sizes.values())

        # P_task ∝ N_task^(1/T)
        adjusted = {
            task: size ** (1 / self.args.temperature)
            for task, size in task_sizes.items()
        }
        total_adj = sum(adjusted.values())
        self.task_probs = {
            task: adj / total_adj
            for task, adj in adjusted.items()
        }

        print(f"Loaded {total_samples:,} precomputed samples from {len(self.task_indices)} tasks")
        print(f"\nTemperature sampling (T={self.args.temperature}):")
        for task, prob in sorted(self.task_probs.items(), key=lambda x: -x[1]):
            orig_prob = task_sizes[task] / total_samples
            print(f"  {task}: {prob:.4f} (original: {orig_prob:.4f}, samples: {task_sizes[task]:,})")

        # Random generator for sampling
        self.rng = np.random.default_rng(self.args.seed if hasattr(self.args, 'seed') else 42)

        # No retriever needed for precomputed data
        self.retriever = None
        self.data_loader = None

        # Load validation data if path provided
        if self.args.precomputed_val_path:
            self._load_validation_data()

    def _load_validation_data(self) -> None:
        """Load precomputed validation data for evaluation."""
        import pandas as pd

        path = self.args.precomputed_val_path
        print(f"\nLoading precomputed validation data from {path}...")

        if os.path.isdir(path):
            # Load all *_dev.parquet files
            for fname in sorted(os.listdir(path)):
                if fname.endswith('_dev.parquet'):
                    task = fname.replace('_dev.parquet', '')
                    fpath = os.path.join(path, fname)
                    df = pd.read_parquet(fpath, engine='fastparquet')
                    self.precomputed_val_data[task] = df.to_dict('records')
                    print(f"  {task}: {len(self.precomputed_val_data[task])} samples")
        else:
            # Single parquet file
            df = pd.read_parquet(path, engine='fastparquet')
            for task in df['task'].unique():
                task_df = df[df['task'] == task]
                self.precomputed_val_data[task] = task_df.to_dict('records')
                print(f"  {task}: {len(self.precomputed_val_data[task])} samples")

        total_val = sum(len(samples) for samples in self.precomputed_val_data.values())
        print(f"Total validation samples: {total_val:,}")

    def _init_realtime_data(self) -> None:
        """Initialize real-time retrieval data pipeline (original behavior)."""
        # Multi-task data loader with temperature sampling
        print(f"\nLoading KILT tasks with T={self.args.temperature} sampling...")
        self.data_loader = MultiTaskKILTLoader(
            temperature=self.args.temperature,
            cache_dir=self.args.data_dir
        )

        # GTR retriever
        print("\nLoading GTR-T5-Base retriever...")
        self.retriever = GTRRetriever(
            index_path=self.args.index_path,
            device=self.args.device
        )

    def _transform_input_text(self, input_text: str) -> str:
        """
        Transform input text to pure FiD format (optional).

        Original FiD-Light format: "query: {query} index: {j+1} context: {context}"
        Pure FiD format: "question: {query} context: {context}"

        If strip_index is enabled, removes the "index: X" part.
        """
        if not self.args.strip_index:
            return input_text

        # Remove "index: X " part
        import re
        # Pattern: "query: ... index: X context: ..."
        # Replace with: "question: ... context: ..."
        transformed = re.sub(r'index: \d+ ', '', input_text)
        # Optionally rename "query:" to "question:" for original FiD format
        transformed = transformed.replace('query:', 'question:')
        return transformed

    def sample_precomputed(self) -> Optional[Dict[str, Any]]:
        """
        Sample a training example from precomputed data.

        Uses temperature-weighted task sampling (same as MultiTaskKILTLoader).

        KEY DIFFERENCE from FiD-Light:
        - Uses 'answer' field as target (not 'target_text')
        - target_text format "index: X text: Y" is NOT used

        Returns:
            Sample dict with input_texts, answer (as target), etc.
        """
        # Sample task according to temperature-adjusted probabilities
        tasks = list(self.task_probs.keys())
        probs = [self.task_probs[t] for t in tasks]
        task = self.rng.choice(tasks, p=probs)

        # Sample random index from task
        indices = self.task_indices[task]
        idx = self.rng.choice(indices)

        # Get input_texts and optionally transform
        input_texts = self.precomputed_data["input_texts"][idx]
        if self.args.strip_index:
            input_texts = [self._transform_input_text(t) for t in input_texts]

        # Build sample from precomputed data
        # KEY: Use 'answer' as target, NOT 'target_text'
        return {
            "id": self.precomputed_data["id"][idx],
            "task": self.precomputed_data["task"][idx],
            "query": self.precomputed_data["query"][idx],
            "answer": self.precomputed_data["answer"][idx],  # This IS the target!
            "input_texts": input_texts,
            "target_text": self.precomputed_data["answer"][idx],  # Pure answer, no index prefix
        }

    def sample_precomputed_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of training examples."""
        batch = []
        for _ in range(batch_size):
            batch.append(self.sample_precomputed())
        return batch

    def encode_and_fuse_batch(
        self,
        batch_input_texts: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and fuse a BATCH of samples (NO compression for pure FiD).

        Args:
            batch_input_texts: List of [List of 40 passages per query]
                              Shape: [batch_size, n_passages]

        Returns:
            fused_hidden: [batch_size, n_passages * seq_len, hidden_dim]
            fused_mask: [batch_size, n_passages * seq_len]
        """
        encoder = self.get_base_model().get_encoder()
        bsz = len(batch_input_texts)
        n_passages = len(batch_input_texts[0])  # 40

        # Flatten all passages: [bsz * n_passages] texts
        all_texts = []
        for sample_texts in batch_input_texts:
            all_texts.extend(sample_texts)

        # Tokenize all at once
        inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            max_length=self.args.max_input_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        passage_length = input_ids.shape[1]

        # Encode all passages - use data_parallel for multi-GPU
        if self.use_multi_gpu and self.n_gpu > 1:
            # Wrap encoder for data_parallel compatibility
            encoder_wrapper = EncoderWrapper(encoder).to(self.device)
            device_ids = list(range(self.n_gpu))
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            # data_parallel splits batch across GPUs automatically
            hidden_states = data_parallel(
                encoder_wrapper,
                (input_ids, attention_mask),
                device_ids=device_ids
            )
        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            encoder_output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = encoder_output.last_hidden_state
        hidden_dim = hidden_states.shape[-1]

        # Reshape: [bsz * n_passages, seq_len, hidden] -> [bsz, n_passages, seq_len, hidden]
        hidden_states = hidden_states.view(bsz, n_passages, passage_length, hidden_dim)
        attention_mask = attention_mask.view(bsz, n_passages, passage_length)

        # FiD-Light compression: first k vectors per passage
        # For PURE FiD, k = max_input_length (no compression)
        k = self.args.compression_k
        actual_k = min(k, passage_length)
        compressed = hidden_states[:, :, :actual_k, :].contiguous()
        comp_mask = attention_mask[:, :, :actual_k]

        # Reshape for decoder: [bsz, n_passages * k, hidden_dim]
        fused_hidden = compressed.reshape(bsz, n_passages * actual_k, hidden_dim)
        fused_mask = comp_mask.reshape(bsz, n_passages * actual_k)

        return fused_hidden, fused_mask

    def training_step_batch(self, batch: List[Dict[str, Any]]) -> float:
        """
        Execute one training step with a BATCH of samples.

        KEY DIFFERENCE from FiD-Light:
        - Target is just the answer (e.g., "Paris")
        - NOT "index: 1 text: Paris"

        Args:
            batch: List of prepared training samples

        Returns:
            Average loss value for this batch
        """
        self.model.train()

        # Extract input_texts and targets from batch
        batch_input_texts = [sample["input_texts"] for sample in batch]
        # Use 'answer' as target (NOT 'target_text' which has index prefix in FiD-Light)
        target_texts = [sample["answer"] for sample in batch]

        # Prepare targets (outside autocast)
        target_inputs = self.tokenizer(
            target_texts,
            return_tensors="pt",
            max_length=self.args.max_output_length,
            truncation=True,
            padding=True
        )
        labels = target_inputs["input_ids"].to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass with optional BF16 mixed precision
        if self.args.bf16:
            # BF16 for A100 (T5 is BF16-native, FP16 causes NaN)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fused_hidden, fused_mask = self.encode_and_fuse_batch(batch_input_texts)
                encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden)
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=fused_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / self.args.gradient_accumulation_steps
        else:
            # FP32 default
            fused_hidden, fused_mask = self.encode_and_fuse_batch(batch_input_texts)
            encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden)
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=fused_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass (BF16 doesn't need GradScaler like FP16)
        loss.backward()

        return loss.item() * self.args.gradient_accumulation_steps

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.args.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

    def train(self) -> None:
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Pure FiD Training (No Source Pointer)")
        print(f"{'='*60}")
        print(f"Total steps: {self.args.total_steps:,}")
        print(f"Micro-batch size: {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        print(f"Compression k: {self.args.compression_k} (384 = no compression)")
        print(f"Passages per query: {self.args.num_passages}")
        print(f"Save every: {self.args.save_steps} steps")
        print(f"Eval every: {self.args.eval_steps} steps")
        print(f"Strip index from input: {self.args.strip_index}")

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save config
        self._save_config()

        # Training loop
        start_time = time.time()
        accumulated_loss = 0.0
        pbar = tqdm(total=self.args.total_steps, desc="Training", initial=self.global_step)

        while self.global_step < self.args.total_steps:
            # Get batch of samples (precomputed or real-time)
            if self.use_precomputed:
                # Use precomputed data (no retrieval needed)
                batch = self.sample_precomputed_batch(self.args.batch_size)
            else:
                # Original behavior: real-time retrieval
                raw_batch = self.data_loader.sample_batch(batch_size=self.args.batch_size)
                if not raw_batch:
                    continue

                batch = []
                for raw_sample in raw_batch:
                    # Prepare sample with provenance verification
                    sample = prepare_training_sample(
                        raw_sample,
                        self.retriever,
                        num_passages=self.args.num_passages,
                        max_input_tokens=self.args.max_input_length
                    )
                    if sample is not None:
                        # For pure FiD, use answer as target
                        sample["target_text"] = sample["answer"]
                        batch.append(sample)
                    else:
                        self.total_skipped += 1

                if not batch:
                    continue

            # Training step with batch
            loss = self.training_step_batch(batch)
            accumulated_loss += loss
            self.accumulated_steps += 1

            # Gradient accumulation
            if self.accumulated_steps % self.args.gradient_accumulation_steps == 0:
                self.optimizer_step()

                # Record loss
                avg_loss = accumulated_loss / self.args.gradient_accumulation_steps
                self.loss_history.append(avg_loss)
                accumulated_loss = 0.0

                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "skipped": self.total_skipped,
                    "lr": f"{self.get_lr():.2e}"
                })

                # Evaluation (run before checkpoint so eval results are included)
                if self.global_step > 0 and self.global_step % self.args.eval_steps == 0:
                    self.evaluate()

                # Checkpointing
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
        print(f"Samples skipped: {self.total_skipped:,}")
        print(f"Final loss: {self.loss_history[-1]:.4f}" if self.loss_history else "")
        print(f"Checkpoints saved to: {self.args.output_dir}")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, final: bool = False) -> None:
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        checkpoint_path = os.path.join(self.args.output_dir, suffix)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\nSaving checkpoint to {checkpoint_path}...")

        # Save model and tokenizer (use base model for DataParallel)
        self.get_base_model().save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "accumulated_steps": self.accumulated_steps,
            "total_skipped": self.total_skipped,
            "loss_history": self.loss_history,
            "eval_history": self.eval_history,
            "args": vars(self.args),
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        # Save loss curve plot data
        if self.loss_history:
            np.save(
                os.path.join(checkpoint_path, "loss_history.npy"),
                np.array(self.loss_history)
            )

        # Save optimizer and scheduler state for proper resume
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
        """Load checkpoint and resume training."""
        print(f"Loading checkpoint from {checkpoint_path}...")

        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.model.to(self.device)

        # Re-wrap with DataParallel if multi-GPU
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)

            self.global_step = state.get("global_step", 0)
            self.accumulated_steps = state.get("accumulated_steps", 0)
            self.total_skipped = state.get("total_skipped", 0)
            self.loss_history = state.get("loss_history", [])
            self.eval_history = state.get("eval_history", [])

            print(f"Resumed from step {self.global_step}")

        # Reinitialize optimizer (needed to create optimizer object)
        self._init_optimizer()

        # Load optimizer and scheduler state if available
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

        if os.path.exists(optimizer_path):
            print(f"  Loading optimizer state from {optimizer_path}...")
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            if os.path.exists(scheduler_path) and self.scheduler is not None:
                print(f"  Loading scheduler state from {scheduler_path}...")
                self.scheduler.load_state_dict(torch.load(scheduler_path))
            if self.scheduler is not None:
                print(f"  Current LR after resume: {self.scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"  Warning: optimizer.pt not found, using fast-forward fallback")
            if self.global_step > 0 and self.scheduler is not None:
                print(f"  Fast-forwarding scheduler to step {self.global_step}...")
                for _ in range(self.global_step):
                    self.scheduler.step()
                print(f"  Current LR after resume: {self.scheduler.get_last_lr()[0]:.2e}")

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation sets.

        Pure FiD evaluation: only EM (Exact Match), no provenance metrics.

        Returns:
            Dict of metrics per task
        """
        print(f"\nEvaluating at step {self.global_step}...")

        # Check if we have precomputed validation data
        if self.precomputed_val_data:
            return self._evaluate_precomputed()
        elif self.use_precomputed:
            # Using precomputed training data but no validation data
            print("  Skipping evaluation (no precomputed validation data)")
            if self.loss_history:
                print(f"  Current training loss: {self.loss_history[-1]:.4f}")
            return {}
        else:
            # Fall back to real-time retrieval
            return self._evaluate_realtime()

    def _evaluate_precomputed(self) -> Dict[str, float]:
        """Evaluate using precomputed validation data."""
        self.model.eval()
        results = {}

        for task, samples in self.precomputed_val_data.items():
            eval_samples = samples[:self.args.eval_samples]
            correct = 0
            total = 0

            with torch.no_grad():
                for sample in tqdm(eval_samples, desc=f"  {task}", leave=False):
                    input_texts = sample["input_texts"]

                    # Optionally transform input
                    if self.args.strip_index:
                        input_texts = [self._transform_input_text(t) for t in input_texts]

                    gold_answer = sample["answer"].lower().strip()

                    # Encode and fuse (no compression for pure FiD)
                    fused_hidden, fused_mask = self.encode_and_fuse_batch([input_texts])

                    # Generate - pure FiD outputs answer directly
                    outputs = self.get_base_model().generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                        attention_mask=fused_mask,
                        max_length=self.args.max_output_length,
                        num_beams=self.args.num_beams
                    )

                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Pure FiD: generated text IS the answer (no extraction needed)
                    pred_answer = generated.lower().strip()

                    # Check exact match (normalized)
                    gold_norm = self._normalize_answer(gold_answer)
                    pred_norm = self._normalize_answer(pred_answer)
                    if gold_norm == pred_norm:
                        correct += 1
                    total += 1

            if total > 0:
                em = correct / total
                results[task] = em
                print(f"  {task}: EM={em:.4f} ({correct}/{total})")

        self.eval_history.append({"step": self.global_step, "results": results})
        self.model.train()
        return results

    def _evaluate_realtime(self) -> Dict[str, float]:
        """Evaluate using real-time retrieval (original behavior)."""
        if self.data_loader is None:
            print("  Skipping evaluation (no data loader)")
            return {}

        self.model.eval()
        results = {}
        tasks_to_eval = ["nq", "triviaqa_support_only"]  # Quick eval on 2 tasks

        for task in tasks_to_eval:
            val_samples = self.data_loader.get_validation_samples(
                task, n_samples=self.args.eval_samples
            )

            if not val_samples:
                continue

            correct = 0
            total = 0

            with torch.no_grad():
                for sample in val_samples[:self.args.eval_samples]:
                    sample["_task"] = task
                    prepared = prepare_training_sample(
                        sample, self.retriever,
                        num_passages=self.args.num_passages
                    )

                    if prepared is None:
                        continue

                    # Encode and fuse (batch of 1)
                    fused_hidden, fused_mask = self.encode_and_fuse_batch([prepared["input_texts"]])

                    # Generate
                    outputs = self.get_base_model().generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                        attention_mask=fused_mask,
                        max_length=self.args.max_output_length,
                        num_beams=self.args.num_beams
                    )

                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Pure FiD: generated text IS the answer
                    pred_answer = generated

                    # Check exact match
                    gold_answer = prepared["answer"]
                    gold_norm = self._normalize_answer(gold_answer)
                    pred_norm = self._normalize_answer(pred_answer)

                    if gold_norm == pred_norm:
                        correct += 1
                    total += 1

            if total > 0:
                em = correct / total
                results[task] = em
                print(f"  {task}: EM={em:.4f} ({correct}/{total})")

        self.eval_history.append({
            "step": self.global_step,
            "results": results
        })

        self.model.train()
        return results

    def _save_config(self) -> None:
        """Save training configuration."""
        config = {
            "model_name": self.args.model_name,
            "compression_k": self.args.compression_k,
            "max_input_length": self.args.max_input_length,
            "max_output_length": self.args.max_output_length,
            "num_passages": self.args.num_passages,
            "learning_rate": self.args.learning_rate,
            "batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "effective_batch_size": self.args.batch_size * self.args.gradient_accumulation_steps,
            "total_steps": self.args.total_steps,
            "temperature": self.args.temperature,
            "num_beams": self.args.num_beams,
            "strip_index": self.args.strip_index,
            "paper_reference": "Izacard & Grave (2021) - FiD (Pure, No Source Pointer)"
        }

        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pure FiD Training (Izacard & Grave, 2021)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="T5 model name (t5-base or t5-large)")
    parser.add_argument("--compression_k", type=int, default=250,
                        help="Vectors per passage (250 = paper setting, no compression)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (paper: 0.1)")

    # Input/Output (FiD paper settings)
    parser.add_argument("--max_input_length", type=int, default=250,
                        help="Max input tokens per passage (paper: 250 word pieces)")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Max output tokens")
    parser.add_argument("--num_passages", type=int, default=100,
                        help="Number of passages per query (paper: 100)")

    # Training (FiD paper settings)
    parser.add_argument("--total_steps", type=int, default=10000,
                        help="Total training steps (paper: 10k)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Gradient accumulation (paper: effective batch=64)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Adam learning rate (paper: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Learning rate warmup steps (paper: 0, constant lr)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Multi-task
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for task sampling")

    # Decoding (FiD paper uses greedy decoding)
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam size for evaluation (paper: 1 = greedy)")

    # Checkpointing (paper: eval every 500 steps, 10k total)
    parser.add_argument("--output_dir", type=str, default="checkpoints/fid_pure",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps (paper: 500)")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Samples per task for evaluation")

    # Data
    parser.add_argument("--data_dir", type=str, default="kilt_data",
                        help="KILT data directory")
    parser.add_argument("--index_path", type=str, default="kilt_data/gtr_faiss_index",
                        help="GTR Faiss index path")
    parser.add_argument("--precomputed_path", type=str, default=None,
                        help="Path to precomputed retrieval data (.parquet). "
                             "If provided, skips real-time retrieval.")
    parser.add_argument("--precomputed_val_path", type=str, default=None,
                        help="Path to precomputed validation data (directory with *_dev.parquet)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    # FiD-specific options
    parser.add_argument("--strip_index", action="store_true",
                        help="Strip 'index: X' from input texts for pure FiD format")

    # Device and Multi-GPU
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use all available GPUs with DataParallel")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision (recommended for A100, T5 is BF16-native)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Quick test
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test mode (100 steps)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps for quick test mode")

    # Batch processing
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro-batch size per GPU (increase to use more GPU memory)")

    args = parser.parse_args()

    # Quick test overrides
    if args.quick_test:
        args.total_steps = args.steps
        args.gradient_accumulation_steps = 4
        args.save_steps = 50
        args.eval_steps = 25
        args.eval_samples = 20
        print("Quick test mode enabled")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("Pure FiD Training (Izacard & Grave, 2021)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Compression k: {args.compression_k} (384 = no compression = original FiD)")
    print(f"Passages: {args.num_passages}")
    print(f"Training steps: {args.total_steps:,}")
    print(f"Micro-batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Device: {args.device}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print(f"BF16 mixed precision: {args.bf16}")
    print(f"Strip index from input: {args.strip_index}")
    print(f"Output: {args.output_dir}")
    if args.precomputed_path:
        print(f"Using precomputed data: {args.precomputed_path}")
    else:
        print("Using real-time retrieval (slower)")

    # Initialize trainer
    trainer = FiDPureTrainer(args)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
