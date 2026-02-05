"""
Inference Demo Module for FiDLight Web Demo
=============================================

Provides an interactive inference engine for the Web UI.

Features:
- Support all 6 model types: FiD-Light, FiD Pure, Stochastic RAG (T5-base & T5Gemma2)
- Load trained model checkpoint with algorithm-specific handling
- Load GTR retriever with full Wikipedia index (supports finetuned retriever)
- Answer questions with source attribution
- Production mode only (no demo/mini-KB support)

Usage:
    from web_demo.inference_demo import InferenceEngine

    # FiD-Light T5-base
    engine = InferenceEngine(
        checkpoint_path="checkpoints/fidlight_paper/final",
        algorithm="fidlight",
        model_type="t5base",
        index_path="kilt_data/gtr_faiss_index_finetuned",
    )

    # Stochastic RAG T5Gemma2
    engine = InferenceEngine(
        checkpoint_path="checkpoints/stochastic_rag_t5gemma/final",
        algorithm="stochastic_rag",
        model_type="t5gemma",
        retriever_path="checkpoints/gtr_finetuned",
        index_path="kilt_data/gtr_faiss_index_finetuned",
    )

    result = engine.answer_question("Who is the president of the United States?")
"""

import json
import os
import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import (
        T5Tokenizer,
        T5ForConditionalGeneration,
        AutoProcessor,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:
    T5Tokenizer = None
    T5ForConditionalGeneration = None
    AutoProcessor = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    BaseModelOutput = None


# Algorithm configurations
ALGORITHM_CONFIGS = {
    "fidlight": {
        "num_passages": 40,
        "compression_k": 64,
        "has_source_pointer": True,
        "has_reranker": False,
        "input_format": "fidlight",  # "query: {Q} index: {i} context: {T} {P}"
    },
    "fid_pure": {
        "num_passages": 100,
        "compression_k": 250,
        "has_source_pointer": False,
        "has_reranker": False,
        "input_format": "fid_pure",  # "question: {Q} title: {T} context: {P}"
    },
    "stochastic_rag": {
        "num_passages": 40,  # candidates
        "n_selected": 10,    # after reranking
        "compression_k": 64,
        "has_source_pointer": True,
        "has_reranker": True,
        "input_format": "fidlight",
    },
}


@dataclass
class InferenceResult:
    """Result from inference engine."""
    query: str
    answer: str
    source_indices: List[int]
    passages: List[Dict[str, Any]]
    raw_output: str
    algorithm: str = "fidlight"
    model_type: str = "t5base"
    confidence: float = 0.0
    latency_ms: float = 0.0
    reranker_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "source_indices": self.source_indices,
            "passages": [
                {
                    "rank": p.get("rank", i + 1),
                    "title": p.get("title", ""),
                    "text": p.get("text", "")[:500],  # Truncate for display
                    "score": p.get("score", 0.0),
                    "is_source": (i + 1) in self.source_indices,
                    "reranker_score": self.reranker_scores[i] if self.reranker_scores and i < len(self.reranker_scores) else None,
                }
                for i, p in enumerate(self.passages)
            ],
            "raw_output": self.raw_output,
            "algorithm": self.algorithm,
            "model_type": self.model_type,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


class InferenceEngine:
    """
    Multi-algorithm inference engine for web demo.

    Supports:
    - FiD-Light (T5-base, T5Gemma2)
    - FiD Pure (T5-base, T5Gemma2)
    - Stochastic RAG (T5-base, T5Gemma2)
    """

    def __init__(
        self,
        checkpoint_path: str,
        algorithm: str = "fidlight",
        model_type: str = "t5base",
        retriever_path: str = None,
        index_path: str = None,
        wiki_arrow_path: str = None,
        compression_k: int = None,
        num_passages: int = None,
        n_selected: int = None,
        num_beams: int = 4,
        max_output_length: int = 64,
        max_input_length: int = 384,
        device: str = None,
        shared_retriever = None,
    ):
        """
        Initialize the inference engine.

        Args:
            checkpoint_path: Path to trained model checkpoint
            algorithm: "fidlight", "fid_pure", or "stochastic_rag"
            model_type: "t5base" or "t5gemma"
            retriever_path: Path to finetuned retriever (optional, uses default GTR if None)
            index_path: Path to Faiss index directory
            wiki_arrow_path: Path to Wikipedia Arrow dataset (optional)
            compression_k: Override compression factor (tokens per passage)
            num_passages: Override number of passages to retrieve
            n_selected: Override number of passages after reranking (SR only)
            num_beams: Beam search width
            max_output_length: Max tokens to generate
            max_input_length: Max tokens per passage
            device: 'cuda' or 'cpu'
            shared_retriever: Pre-loaded GTRRetriever instance to share (for Compare mode)
        """
        if T5ForConditionalGeneration is None:
            raise ImportError("transformers not installed")

        if algorithm not in ALGORITHM_CONFIGS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be one of {list(ALGORITHM_CONFIGS.keys())}")

        if model_type not in ["t5base", "t5gemma"]:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 't5base' or 't5gemma'")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.algorithm = algorithm
        self.model_type = model_type
        self.num_beams = num_beams
        self.max_output_length = max_output_length
        self.max_input_length = max_input_length
        self.is_loaded = False

        # Get algorithm-specific config (use as defaults)
        config = ALGORITHM_CONFIGS[algorithm]

        # Allow user overrides for parameters
        self.num_passages = num_passages if num_passages is not None else config["num_passages"]
        self.k = compression_k if compression_k is not None else config["compression_k"]
        self.n_selected = n_selected if n_selected is not None else config.get("n_selected", self.num_passages)

        # These come from config only
        self.has_source_pointer = config["has_source_pointer"]
        self.has_reranker = config["has_reranker"]
        self.input_format = config["input_format"]

        # Store paths for lazy loading
        self._checkpoint_path = checkpoint_path
        self._retriever_path = retriever_path
        self._index_path = index_path
        self._wiki_arrow_path = wiki_arrow_path

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.retriever = None
        self.reranker = None
        self.load_error = None  # Store last load error for UI display
        self._shared_retriever = shared_retriever  # Pre-loaded retriever for sharing

    def load(self) -> bool:
        """
        Load model and retriever.

        Returns:
            True if loaded successfully
        """
        if self.is_loaded:
            return True

        self.load_error = None

        try:
            # Check checkpoint path exists
            import os
            if not os.path.exists(self._checkpoint_path):
                self.load_error = f"Checkpoint path not found: {self._checkpoint_path}"
                print(self.load_error)
                return False

            # Load model based on model_type
            print(f"Loading {self.algorithm} model ({self.model_type}) from {self._checkpoint_path}...")

            if self.model_type == "t5gemma":
                self._load_t5gemma_model()
            else:
                self._load_t5base_model()

            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded on {self.device}")

            # Load reranker for Stochastic RAG
            if self.has_reranker:
                self._load_reranker()

            # Load GTR retriever (or use shared one)
            if self._shared_retriever is not None:
                print("Using shared retriever (pre-loaded)")
                self.retriever = self._shared_retriever
            elif self._index_path:
                self._load_retriever()

            self.is_loaded = True
            return True

        except Exception as e:
            self.load_error = str(e)
            print(f"Error loading inference engine: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_t5base_model(self):
        """
        Load T5-base model and tokenizer.
        Matches training/train_fidlight_paper.py and training/train_stochastic_rag.py load_checkpoint().
        """
        from pathlib import Path

        checkpoint_path = Path(self._checkpoint_path)
        self._diagnose_checkpoint(checkpoint_path)

        # Load tokenizer (same as training scripts)
        print(f"Loading T5Tokenizer from {checkpoint_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(str(checkpoint_path))

        # Load model (same as training scripts)
        print(f"Loading T5ForConditionalGeneration from {checkpoint_path}...")
        self.model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path))

    def _load_t5gemma_model(self):
        """
        Load T5Gemma2 model with special handling.
        Matches training/train_fidlight_t5gemma.py and training/train_stochastic_rag_t5gemma.py load_checkpoint().
        """
        from pathlib import Path

        checkpoint_path = Path(self._checkpoint_path)
        self._diagnose_checkpoint(checkpoint_path)

        # Load tokenizer first (same as training scripts: AutoTokenizer.from_pretrained(checkpoint_path))
        print(f"Loading AutoTokenizer from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        self.processor = None

        # Load model with bfloat16 (same as training scripts)
        print(f"Loading AutoModelForSeq2SeqLM with BF16 from {checkpoint_path}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16
        )

        # Set decoder_start_token_id to prevent gibberish (same as training scripts)
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.config.bos_token_id or 2
            print(f"Set decoder_start_token_id to {self.model.config.decoder_start_token_id}")

    def _diagnose_checkpoint(self, checkpoint_path):
        """Diagnose checkpoint directory for common issues."""
        from pathlib import Path

        print(f"\n{'='*50}")
        print(f"Checkpoint diagnosis: {checkpoint_path}")
        print(f"{'='*50}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        # List files in checkpoint
        files = list(checkpoint_path.iterdir())
        print(f"Files in checkpoint ({len(files)}):")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"  {f.name}: {size_mb:.1f} MB" if size_mb > 0 else f"  {f.name}/")

        # Check for required files
        required_files = {
            "config.json": "Model configuration",
            "tokenizer_config.json": "Tokenizer configuration",
        }
        model_files = ["model.safetensors", "pytorch_model.bin", "model.pt"]

        for req, desc in required_files.items():
            path = checkpoint_path / req
            if path.exists():
                print(f"✓ {req} found")
            else:
                print(f"✗ {req} MISSING - {desc}")

        # Check model file
        model_found = False
        for mf in model_files:
            path = checkpoint_path / mf
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {mf} found ({size_mb:.1f} MB)")
                model_found = True

                # Check for suspiciously small model file (likely corrupted)
                if size_mb < 10:
                    print(f"  ⚠ WARNING: Model file is very small, may be corrupted!")
                break

        if not model_found:
            print(f"✗ No model file found! Expected one of: {model_files}")
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")

        print(f"{'='*50}\n")

    def _get_encoder(self):
        """
        Get the encoder from model (handles T5 and T5Gemma2 differences).

        T5: model.encoder
        T5Gemma2: model.get_encoder() or model.encoder or model.model.encoder
        """
        # Try standard T5 encoder
        if hasattr(self.model, 'encoder') and callable(getattr(self.model.encoder, 'forward', None)):
            return self.model.encoder

        # Try get_encoder() method (some models)
        if hasattr(self.model, 'get_encoder'):
            return self.model.get_encoder()

        # Try nested model.model.encoder (T5Gemma2)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            return self.model.model.encoder

        raise AttributeError(
            f"Cannot find encoder in {type(self.model).__name__}. "
            f"Available attributes: {[a for a in dir(self.model) if not a.startswith('_')][:20]}"
        )

    def _load_reranker(self):
        """Load Stochastic RAG reranker from checkpoint."""
        reranker_path = Path(self._checkpoint_path) / "reranker.pt"

        if not reranker_path.exists():
            # Try alternative name
            reranker_path = Path(self._checkpoint_path) / "reranker_state_dict.pt"

        if reranker_path.exists():
            # Determine hidden dimension from model
            hidden_dim = self.model.config.d_model
            print(f"Loading reranker (hidden_dim={hidden_dim}) from {reranker_path}")

            self.reranker = nn.Linear(hidden_dim, 1)
            state_dict = torch.load(reranker_path, map_location=self.device)
            self.reranker.load_state_dict(state_dict)
            self.reranker.to(self.device)
            self.reranker.eval()
            print("Reranker loaded successfully")
        else:
            print(f"Warning: Reranker not found at {reranker_path}, using retriever scores only")
            self.reranker = None

    def _load_retriever(self):
        """Load GTR retriever with optional finetuned model."""
        from utils.gtr_retriever import GTRRetriever

        retriever_model = self._retriever_path
        print(f"Loading GTR retriever from {self._index_path}...")
        if retriever_model:
            print(f"Using finetuned retriever model: {retriever_model}")

        self.retriever = GTRRetriever(
            index_path=self._index_path,
            model_path=retriever_model,
            device=self.device,
            load_wiki=True,
            wiki_arrow_path=self._wiki_arrow_path,
        )

    def unload(self):
        """Unload model and retriever to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.reranker is not None:
            del self.reranker
            self.reranker = None

        if self.retriever is not None:
            del self.retriever
            self.retriever = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False

    def answer_question(self, question: str) -> InferenceResult:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question: The question to answer

        Returns:
            InferenceResult with answer, sources, and passages
        """
        import time
        start_time = time.time()

        # Ensure loaded
        if not self.is_loaded:
            self.load()

        # Step 1: Retrieve passages
        if self.retriever is None:
            return InferenceResult(
                query=question,
                answer="Error: No retriever loaded",
                source_indices=[],
                passages=[],
                raw_output="",
                algorithm=self.algorithm,
                model_type=self.model_type,
            )

        passages = self.retriever.retrieve(question, top_k=self.num_passages)

        # Step 2: Format passages based on algorithm
        input_texts = self._format_passages(passages, question)

        # Step 3: Generate answer (algorithm-specific)
        raw_output, reranker_scores = self._generate(input_texts)

        # Step 4: Parse output
        source_indices, answer = self._parse_output(raw_output)

        # Compute latency
        latency_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            query=question,
            answer=answer,
            source_indices=source_indices,
            passages=passages,
            raw_output=raw_output,
            algorithm=self.algorithm,
            model_type=self.model_type,
            latency_ms=latency_ms,
            reranker_scores=reranker_scores,
        )

    def _format_passages(self, passages: List[Dict], question: str) -> List[str]:
        """Format passages based on algorithm's input format."""
        input_texts = []

        for i, passage in enumerate(passages):
            title = passage.get("title", "")
            text = passage.get("text", "")

            if self.input_format == "fid_pure":
                # FiD Pure format: "question: {Q} title: {T} context: {P}"
                formatted = f"question: {question} title: {title} context: {text}"
            else:
                # FiD-Light format: "query: {Q} index: {i} context: {T} {P}"
                context = f"{title} {text}" if title else text
                formatted = f"query: {question} index: {i + 1} context: {context}"

            input_texts.append(formatted)

        return input_texts

    def _generate(self, input_texts: List[str]) -> Tuple[str, Optional[List[float]]]:
        """
        Generate answer using appropriate method for the algorithm.

        Returns:
            (generated_text, reranker_scores or None)
        """
        if self.algorithm == "stochastic_rag" and self.reranker is not None:
            return self._generate_stochastic_rag(input_texts)
        elif self.algorithm == "fid_pure":
            return self._generate_fid_pure(input_texts), None
        else:
            return self._generate_fidlight(input_texts), None

    def _generate_fidlight(self, input_texts: List[str]) -> str:
        """Generate answer using FiD-Light architecture (k=64, source pointer)."""
        n_passages = len(input_texts)

        # Tokenize all passages
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Encode all passages
        with torch.no_grad():
            encoder_outputs = self._get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            _, seq_len, hidden_dim = last_hidden_state.shape

            # Compress (take first k tokens)
            actual_k = min(self.k, seq_len)
            compressed_states = last_hidden_state[:, :actual_k, :]
            compressed_mask = attention_mask[:, :actual_k]

            # Fuse (reshape for batch_size=1)
            fused_hidden_states = compressed_states.reshape(1, n_passages * actual_k, hidden_dim)
            fused_attention_mask = compressed_mask.reshape(1, n_passages * actual_k)

            # Generate with model-type specific params
            outputs = self._generate_with_model(fused_hidden_states, fused_attention_mask)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def _generate_fid_pure(self, input_texts: List[str]) -> str:
        """Generate answer using FiD Pure architecture (k=250, no compression)."""
        n_passages = len(input_texts)

        # Tokenize all passages with longer max length for FiD Pure
        max_len = min(self.max_input_length, 250)  # FiD Pure uses 250 tokens
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Encode all passages
        with torch.no_grad():
            encoder_outputs = self._get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            _, seq_len, hidden_dim = last_hidden_state.shape

            # FiD Pure: use full passage (k=250, no real compression)
            actual_k = min(self.k, seq_len)
            compressed_states = last_hidden_state[:, :actual_k, :]
            compressed_mask = attention_mask[:, :actual_k]

            # Fuse
            fused_hidden_states = compressed_states.reshape(1, n_passages * actual_k, hidden_dim)
            fused_attention_mask = compressed_mask.reshape(1, n_passages * actual_k)

            # Generate
            outputs = self._generate_with_model(fused_hidden_states, fused_attention_mask)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def _generate_stochastic_rag(self, input_texts: List[str]) -> Tuple[str, List[float]]:
        """Generate answer using Stochastic RAG with reranker selection."""
        n_passages = len(input_texts)

        # Tokenize all passages
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            # Encode all passages
            encoder_outputs = self._get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_state = encoder_outputs.last_hidden_state  # [n_passages, seq_len, hidden]
            _, seq_len, hidden_dim = last_hidden_state.shape

            # Apply reranker to select top-k passages
            # Extract CLS vectors (first token)
            cls_vectors = last_hidden_state[:, 0, :]  # [n_passages, hidden]

            # Score with reranker
            reranker_output = self.reranker(cls_vectors)  # [n_passages, 1]
            scores = reranker_output.squeeze(-1)  # [n_passages]
            reranker_scores = scores.cpu().tolist()

            # Select top-k passages
            _, selected_indices = torch.topk(scores, min(self.n_selected, n_passages))
            selected_indices = selected_indices.sort().values  # Keep order

            # Get selected hidden states and masks
            selected_hidden = last_hidden_state[selected_indices]  # [n_selected, seq_len, hidden]
            selected_mask = attention_mask[selected_indices]  # [n_selected, seq_len]

            # Compress (take first k tokens)
            actual_k = min(self.k, seq_len)
            compressed_states = selected_hidden[:, :actual_k, :]
            compressed_mask = selected_mask[:, :actual_k]

            # Fuse
            n_selected = compressed_states.shape[0]
            fused_hidden_states = compressed_states.reshape(1, n_selected * actual_k, hidden_dim)
            fused_attention_mask = compressed_mask.reshape(1, n_selected * actual_k)

            # Generate
            outputs = self._generate_with_model(fused_hidden_states, fused_attention_mask)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text, reranker_scores

    def _generate_with_model(self, fused_hidden: torch.Tensor, fused_mask: torch.Tensor) -> torch.Tensor:
        """Generate with model-type specific parameters."""
        if self.model_type == "t5gemma":
            # T5Gemma2 needs special generation params to prevent gibberish
            outputs = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                attention_mask=fused_mask,
                max_new_tokens=self.max_output_length,
                num_beams=self.num_beams,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            # T5-base uses simpler generation
            outputs = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                attention_mask=fused_mask,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return outputs

    def _parse_output(self, output: str) -> Tuple[List[int], str]:
        """
        Parse model output to extract indices and answer.

        FiD-Light/SR format: "index: 1,3,5 text: Paris"
        FiD Pure format: "Paris" (just the answer)

        Returns:
            (list of indices, answer text)
        """
        if not self.has_source_pointer:
            # FiD Pure: output is just the answer
            return [], output.strip()

        # FiD-Light and Stochastic RAG: parse structured output
        indices = []
        answer = ""

        # Extract indices
        index_match = re.search(r"index:\s*([0-9,\s]+)", output)
        if index_match:
            indices_str = index_match.group(1)
            for idx_str in indices_str.split(","):
                try:
                    idx = int(idx_str.strip())
                    if 1 <= idx <= self.num_passages:
                        indices.append(idx)
                except ValueError:
                    continue

        # Extract answer text
        text_match = re.search(r"text:\s*(.+)", output, re.DOTALL)
        if text_match:
            answer = text_match.group(1).strip()
        else:
            # Fallback: if no "text:" found, use the whole output
            answer = output.strip()

        return indices, answer


# Singleton instance for web demo
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine(
    checkpoint_path: str = None,
    algorithm: str = "fidlight",
    model_type: str = "t5base",
    retriever_path: str = None,
    index_path: str = None,
    **kwargs,
) -> InferenceEngine:
    """
    Get or create the global inference engine instance.

    Args:
        checkpoint_path: Path to model checkpoint
        algorithm: "fidlight", "fid_pure", or "stochastic_rag"
        model_type: "t5base" or "t5gemma"
        retriever_path: Path to finetuned retriever (optional)
        index_path: Path to Faiss index
        **kwargs: Additional arguments for InferenceEngine

    Returns:
        InferenceEngine instance
    """
    global _inference_engine

    if _inference_engine is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required to create InferenceEngine")

        _inference_engine = InferenceEngine(
            checkpoint_path=checkpoint_path,
            algorithm=algorithm,
            model_type=model_type,
            retriever_path=retriever_path,
            index_path=index_path,
            **kwargs,
        )

    return _inference_engine


def reset_inference_engine():
    """Reset the global inference engine."""
    global _inference_engine
    if _inference_engine is not None:
        _inference_engine.unload()
        _inference_engine = None


# Demo function
def demo():
    """Demo the inference engine (Production mode)."""
    print("=" * 60)
    print("FiD-Light Inference Demo (Production)")
    print("=" * 60)

    # Production paths
    checkpoint_path = "checkpoints/fidlight_paper/final"
    index_path = "kilt_data/gtr_faiss_index_finetuned"

    # Check if paths exist
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return

    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}")
        print("Please build the GTR index first using data_pipeline/build_gtr_index.py")
        return

    # Initialize engine
    engine = InferenceEngine(
        checkpoint_path=checkpoint_path,
        algorithm="fidlight",
        model_type="t5base",
        index_path=index_path,
    )

    # Load model
    if not engine.load():
        print("Failed to load inference engine")
        return

    # Test questions
    questions = [
        "Who is the president of the United States?",
        "What is the capital of France?",
        "When was the Eiffel Tower built?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        result = engine.answer_question(question)

        print(f"Answer: {result.answer}")
        print(f"Sources: {result.source_indices}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"Raw output: {result.raw_output}")

        print("\nTop 3 passages:")
        for passage in result.passages[:3]:
            is_source = "[SOURCE]" if passage["rank"] in result.source_indices else ""
            print(f"  [{passage['rank']}] {passage['title']} (score={passage['score']:.3f}) {is_source}")


if __name__ == "__main__":
    demo()
