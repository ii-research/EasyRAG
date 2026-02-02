"""
Naive RAG Baseline Training Script with T5Gemma2
================================================

Trains a T5Gemma2 model with simple RAG: concatenate all passages into one context.

- Naive RAG: Retrieve passages, concatenate them, use standard encoder-decoder
- Different from FiD which encodes each passage separately then fuses

Input format:
    question: What is the capital of France?
    context: [1] France is a country in Western Europe. Its capital is Paris...
    [2] Paris is the largest city in France...

Output format:
    Paris

Usage:
    # Train from scratch
    python train_naive_rag_t5gemma.py \
        --model_name /path/to/T5Gemma2-270M-270M \
        --precomputed_path kilt_data/precomputed_v5/all_tasks_train.parquet \
        --output_dir checkpoints/naive_rag_t5gemma \
        --num_passages 10

    # Quick test
    python train_naive_rag_t5gemma.py --quick_test --steps 100 \
        --model_name /path/to/T5Gemma2-270M-270M \
        --precomputed_path kilt_data/precomputed_v5/all_tasks_train.parquet
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW

# Web Demo state reporting (optional)
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
                   "algorithm": "naive_rag", "model": "t5gemma"}
        )
    except Exception:
        pass


class NaiveRAGT5GemmaTrainer:
    """
    Naive RAG trainer with T5Gemma2 (concatenate passages, standard encoder-decoder).

    Key features:
    - T5Gemma2 backbone with Vision Tower frozen
    - All passages concatenated into single context string
    - Standard encoder-decoder (NOT Fusion-in-Decoder)
    - BF16 native support
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
        self.precomputed_val_data = {}

        # Initialize components
        self._init_model()
        self._init_optimizer()
        self._init_data()

    def _init_model(self) -> None:
        """Initialize T5Gemma2 model and processor."""
        print(f"\n{'='*60}")
        print("Initializing Naive RAG with T5Gemma2 (Concatenated Passages)")
        print(f"{'='*60}")

        model_name = self.args.model_name
        print(f"Loading {model_name}...")

        # Use AutoProcessor for T5Gemma2
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        # Load model with BF16
        if self.args.bf16:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            )
            print("  Loaded with BF16 precision")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("  Loaded with FP32 precision")

        # Fix decoder_start_token_id
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.config.bos_token_id or 2
            print(f"  Set decoder_start_token_id to {self.model.config.decoder_start_token_id}")

        # Freeze Vision Tower
        vision_tower = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            encoder = self.model.model.encoder
            if hasattr(encoder, 'vision_tower'):
                vision_tower = encoder.vision_tower

        if vision_tower is not None:
            frozen_params = 0
            for param in vision_tower.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            print(f"  Frozen Vision Tower: {frozen_params:,} parameters")

        self.model.to(self.device)

        # Multi-GPU support
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
        s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        return ' '.join(s.split())

    def _init_optimizer(self) -> None:
        """Initialize AdamW optimizer with Cosine decay."""
        print(f"\nInitializing AdamW optimizer...")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Micro-batch size: {self.args.batch_size}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        total_steps = self.args.total_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        print(f"  Using Cosine decay scheduler (total_steps={total_steps})")

    def _init_data(self) -> None:
        """Initialize data loaders."""
        print(f"\n{'='*60}")
        print("Initializing Data Pipeline (Naive RAG - Concatenated Passages)")
        print(f"{'='*60}")
        print(f"Num passages: {self.args.num_passages}")
        print(f"Max input length: {self.args.max_input_length}")

        self._init_precomputed_data()

    def _init_precomputed_data(self) -> None:
        """Load precomputed data for training."""
        import pandas as pd

        print(f"\nLoading precomputed data from {self.args.precomputed_path}...")

        path = self.args.precomputed_path
        if os.path.isdir(path):
            files = sorted([os.path.join(path, f) for f in os.listdir(path)
                           if f.endswith('_train.parquet') and f != 'all_tasks_train.parquet'])
            print(f"Found {len(files)} task files in directory")

            dfs = []
            for fpath in files:
                print(f"  Loading {os.path.basename(fpath)}...")
                df = pd.read_parquet(fpath, engine='fastparquet')
                dfs.append(df)
                print(f"    -> {len(df):,} samples")

            combined_df = pd.concat(dfs, ignore_index=True)
            self.precomputed_data = {col: combined_df[col].tolist() for col in combined_df.columns}
        else:
            df = pd.read_parquet(path, engine='fastparquet')
            self.precomputed_data = {col: df[col].tolist() for col in df.columns}

        # Organize by task for temperature sampling
        self.task_indices = defaultdict(list)
        for i, task in enumerate(self.precomputed_data["task"]):
            self.task_indices[task].append(i)

        # Compute temperature-adjusted sampling probabilities
        task_sizes = {task: len(indices) for task, indices in self.task_indices.items()}
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

        print(f"Loaded {total_samples:,} samples from {len(self.task_indices)} tasks")
        print(f"\nTemperature sampling (T={self.args.temperature}):")
        for task, prob in sorted(self.task_probs.items(), key=lambda x: -x[1]):
            orig_prob = task_sizes[task] / total_samples
            print(f"  {task}: {prob:.4f} (original: {orig_prob:.4f}, samples: {task_sizes[task]:,})")

        self.rng = np.random.default_rng(self.args.seed if hasattr(self.args, 'seed') else 42)

        if self.args.precomputed_val_path:
            self._load_validation_data()

    def _load_validation_data(self) -> None:
        """Load precomputed validation data."""
        import pandas as pd

        path = self.args.precomputed_val_path
        print(f"\nLoading validation data from {path}...")

        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.endswith('_dev.parquet'):
                    task = fname.replace('_dev.parquet', '')
                    fpath = os.path.join(path, fname)
                    df = pd.read_parquet(fpath, engine='fastparquet')
                    self.precomputed_val_data[task] = df.to_dict('records')
                    print(f"  {task}: {len(self.precomputed_val_data[task])} samples")
        else:
            df = pd.read_parquet(path, engine='fastparquet')
            for task in df['task'].unique():
                task_df = df[df['task'] == task]
                self.precomputed_val_data[task] = task_df.to_dict('records')
                print(f"  {task}: {len(self.precomputed_val_data[task])} samples")

        total_val = sum(len(samples) for samples in self.precomputed_val_data.values())
        print(f"Total validation samples: {total_val:,}")

    def _extract_context_from_input_text(self, input_text: str) -> str:
        """Extract context part from FiD-format input text."""
        import re
        match = re.search(r'context:\s*(.+)$', input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return input_text.split(':', 1)[-1].strip() if ':' in input_text else input_text

    def _format_naive_rag_input(self, query: str, input_texts: List[str]) -> str:
        """Format input for Naive RAG: concatenate passages into single context."""
        passages = input_texts[:self.args.num_passages]

        contexts = []
        for i, text in enumerate(passages):
            context = self._extract_context_from_input_text(text)
            contexts.append(f"[{i+1}] {context}")

        context_str = "\n".join(contexts)
        return f"question: {query}\ncontext: {context_str}"

    def sample_precomputed(self) -> Optional[Dict[str, Any]]:
        """Sample a training example with concatenated passages."""
        tasks = list(self.task_probs.keys())
        probs = [self.task_probs[t] for t in tasks]
        task = self.rng.choice(tasks, p=probs)

        indices = self.task_indices[task]
        idx = self.rng.choice(indices)

        query = self.precomputed_data["query"][idx]
        input_texts = self.precomputed_data["input_texts"][idx]
        answer = self.precomputed_data["answer"][idx]

        formatted_input = self._format_naive_rag_input(query, input_texts)

        return {
            "id": self.precomputed_data["id"][idx],
            "task": self.precomputed_data["task"][idx],
            "query": query,
            "answer": answer,
            "formatted_input": formatted_input,
        }

    def sample_precomputed_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of training examples."""
        batch = []
        for _ in range(batch_size):
            batch.append(self.sample_precomputed())
        return batch

    def training_step_batch(self, batch: List[Dict[str, Any]]) -> float:
        """Execute one training step with a batch."""
        self.model.train()

        input_texts = [sample['formatted_input'] for sample in batch]
        target_texts = [sample["answer"] for sample in batch]

        # Add EOS token to targets
        eos_token = self.tokenizer.eos_token or "</s>"
        target_texts_with_eos = [t + eos_token for t in target_texts]

        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.args.max_input_length,
            truncation=True,
            padding=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Tokenize targets
        target_inputs = self.tokenizer(
            target_texts_with_eos,
            return_tensors="pt",
            max_length=self.args.max_output_length,
            truncation=True,
            padding=True
        )
        labels = target_inputs["input_ids"].to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass with BF16
        if self.args.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / self.args.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.item() * self.args.gradient_accumulation_steps

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.args.max_grad_norm
        )

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

    def train(self) -> None:
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Naive RAG Training with T5Gemma2")
        print(f"{'='*60}")
        print(f"Total steps: {self.args.total_steps:,}")
        print(f"Num passages: {self.args.num_passages}")
        print(f"Max input length: {self.args.max_input_length}")
        print(f"Micro-batch size: {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        print(f"Save every: {self.args.save_steps} steps")
        print(f"Eval every: {self.args.eval_steps} steps")
        print(f"BF16: {self.args.bf16}")

        os.makedirs(self.args.output_dir, exist_ok=True)
        self._save_config()

        start_time = time.time()
        accumulated_loss = 0.0
        pbar = tqdm(total=self.args.total_steps, desc="Training", initial=self.global_step)

        while self.global_step < self.args.total_steps:
            batch = self.sample_precomputed_batch(self.args.batch_size)

            loss = self.training_step_batch(batch)
            accumulated_loss += loss
            self.accumulated_steps += 1

            if self.accumulated_steps % self.args.gradient_accumulation_steps == 0:
                self.optimizer_step()

                avg_loss = accumulated_loss / self.args.gradient_accumulation_steps
                self.loss_history.append(avg_loss)
                accumulated_loss = 0.0

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.get_lr():.2e}"
                })

                if self.global_step % 10 == 0:
                    report_training_progress(
                        self.global_step,
                        self.args.total_steps,
                        avg_loss,
                        self.get_lr()
                    )

                if self.global_step > 0 and self.global_step % self.args.eval_steps == 0:
                    self.evaluate()

                if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()

        pbar.close()
        self.save_checkpoint(final=True)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Total steps: {self.global_step:,}")
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

        self.get_base_model().save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        state = {
            "global_step": self.global_step,
            "accumulated_steps": self.accumulated_steps,
            "loss_history": self.loss_history,
            "eval_history": self.eval_history,
            "args": vars(self.args),
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        if self.loss_history:
            np.save(
                os.path.join(checkpoint_path, "loss_history.npy"),
                np.array(self.loss_history)
            )

        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        if self.scheduler is not None:
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(checkpoint_path, "scheduler.pt")
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training."""
        print(f"Loading checkpoint from {checkpoint_path}...")

        if self.args.bf16:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

        self.model.to(self.device)

        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        # Load processor from base model
        base_model_name = self.args.model_name
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        self.tokenizer = self.processor.tokenizer

        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)

            self.global_step = state.get("global_step", 0)
            self.accumulated_steps = state.get("accumulated_steps", 0)
            self.loss_history = state.get("loss_history", [])
            self.eval_history = state.get("eval_history", [])

            print(f"Resumed from step {self.global_step}")

        self._init_optimizer()

        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            if os.path.exists(scheduler_path) and self.scheduler is not None:
                self.scheduler.load_state_dict(torch.load(scheduler_path))

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation sets."""
        print(f"\nEvaluating at step {self.global_step}...")

        if not self.precomputed_val_data:
            print("  Skipping evaluation (no validation data)")
            return {}

        self.model.eval()
        results = {}

        for task, samples in self.precomputed_val_data.items():
            eval_samples = samples[:self.args.eval_samples]
            correct = 0
            total = 0

            with torch.no_grad():
                for sample in tqdm(eval_samples, desc=f"  {task}", leave=False):
                    query = sample["query"]
                    input_texts = sample["input_texts"]
                    gold_answer = sample["answer"].lower().strip()

                    formatted_input = self._format_naive_rag_input(query, input_texts)

                    inputs = self.tokenizer(
                        formatted_input,
                        return_tensors="pt",
                        max_length=self.args.max_input_length,
                        truncation=True
                    )
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)

                    outputs = self.get_base_model().generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.args.max_output_length,
                        num_beams=self.args.num_beams,
                        do_sample=False,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_answer = generated.lower().strip()

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

    def _save_config(self) -> None:
        """Save training configuration."""
        config = {
            "model_name": self.args.model_name,
            "model_type": "T5Gemma2",
            "algorithm": "naive_rag",
            "description": "Simple RAG - concatenate passages into single context",
            "num_passages": self.args.num_passages,
            "max_input_length": self.args.max_input_length,
            "max_output_length": self.args.max_output_length,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "warmup_steps": self.args.warmup_steps,
            "batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "effective_batch_size": self.args.batch_size * self.args.gradient_accumulation_steps,
            "total_steps": self.args.total_steps,
            "temperature": self.args.temperature,
            "num_beams": self.args.num_beams,
            "bf16": self.args.bf16,
        }

        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Naive RAG Baseline Training with T5Gemma2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to T5Gemma2 model")

    # RAG settings
    parser.add_argument("--num_passages", type=int, default=10,
                        help="Number of passages to concatenate")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Max input tokens")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Max output tokens")

    # Training
    parser.add_argument("--total_steps", type=int, default=10000,
                        help="Total training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")

    # Multi-task
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for task sampling")

    # Decoding
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam size for evaluation")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints/naive_rag_t5gemma",
                        help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Samples per task for evaluation")

    # Data
    parser.add_argument("--precomputed_path", type=str, required=True,
                        help="Path to precomputed training data")
    parser.add_argument("--precomputed_val_path", type=str, default=None,
                        help="Path to precomputed validation data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use all available GPUs")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 (default: True)")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable BF16")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Quick test
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test mode")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps for quick test")

    # Batch size
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro-batch size")

    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False

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
    print("Naive RAG Baseline Training with T5Gemma2")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Num passages: {args.num_passages}")
    print(f"Max input length: {args.max_input_length}")
    print(f"Training steps: {args.total_steps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Device: {args.device}")
    print(f"BF16: {args.bf16}")
    print(f"Output: {args.output_dir}")

    trainer = NaiveRAGT5GemmaTrainer(args)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
