"""
GTR Retriever Fine-Tuning Script
================================

Fine-tunes GTR-T5-Base on KILT training data.

Data Split (Academic Rigor):
- Automatically splits input data: 90% train, 5% dev, 5% test
- Dev set: Used for evaluation during training (model NEVER sees during training)
- Test set: Held out for final evaluation (saved to output_dir/data_splits/)
- Split is deterministic (controlled by --seed) for reproducibility
- This ensures reported metrics reflect true generalization, not memorization

Supports two loss functions:
1. inbatch_softmax (default, GTR paper): In-batch sampled softmax loss with hard negatives
   - Uses cosine similarity with temperature
   - Treats all samples in batch as negatives (in-batch negatives)
   - Reference: "Large Dual Encoders Are Generalizable Retrievers" (Ni et al., 2021)

2. triplet: TripletLoss with cosine distance
   - Only uses explicit negative from each triplet
   - Requires fp32 for numerical stability
   - Reference: FiD-Light Appendix B

GTR paper training settings:
- batch_size=2048 on TPU
- temperature=0.01
- learning_rate=1e-3 with Adafactor
- 20K fine-tuning steps

Usage:
    # GTR paper style (recommended)
    python train_gtr_retriever.py \
        --train_data kilt_data/retrieval_training_data.jsonl \
        --output_dir checkpoints/gtr_kilt_finetuned \
        --loss_type inbatch_softmax \
        --temperature 0.01 \
        --learning_rate 1e-3 \
        --batch_size 64 \
        --gradient_accumulation_steps 3 \
        --steps 10000 \
        --no_bf16

    # TripletLoss style (alternative)
    python train_gtr_retriever.py \
        --train_data kilt_data/retrieval_training_data.jsonl \
        --output_dir checkpoints/gtr_kilt_finetuned \
        --loss_type triplet \
        --no_bf16 \
        --learning_rate 5e-4 \
        --batch_size 64 \
        --gradient_accumulation_steps 3 \
        --steps 10000

Output structure:
    output_dir/
    ├── training_config.json      # Full config with data split info
    ├── data_splits/
    │   ├── dev.jsonl             # Dev set (5%)
    │   └── test.jsonl            # Test set (5%) - for final evaluation
    ├── checkpoint-XXXX/          # Model checkpoints
    └── training_log.csv          # Training metrics
"""

import argparse
import json
import os
import random
import csv
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from tqdm import tqdm
from transformers import Adafactor

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator


class TripletDataset(Dataset):
    """Dataset for triplet training examples."""

    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_training_data(data_path: str, max_samples: int = None) -> List[InputExample]:
    """
    Load training triples from jsonl file.

    Args:
        data_path: Path to jsonl file with triples
        max_samples: Max samples to load (None = all)

    Returns:
        List of InputExample for sentence-transformers
    """
    examples = []

    print(f"Loading training data from {data_path}...")
    with open(data_path, "r") as f:
        for line in tqdm(f, desc="Loading"):
            if max_samples and len(examples) >= max_samples:
                break

            data = json.loads(line)
            # InputExample for TripletLoss: (anchor, positive, negative)
            example = InputExample(
                texts=[data["query"], data["positive"], data["negative"]]
            )
            examples.append(example)

    print(f"Loaded {len(examples)} training examples")
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GTR retriever on KILT data"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training triples (jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/gtr_kilt_finetuned",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="sentence-transformers/gtr-t5-base",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps (paper: 100K)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Learning rate (paper: 0.005)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5000,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective_batch = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max training samples (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="inbatch_softmax",
        choices=["inbatch_softmax", "triplet"],
        help="Loss function: 'inbatch_softmax' (GTR paper, recommended) or 'triplet' (FiD-Light)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Softmax temperature for inbatch_softmax loss (GTR paper: 0.01)"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision (recommended for A100 + T5)"
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bf16, use fp32"
    )

    args = parser.parse_args()

    # Handle bf16 flag
    args.use_bf16 = args.bf16 and not args.no_bf16
    if args.use_bf16 and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 not supported on this GPU, falling back to FP32")
        args.use_bf16 = False

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    print("=" * 60)
    print("GTR Retriever Fine-Tuning")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Eval every: {args.eval_steps} steps")
    print(f"Save every: {args.save_steps} steps")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == "inbatch_softmax":
        print(f"Temperature: {args.temperature}")
    print(f"Precision: {'bf16' if args.use_bf16 else 'fp32'}")
    print()

    # Load model
    print("Loading base model...")
    model = SentenceTransformer(args.base_model)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Multi-GPU support using custom DataParallel wrapper
    n_gpus = torch.cuda.device_count()
    use_multi_gpu = n_gpus > 1

    class DataParallelWithConfig(torch.nn.DataParallel):
        """DataParallel wrapper that preserves model attributes like config."""
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    if use_multi_gpu:
        print(f"Using {n_gpus} GPUs with DataParallel")
        # Wrap the underlying transformer model
        original_model = model._first_module().auto_model
        model._first_module().auto_model = DataParallelWithConfig(original_model)

    # Load all data and perform train/dev/test split
    # This is critical for academic rigor - model must never see dev/test during training
    all_examples = load_training_data(args.train_data, args.max_samples)

    if not all_examples:
        print("Error: No training examples loaded!")
        return

    # Shuffle with fixed seed for reproducibility
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Split: 90% train, 5% dev, 5% test
    total_samples = len(all_examples)
    dev_size = int(total_samples * 0.05)
    test_size = int(total_samples * 0.05)
    train_size = total_samples - dev_size - test_size

    train_examples = all_examples[:train_size]
    dev_examples = all_examples[train_size:train_size + dev_size]
    test_examples = all_examples[train_size + dev_size:]

    print(f"\n{'='*60}")
    print("Data Split (for academic rigor):")
    print(f"  Total samples:  {total_samples}")
    print(f"  Train samples:  {len(train_examples)} (90%) - Model learns from these")
    print(f"  Dev samples:    {len(dev_examples)} (5%)  - Used for evaluation during training")
    print(f"  Test samples:   {len(test_examples)} (5%)  - Held out for final evaluation")
    print(f"  Random seed:    {args.seed}")
    print(f"{'='*60}\n")

    # Save dev and test splits for later analysis and final evaluation
    splits_dir = os.path.join(args.output_dir, "data_splits")
    os.makedirs(splits_dir, exist_ok=True)

    def save_examples_to_jsonl(examples, filepath):
        """Save InputExamples to JSONL format."""
        with open(filepath, 'w') as f:
            for ex in examples:
                record = {
                    "query": ex.texts[0],
                    "positive": ex.texts[1],
                    "negative": ex.texts[2]
                }
                f.write(json.dumps(record) + '\n')

    save_examples_to_jsonl(dev_examples, os.path.join(splits_dir, "dev.jsonl"))
    save_examples_to_jsonl(test_examples, os.path.join(splits_dir, "test.jsonl"))
    print(f"Saved dev and test splits to {splits_dir}/")

    # Custom collate function for InputExample
    def collate_fn(batch):
        """Just return the list of InputExample as-is."""
        return batch

    # Create DataLoader
    train_dataloader = DataLoader(
        TripletDataset(train_examples),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # Calculate epochs needed for target steps
    steps_per_epoch = len(train_dataloader)
    num_epochs = (args.steps + steps_per_epoch - 1) // steps_per_epoch

    print(f"\nTraining setup:")
    print(f"  Total examples: {len(train_examples)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs needed: {num_epochs}")
    print(f"  Total steps: {num_epochs * steps_per_epoch}")
    print()

    # Create evaluator using dev set (model has NEVER seen these during training)
    # This gives us true generalization performance, not memorization
    evaluator = TripletEvaluator.from_input_examples(
        dev_examples,
        name="dev_set"
    )
    print(f"Evaluator using {len(dev_examples)} held-out dev samples")

    # Define loss: TripletLoss (per user selection and paper)
    train_loss = losses.TripletLoss(model)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config with data split information
    config = {
        "base_model": args.base_model,
        "train_data": args.train_data,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        # Data split information (critical for reproducibility)
        "data_split": {
            "total_samples": total_samples,
            "train_samples": len(train_examples),
            "dev_samples": len(dev_examples),
            "test_samples": len(test_examples),
            "train_ratio": 0.90,
            "dev_ratio": 0.05,
            "test_ratio": 0.05,
            "random_seed": args.seed,
            "split_files": {
                "dev": "data_splits/dev.jsonl",
                "test": "data_splits/test.jsonl"
            }
        },
        "optimizer": "Adafactor",
        "loss": args.loss_type,
        "temperature": args.temperature if args.loss_type == "inbatch_softmax" else None,
        "precision": "bf16" if args.use_bf16 else "fp32",
        "paper_reference": "GTR (Ni et al., 2021)" if args.loss_type == "inbatch_softmax" else "FiD-Light (Hofstatter et al., 2023)",
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Starting training...")
    print("=" * 60)

    # Custom training loop with gradient accumulation
    device = model.device

    # Setup Adafactor optimizer (per GTR and FiD-Light papers)
    # scale_parameter=False: use fixed learning rate instead of relative step
    # relative_step=False: disable automatic LR scheduling
    # warmup_init=False: we handle warmup manually
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # Setup learning rate scheduler with warmup
    total_steps = args.steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-10,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )
    constant_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_steps - args.warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[args.warmup_steps]
    )

    # Training state
    global_step = 0
    best_score = -1
    accumulated_loss = 0.0
    accumulation_count = 0

    # Create eval directory
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # CSV for logging
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "lr", "accuracy"])

    # IMPORTANT: Keep model in eval mode to avoid NaN issue with GTR-T5/SentenceTransformers
    # Gradients still flow in eval mode, only dropout is disabled (which is fine for fine-tuning)
    model.eval()

    pbar = tqdm(total=args.steps, desc="Training")

    # Get tokenizer for manual encoding
    tokenizer = model.tokenizer

    # Setup autocast for bf16
    use_amp = args.use_bf16
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    def encode_with_grad(texts):
        """Encode texts with gradient tracking using SentenceTransformer internals."""
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward through all modules in the SentenceTransformer pipeline
        # The model is a Sequential: typically [Transformer, Pooling, (optional) Normalize]
        features = encoded
        for module in model:
            features = module(features)

        embeddings = features["sentence_embedding"]

        # Normalize embeddings with eps for numerical stability
        # Use larger eps (1e-4) for BF16 compatibility (1e-8 rounds to 0 in BF16)
        normalize_eps = 1e-4 if use_amp else 1e-8
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=normalize_eps)
        return embeddings

    while global_step < args.steps:
        for batch in train_dataloader:
            if global_step >= args.steps:
                break

            # Forward pass with autocast for bf16
            # batch is a list of InputExample, need to extract texts
            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]

            # Forward pass in bf16 for speed
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # Encode with gradient tracking
                anchor_emb = encode_with_grad(anchors)
                positive_emb = encode_with_grad(positives)
                negative_emb = encode_with_grad(negatives)

            # Compute loss based on loss_type
            if args.loss_type == "inbatch_softmax":
                # GTR paper: In-batch sampled softmax loss with hard negatives
                # Convert to fp32 for numerical stability with small temperature (0.01)
                anchor_emb_f = anchor_emb.float()
                positive_emb_f = positive_emb.float()
                negative_emb_f = negative_emb.float()

                batch_size = anchor_emb_f.size(0)

                # Concatenate all passage embeddings: [P0, P1, ..., N0, N1, ...]
                # Shape: (batch_size * 2, embedding_dim)
                all_passages = torch.cat([positive_emb_f, negative_emb_f], dim=0)

                # Compute similarity matrix: (batch_size, batch_size * 2)
                # Each row i: sim(Qi, P0), sim(Qi, P1), ..., sim(Qi, N0), sim(Qi, N1), ...
                similarity_matrix = torch.matmul(anchor_emb_f, all_passages.t()) / args.temperature

                # Labels: for query i, the positive is at index i (first batch_size columns are positives)
                labels = torch.arange(batch_size, device=device)

                # Cross-entropy loss (softmax over all candidates)
                loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)

            else:
                # TripletLoss: requires fp32 for numerical stability
                anchor_emb = anchor_emb.float()
                positive_emb = positive_emb.float()
                negative_emb = negative_emb.float()

                # TripletLoss: max(0, margin + d(anchor, positive) - d(anchor, negative))
                # Using cosine distance = 1 - cosine_similarity
                margin = 0.5

                pos_dist = 1 - torch.nn.functional.cosine_similarity(anchor_emb, positive_emb)
                neg_dist = 1 - torch.nn.functional.cosine_similarity(anchor_emb, negative_emb)

                loss = torch.nn.functional.relu(margin + pos_dist - neg_dist).mean()

            # Scale loss for gradient accumulation (outside autocast)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item() * args.gradient_accumulation_steps
            accumulation_count += 1

            # Gradient accumulation step
            if accumulation_count >= args.gradient_accumulation_steps:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accumulated_loss / accumulation_count
                accumulated_loss = 0.0
                accumulation_count = 0

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Evaluation
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Model is already in eval mode, no need to switch
                    eval_result = evaluator(model, output_path=eval_dir)
                    # Keep in eval mode to avoid NaN issue

                    # Extract accuracy from result (could be dict or float)
                    if isinstance(eval_result, dict):
                        # TripletEvaluator returns dict with 'accuracy' or similar key
                        score = eval_result.get('accuracy', eval_result.get('cosine_accuracy', list(eval_result.values())[0] if eval_result else 0.0))
                    else:
                        score = float(eval_result)

                    # Log to CSV
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, avg_loss, scheduler.get_last_lr()[0], score])

                    print(f"\n  Step {global_step}: loss={avg_loss:.4f}, accuracy={score:.4f}")

                    # Save best model
                    if score > best_score:
                        best_score = score
                        model.save(args.output_dir)
                        print(f"  New best model saved! (accuracy={score:.4f})")

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save(checkpoint_dir)
                    print(f"\n  Checkpoint saved to {checkpoint_dir}")

    pbar.close()

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    model.save(final_dir)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best accuracy: {best_score:.4f}")
    print(f"Best model saved to: {args.output_dir}")
    print(f"Final model saved to: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
