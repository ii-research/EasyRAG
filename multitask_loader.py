"""
Multi-Task KILT Data Loader with Temperature Sampling
======================================================

Implements the multi-task training data pipeline for FiD-Light paper reproduction.

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023)

Key features:
1. Temperature sampling: P_task ∝ (N_task)^(1/T) where T=2
2. Provenance verification: Skip samples where retrieval misses provenance
3. Source pointer target generation: "index: {i1,i2} text: {answer}"

Usage:
    from multitask_loader import MultiTaskKILTLoader, prepare_training_sample
    from gtr_retriever import GTRRetriever

    loader = MultiTaskKILTLoader(temperature=2.0)
    retriever = GTRRetriever()

    # Sample a batch
    batch = loader.sample_batch(batch_size=1)

    # Prepare for training (with provenance verification)
    for sample in batch:
        prepared = prepare_training_sample(sample, retriever)
        if prepared is None:
            continue  # Skip - no provenance match
        # Use prepared["input_texts"], prepared["target_text"] for training
"""

import os
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

import numpy as np

from kilt_loader import load_filtered_kilt_task


class MultiTaskKILTLoader:
    """
    Multi-task data loader with temperature-based sampling.

    Implements the sampling strategy from FiD-Light paper:
    - Sample rate: P_task ∝ (N_task)^(1/T)
    - Temperature T=2 balances between uniform and size-proportional sampling

    Supported KILT tasks (from kilt_data/filtered/):
    - nq: Natural Questions (76,945 samples)
    - fever: Fact verification (71,257 samples)
    - hotpotqa: Multi-hop QA (68,659 samples)
    - structured_zeroshot: Zero-shot RE (132,063 samples)
    - trex: T-REx slot filling (2,284,168 samples)
    - triviaqa_support_only: TriviaQA (52,886 samples)
    - aidayago2: Entity linking (18,395 samples)
    - wow: Wizard of Wikipedia dialogue (54,330 samples)
    """

    # KILT tasks for FiD-Light training (3 QA tasks only)
    TASKS = [
        "nq",
        "hotpotqa",
        "triviaqa_support_only",
    ]

    # Task-specific prefixes for multi-task learning
    TASK_PREFIXES = {
        "nq": "question:",
        "fever": "claim:",
        "hotpotqa": "question:",
        "structured_zeroshot": "relation:",
        "trex": "fact:",
        "triviaqa_support_only": "question:",
        "aidayago2": "entity:",
        "wow": "dialogue:",
    }

    # Task types for evaluation grouping
    TASK_TYPES = {
        "nq": "qa",
        "fever": "fact_verification",
        "hotpotqa": "qa",
        "structured_zeroshot": "slot_filling",
        "trex": "slot_filling",
        "triviaqa_support_only": "qa",
        "aidayago2": "entity_linking",
        "wow": "dialogue",
    }

    def __init__(
        self,
        temperature: float = 2.0,
        cache_dir: str = "kilt_data",
        tasks: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize the multi-task loader.

        Args:
            temperature: Sampling temperature T (default 2.0 per paper)
                - T=1: Sample proportional to dataset size
                - T→∞: Uniform sampling across tasks
                - T=2: Balanced (paper default)
            cache_dir: Directory containing filtered KILT data
            tasks: List of task names to load (default: all 8 tasks)
            seed: Random seed for reproducibility
        """
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.tasks_to_load = tasks if tasks else self.TASKS
        self.rng = np.random.default_rng(seed)

        # Task data storage
        self.task_data: Dict[str, List[Dict]] = {}
        self.task_sizes: Dict[str, int] = {}
        self.task_probs: Dict[str, float] = {}

        # Load all tasks
        self._load_tasks()

        # Compute sampling probabilities
        self._compute_sampling_probs()

    def _load_tasks(self) -> None:
        """Load all filtered KILT task datasets."""
        print(f"Loading {len(self.tasks_to_load)} KILT tasks...")

        for task in self.tasks_to_load:
            try:
                data = load_filtered_kilt_task(
                    task,
                    split="train",
                    cache_dir=self.cache_dir
                )
                self.task_data[task] = data
                self.task_sizes[task] = len(data)
                print(f"  {task}: {len(data):,} samples")
            except FileNotFoundError as e:
                print(f"  {task}: SKIPPED - {e}")

        total = sum(self.task_sizes.values())
        print(f"Total: {total:,} samples across {len(self.task_data)} tasks")

    def _compute_sampling_probs(self) -> None:
        """
        Compute temperature-adjusted sampling probabilities.

        Formula: P_task ∝ (N_task)^(1/T)

        With T=2:
        - Large datasets (e.g., trex 2.3M) get downweighted
        - Small datasets (e.g., aidayago2 18K) get upweighted
        """
        if not self.task_sizes:
            return

        # Apply temperature: N^(1/T)
        adjusted = {
            task: size ** (1 / self.temperature)
            for task, size in self.task_sizes.items()
        }

        # Normalize to probabilities
        total = sum(adjusted.values())
        self.task_probs = {
            task: adj / total
            for task, adj in adjusted.items()
        }

        print(f"\nTemperature sampling (T={self.temperature}):")
        for task, prob in sorted(self.task_probs.items(), key=lambda x: -x[1]):
            orig_prob = self.task_sizes[task] / sum(self.task_sizes.values())
            print(f"  {task}: {prob:.4f} (original: {orig_prob:.4f})")

    def sample_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Sample a batch using temperature-weighted task sampling.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of samples, each with added "_task" field
        """
        if not self.task_probs:
            return []

        tasks = list(self.task_probs.keys())
        probs = [self.task_probs[t] for t in tasks]

        # Sample tasks according to temperature-adjusted probabilities
        sampled_tasks = self.rng.choice(tasks, size=batch_size, p=probs)

        batch = []
        for task in sampled_tasks:
            # Random sample from the task
            idx = self.rng.integers(len(self.task_data[task]))
            sample = self.task_data[task][idx].copy()
            sample["_task"] = task
            batch.append(sample)

        return batch

    def get_task_prefix(self, task: str) -> str:
        """Get the task-specific input prefix."""
        return self.TASK_PREFIXES.get(task, "query:")

    def get_validation_samples(
        self,
        task: str,
        n_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get validation samples for a specific task.

        Args:
            task: Task name
            n_samples: Number of samples (None = all)

        Returns:
            List of validation samples
        """
        try:
            data = load_filtered_kilt_task(
                task,
                split="validation",
                cache_dir=self.cache_dir
            )
            if n_samples:
                data = data[:n_samples]
            return data
        except FileNotFoundError:
            print(f"Warning: Validation data not found for {task}")
            return []

    @property
    def total_samples(self) -> int:
        """Total number of training samples across all tasks."""
        return sum(self.task_sizes.values())

    @property
    def num_tasks(self) -> int:
        """Number of loaded tasks."""
        return len(self.task_data)

    def __repr__(self) -> str:
        return (
            f"MultiTaskKILTLoader(tasks={self.num_tasks}, "
            f"samples={self.total_samples:,}, T={self.temperature})"
        )


def extract_provenance_ids(sample: Dict[str, Any]) -> Set[str]:
    """
    Extract all Wikipedia IDs from a sample's provenance.

    Args:
        sample: KILT sample with "output" field containing provenance

    Returns:
        Set of wikipedia_id strings
    """
    wiki_ids = set()

    outputs = sample.get("output", [])
    for output in outputs:
        provenances = output.get("provenance", [])
        for prov in provenances:
            wiki_id = prov.get("wikipedia_id")
            if wiki_id:
                wiki_ids.add(str(wiki_id))

    return wiki_ids


def extract_answer(sample: Dict[str, Any]) -> str:
    """
    Extract the answer from a KILT sample.

    Args:
        sample: KILT sample

    Returns:
        Answer string (first non-empty answer found)
    """
    outputs = sample.get("output", [])
    for output in outputs:
        answer = output.get("answer", "")
        if answer:
            return answer
    return ""


def prepare_training_sample(
    sample: Dict[str, Any],
    retriever: Any,  # GTRRetriever
    num_passages: int = 40,
    max_input_tokens: int = 384,
    max_chars_per_passage: int = 1500
) -> Optional[Dict[str, Any]]:
    """
    Prepare a single training sample with provenance verification.

    This is the core function that implements the FiD-Light training data pipeline:
    1. Extract provenance Wikipedia IDs from the training sample
    2. Retrieve top-K passages using GTR-T5-Base
    3. Check if ANY retrieved passage matches provenance
    4. If no match: return None (skip this sample)
    5. If match: generate source pointer targets and formatted inputs

    Paper requirement: Only train on samples where retrieval finds the provenance.

    Args:
        sample: KILT sample with "input", "output", and optional "_task"
        retriever: GTRRetriever instance
        num_passages: Number of passages to retrieve (default 40 per paper)
        max_input_tokens: Max tokens per input (384 per paper)
        max_chars_per_passage: Approximate char limit per passage

    Returns:
        Prepared sample dict:
        {
            "input_texts": List[str],     # "query: Q index: i context: P" for each passage
            "target_text": str,           # "index: {i1,i2} text: {answer}"
            "matching_indices": List[int], # 1-based indices of matching passages
            "task": str,                  # Task name
            "query": str,                 # Original query
            "answer": str,                # Ground truth answer
            "retrieved_passages": List[Dict],  # Full passage info
        }
        or None if no provenance match
    """
    # Extract query and answer
    query = sample.get("input", "")
    answer = extract_answer(sample)
    task = sample.get("_task", "unknown")

    if not query or not answer:
        return None

    # Extract provenance Wikipedia IDs
    provenance_ids = extract_provenance_ids(sample)
    if not provenance_ids:
        return None

    # Retrieve passages
    retrieved = retriever.retrieve(query, top_k=num_passages)

    # Find matching indices (1-based for source pointer)
    matching_indices = []
    for i, passage in enumerate(retrieved):
        wiki_id = passage.get("wikipedia_id", "")
        if wiki_id in provenance_ids:
            matching_indices.append(i + 1)  # 1-based

    # Skip if no provenance match (paper requirement)
    if not matching_indices:
        return None

    # Format input texts: "query: Q index: i context: P"
    input_texts = []
    for i, passage in enumerate(retrieved):
        title = passage.get("title", "")
        text = passage.get("text", "")

        # Truncate passage text to fit within token budget
        context = f"{title} {text}"
        if len(context) > max_chars_per_passage:
            context = context[:max_chars_per_passage]

        formatted = f"query: {query} index: {i+1} context: {context}"
        input_texts.append(formatted)

    # Format target: "index: {i1,i2} text: {answer}"
    # Keep up to 3 matching indices (most samples have 1-2)
    indices_str = ",".join(str(idx) for idx in matching_indices[:3])
    target_text = f"index: {indices_str} text: {answer}"

    return {
        "input_texts": input_texts,
        "target_text": target_text,
        "matching_indices": matching_indices,
        "task": task,
        "query": query,
        "answer": answer,
        "retrieved_passages": retrieved,
    }


def compute_provenance_hit_rate(
    loader: MultiTaskKILTLoader,
    retriever: Any,
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute provenance hit rate for each task.

    This measures how often the retriever finds the gold provenance.

    Args:
        loader: MultiTaskKILTLoader instance
        retriever: GTRRetriever instance
        n_samples: Number of samples to test

    Returns:
        Dict mapping task -> hit rate (0.0-1.0)
    """
    hit_rates = {}

    for task in loader.task_data.keys():
        samples = loader.task_data[task][:n_samples]
        hits = 0

        for sample in samples:
            sample["_task"] = task
            prepared = prepare_training_sample(sample, retriever)
            if prepared is not None:
                hits += 1

        hit_rate = hits / len(samples) if samples else 0
        hit_rates[task] = hit_rate
        print(f"{task}: {hit_rate:.2%} ({hits}/{len(samples)})")

    return hit_rates


class TrainingDataIterator:
    """
    Iterator that yields prepared training samples.

    Handles provenance verification and sample skipping automatically.
    """

    def __init__(
        self,
        loader: MultiTaskKILTLoader,
        retriever: Any,
        num_passages: int = 40,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the training data iterator.

        Args:
            loader: MultiTaskKILTLoader
            retriever: GTRRetriever
            num_passages: Passages per query
            max_samples: Max samples to yield (None = infinite)
        """
        self.loader = loader
        self.retriever = retriever
        self.num_passages = num_passages
        self.max_samples = max_samples

        # Statistics
        self.total_sampled = 0
        self.total_skipped = 0
        self.task_stats = defaultdict(lambda: {"sampled": 0, "skipped": 0})

    def __iter__(self):
        """Iterate over training samples."""
        yielded = 0

        while True:
            if self.max_samples and yielded >= self.max_samples:
                break

            # Sample a batch of 1
            batch = self.loader.sample_batch(batch_size=1)
            if not batch:
                continue

            sample = batch[0]
            task = sample.get("_task", "unknown")
            self.total_sampled += 1
            self.task_stats[task]["sampled"] += 1

            # Prepare with provenance verification
            prepared = prepare_training_sample(
                sample,
                self.retriever,
                num_passages=self.num_passages
            )

            if prepared is None:
                self.total_skipped += 1
                self.task_stats[task]["skipped"] += 1
                continue

            yielded += 1
            yield prepared

    @property
    def skip_rate(self) -> float:
        """Overall skip rate due to provenance miss."""
        if self.total_sampled == 0:
            return 0.0
        return self.total_skipped / self.total_sampled

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            "total_sampled": self.total_sampled,
            "total_skipped": self.total_skipped,
            "skip_rate": self.skip_rate,
            "task_stats": dict(self.task_stats)
        }


def demo():
    """Demo the multi-task loader."""
    print("=" * 60)
    print("Multi-Task KILT Loader Demo")
    print("=" * 60)

    # Initialize loader
    loader = MultiTaskKILTLoader(temperature=2.0)
    print(f"\n{loader}")

    # Sample some batches
    print("\nSampling 10 samples:")
    batch = loader.sample_batch(batch_size=10)

    task_counts = defaultdict(int)
    for sample in batch:
        task = sample["_task"]
        task_counts[task] += 1
        print(f"  [{task}] {sample['input'][:50]}...")

    print(f"\nTask distribution: {dict(task_counts)}")

    # Show provenance extraction
    print("\n" + "=" * 60)
    print("Provenance Extraction Demo")
    print("=" * 60)

    for task in ["nq", "hotpotqa"]:
        if task in loader.task_data:
            sample = loader.task_data[task][0]
            prov_ids = extract_provenance_ids(sample)
            answer = extract_answer(sample)
            print(f"\n[{task}]")
            print(f"  Query: {sample['input'][:60]}...")
            print(f"  Answer: {answer[:60]}...")
            print(f"  Provenance IDs: {list(prov_ids)[:3]}")


if __name__ == "__main__":
    demo()
