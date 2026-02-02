"""
KILT Dataset Loader
===================
Provides convenient functions to load KILT tasks and KILT Wikipedia datasets.

Usage:
    from kilt_loader import load_kilt_task, load_kilt_wikipedia, get_kilt_task_names
"""

from datasets import load_dataset
from typing import Optional, List, Dict, Any, Iterator
import os

# Default cache directory
DEFAULT_CACHE_DIR = "./kilt_data"

# All supported KILT task names (11 subsets)
KILT_TASK_NAMES = [
    "nq",                    # Natural Questions - Open-domain QA (91.7k rows)
    "aidayago2",             # AIDA CoNLL-YAGO - Entity linking (27.6k rows)
    "cweb",                  # CWEB - Entity linking (11.1k rows)
    "eli5",                  # Explain Like I'm Five - Long-form generation (275k rows)
    "fever",                 # FEVER - Fact verification (126k rows)
    "hotpotqa",              # HotpotQA - Multi-hop reasoning QA (100k rows)
    "structured_zeroshot",   # Zero Shot RE - Zero-shot relation extraction (157k rows)
    "trex",                  # T-REx - Slot filling (2.29M rows)
    "triviaqa_support_only", # TriviaQA - Knowledge QA (73.8k rows) - Note: ID only
    "wned",                  # WNED - Entity linking (6.77k rows)
    "wow",                   # Wizard of Wikipedia - Dialogue generation (69.7k rows)
]


def get_kilt_task_names() -> List[str]:
    """Return all available KILT task names."""
    return KILT_TASK_NAMES.copy()


def load_kilt_task(
    task_name: str,
    split: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    streaming: bool = False
) -> Any:
    """
    Load specified KILT task dataset.

    Args:
        task_name: Task name, e.g., "nq", "triviaqa", "hotpotqa"
        split: Data split, e.g., "train", "validation", "test". None returns all splits
        cache_dir: Data cache directory
        streaming: Whether to use streaming loading

    Returns:
        DatasetDict or Dataset

    Example:
        >>> nq_data = load_kilt_task("nq")
        >>> train = nq_data["train"]
        >>> for example in train:
        ...     question = example["input"]
        ...     answers = example["output"]
    """
    if task_name not in KILT_TASK_NAMES:
        raise ValueError(
            f"Unknown task name: {task_name}. "
            f"Available tasks: {KILT_TASK_NAMES}"
        )

    dataset = load_dataset(
        "facebook/kilt_tasks",
        name=task_name,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=True
    )

    if split is not None:
        return dataset[split]
    return dataset


def load_kilt_wikipedia(
    cache_dir: str = DEFAULT_CACHE_DIR,
    streaming: bool = True
) -> Any:
    """
    Load KILT Wikipedia knowledge base (5,903,530 articles).

    Since HuggingFace no longer supports loading script, load from local JSON file.

    Args:
        cache_dir: Data cache directory, should contain kilt_knowledgesource.json
        streaming: Whether to use generator mode (recommended, saves memory)

    Returns:
        If streaming=True, returns generator
        If streaming=False, returns list of articles (Warning: requires large memory)

    Example:
        >>> for article in load_kilt_wikipedia(streaming=True):
        ...     title = article["wikipedia_title"]
        ...     paragraphs = article["text"]  # List of paragraphs
        ...     wiki_id = article["wikipedia_id"]
    """
    import json

    json_path = os.path.join(cache_dir, "kilt_knowledgesource.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"KILT Wikipedia file does not exist: {json_path}\n"
            f"Please run first: python download_kilt_data.py --wikipedia-only"
        )

    def article_generator():
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.strip())

    if streaming:
        return article_generator()
    else:
        print("Warning: Loading complete dataset requires large memory...")
        articles = []
        for article in article_generator():
            articles.append(article)
        return articles


def parse_kilt_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a sample from KILT task.

    Args:
        example: A sample from KILT dataset

    Returns:
        Dictionary containing parsed information:
        - question: Input question/query
        - answers: List of answers
        - provenance: Wikipedia ID and position information of supporting passages
    """
    parsed = {
        "id": example.get("id", ""),
        "question": example.get("input", ""),
        "answers": [],
        "provenance": []
    }

    # Parse outputs (answers and sources)
    outputs = example.get("output", [])
    for output in outputs:
        # Get answer
        answer = output.get("answer", "")
        if answer and answer not in parsed["answers"]:
            parsed["answers"].append(answer)

        # Get supporting passage information
        provenances = output.get("provenance", [])
        for prov in provenances:
            parsed["provenance"].append({
                "wikipedia_id": prov.get("wikipedia_id", ""),
                "title": prov.get("title", ""),
                "start_paragraph_id": prov.get("start_paragraph_id", -1),
                "end_paragraph_id": prov.get("end_paragraph_id", -1),
                "start_character": prov.get("start_character", -1),
                "end_character": prov.get("end_character", -1),
            })

    return parsed


def build_passage_index(
    wiki_dataset,
    max_passages: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Build passage index from KILT Wikipedia (indexed by wikipedia_id).

    Warning: Complete dataset has 5,903,530 articles; recommend setting max_passages limit.

    Args:
        wiki_dataset: Generator or list returned by load_kilt_wikipedia()
        max_passages: Maximum passages to index, None means all
        show_progress: Whether to show progress

    Returns:
        Dict[wikipedia_id, passage_info]
    """
    index = {}

    for i, article in enumerate(wiki_dataset):
        if max_passages is not None and i >= max_passages:
            break

        if show_progress and i % 100000 == 0:
            print(f"Indexed {i} articles...")

        wiki_id = article.get("wikipedia_id", "")
        if wiki_id:
            index[wiki_id] = {
                "title": article.get("wikipedia_title", ""),
                "text": article.get("text", []),  # text is list of paragraphs in original JSON
                "anchors": article.get("anchors", []),
                "categories": article.get("categories", ""),
                "wikidata_info": article.get("wikidata_info", {}),
            }

    print(f"Index building complete, total {len(index)} articles")
    return index


def get_sample_data(
    task_name: str = "nq",
    n_samples: int = 5,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> List[Dict[str, Any]]:
    """
    Get sample data from specified task (for testing and demonstration).

    Args:
        task_name: Task name
        n_samples: Number of samples
        cache_dir: Cache directory

    Returns:
        List of parsed samples
    """
    dataset = load_kilt_task(task_name, split="validation", cache_dir=cache_dir)
    samples = []

    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append(parse_kilt_example(example))

    return samples


class KILTWikipediaDB:
    """
    KILT Wikipedia SQLite database query class.
    For fast article retrieval by wikipedia_id after retrieval.

    Usage:
        db = KILTWikipediaDB()
        article = db.get_by_id("45267196")
        articles = db.get_by_ids(["45267196", "21247"])
    """

    def __init__(self, db_path: str = None, cache_dir: str = DEFAULT_CACHE_DIR):
        import sqlite3

        if db_path is None:
            db_path = os.path.join(cache_dir, "kilt_wikipedia.db")

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite database does not exist: {db_path}\n"
                f"Please run first: python build_wiki_index.py --format sqlite"
            )

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_by_id(self, wikipedia_id: str) -> Optional[Dict[str, Any]]:
        """Get single article by wikipedia_id."""
        import json

        self.cursor.execute(
            'SELECT wikipedia_id, wikipedia_title, text, categories FROM articles WHERE wikipedia_id = ?',
            (wikipedia_id,)
        )
        result = self.cursor.fetchone()

        if result:
            return {
                'wikipedia_id': result[0],
                'wikipedia_title': result[1],
                'text': json.loads(result[2]) if result[2] else [],
                'categories': result[3]
            }
        return None

    def get_by_ids(self, wikipedia_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch get multiple articles."""
        results = []
        for wiki_id in wikipedia_ids:
            article = self.get_by_id(wiki_id)
            if article:
                results.append(article)
        return results

    def get_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article by title."""
        import json

        self.cursor.execute(
            'SELECT wikipedia_id, wikipedia_title, text, categories FROM articles WHERE wikipedia_title = ?',
            (title,)
        )
        result = self.cursor.fetchone()

        if result:
            return {
                'wikipedia_id': result[0],
                'wikipedia_title': result[1],
                'text': json.loads(result[2]) if result[2] else [],
                'categories': result[3]
            }
        return None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class KILTWikipediaArrow:
    """
    KILT Wikipedia Arrow format query class.
    Uses HuggingFace datasets Arrow format, supports fast lookup by ID.

    Usage:
        db = KILTWikipediaArrow()
        article = db.get_by_id("45267196")
        articles = db.get_by_ids(["45267196", "21247"])
    """

    def __init__(self, arrow_path: str = None, cache_dir: str = DEFAULT_CACHE_DIR):
        from datasets import load_from_disk
        import pickle

        if arrow_path is None:
            arrow_path = os.path.join(cache_dir, "kilt_wikipedia_arrow")

        if not os.path.exists(arrow_path):
            raise FileNotFoundError(
                f"Arrow data does not exist: {arrow_path}\n"
                f"Please run first: python build_wiki_index.py --format arrow"
            )

        # Load Arrow dataset
        print(f"Loading Arrow dataset: {arrow_path}")
        self.dataset = load_from_disk(arrow_path)

        # Load ID mapping
        mapping_path = os.path.join(arrow_path, 'id_to_idx.pkl')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"ID mapping file does not exist: {mapping_path}\n"
                f"Please re-run: python build_wiki_index.py --format arrow"
            )

        with open(mapping_path, 'rb') as f:
            self.id_to_idx = pickle.load(f)

        print(f"Loaded {len(self.dataset)} articles, {len(self.id_to_idx)} ID mappings")

    def get_by_id(self, wikipedia_id: str) -> Optional[Dict[str, Any]]:
        """Get single article by wikipedia_id."""
        idx = self.id_to_idx.get(wikipedia_id)
        if idx is not None:
            return self.dataset[idx]
        return None

    def get_by_ids(self, wikipedia_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch get multiple articles."""
        indices = [self.id_to_idx[wid] for wid in wikipedia_ids if wid in self.id_to_idx]
        if indices:
            # Use select for efficient batch query
            return self.dataset.select(indices)
        return []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preload_to_memory(self):
        """
        Preload all articles to memory, eliminating IO bottleneck.
        Requires ~10-15GB memory, but significantly improves batch query speed.
        """
        import time
        from tqdm import tqdm
        print(f"Preloading {len(self.dataset)} articles to memory (Pandas acceleration)...")

        # Step 1: Arrow -> Pandas (main time consumption)
        print("  [1/3] Arrow -> Pandas DataFrame...", end=" ", flush=True)
        t0 = time.time()
        df = self.dataset.to_pandas()
        print(f"Done ({time.time()-t0:.1f}s)")

        # Step 2: DataFrame -> list of dicts
        print("  [2/3] DataFrame -> Records...", end=" ", flush=True)
        t0 = time.time()
        records = df.to_dict('records')
        del df
        print(f"Done ({time.time()-t0:.1f}s)")

        # Step 3: Build index (can add progress bar)
        print("  [3/3] Building index...")
        self.memory_cache = {}
        for r in tqdm(records, desc="  Building index"):
            wiki_id = r.get('wikipedia_id', '')
            if wiki_id:
                self.memory_cache[wiki_id] = r
        del records

        print(f"Preload complete! Memory cache {len(self.memory_cache)} articles")
        self._use_memory_cache = True

    def is_preloaded(self) -> bool:
        """Check if already preloaded to memory."""
        return hasattr(self, '_use_memory_cache') and self._use_memory_cache

    def get_from_cache(self, wikipedia_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch get articles from memory cache, returns {wiki_id: article} dict."""
        if self.is_preloaded():
            return {wid: self.memory_cache[wid] for wid in wikipedia_ids if wid in self.memory_cache}
        # Fallback: Read from disk
        indices = [self.id_to_idx[wid] for wid in wikipedia_ids if wid in self.id_to_idx]
        if indices:
            articles = self.dataset.select(indices)
            return {a.get("wikipedia_id", ""): dict(a) for a in articles}
        return {}


def load_filtered_kilt_task(
    task_name: str,
    split: str = "train",
    filtered_dir: str = None,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> List[Dict[str, Any]]:
    """
    Load filtered KILT task data (only samples with provenance).

    Args:
        task_name: Task name, e.g., "nq", "triviaqa_support_only", "hotpotqa", "fever"
        split: Data split, "train" or "validation"
        filtered_dir: Filtered data directory, default {cache_dir}/filtered
        cache_dir: Cache directory

    Returns:
        List of samples, each containing id, input, output, meta

    Example:
        >>> nq_train = load_filtered_kilt_task("nq", "train")
        >>> for example in nq_train:
        ...     question = example["input"]
        ...     provenance = example["output"][0]["provenance"]
        ...     wiki_id = provenance[0]["wikipedia_id"]
    """
    import json

    if filtered_dir is None:
        filtered_dir = os.path.join(cache_dir, "filtered")

    file_path = os.path.join(filtered_dir, f"{task_name}_{split}.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Filtered data file does not exist: {file_path}\n"
            f"Please run first: python filter_kilt_data.py --tasks {task_name}"
        )

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return data


def get_filtered_task_stats(
    cache_dir: str = DEFAULT_CACHE_DIR
) -> Dict[str, Dict[str, int]]:
    """
    Get statistics for all filtered data.

    Returns:
        {task_name: {split: count, ...}, ...}
    """
    import json

    filtered_dir = os.path.join(cache_dir, "filtered")
    stats = {}

    if not os.path.exists(filtered_dir):
        return stats

    for filename in os.listdir(filtered_dir):
        if filename.endswith('.jsonl'):
            # Parse filename: task_split.jsonl
            parts = filename[:-6].rsplit('_', 1)  # Remove .jsonl and split by last _
            if len(parts) == 2:
                task_name, split = parts

                # Count lines
                file_path = os.path.join(filtered_dir, filename)
                with open(file_path, 'r') as f:
                    count = sum(1 for _ in f)

                if task_name not in stats:
                    stats[task_name] = {}
                stats[task_name][split] = count

    return stats


if __name__ == "__main__":
    # Test loading functionality
    print("Testing KILT data loader...")
    print(f"Available tasks: {get_kilt_task_names()}")

    # Test loading NQ task
    print("\nLoading Natural Questions validation samples...")
    try:
        samples = get_sample_data("nq", n_samples=3)
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"Question: {sample['question']}")
            print(f"Answers: {sample['answers']}")
            print(f"Provenance count: {len(sample['provenance'])}")
    except Exception as e:
        print(f"Loading failed: {e}")
        print("Please run python download_kilt_data.py to download data first")
