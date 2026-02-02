"""
GTR-T5-Base Retriever for FiD-Light
====================================

Dense retriever using GTR-T5-Base embeddings and Faiss index over KILT Wikipedia.

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023)

Usage:
    from gtr_retriever import GTRRetriever

    retriever = GTRRetriever()
    results = retriever.retrieve("Who is the president?", top_k=40)
    # results: [{"wikipedia_id": "...", "title": "...", "text": "...", "score": 0.85}, ...]

    # Batch retrieval for training
    batch_results = retriever.batch_retrieve(["query1", "query2"], top_k=40)
"""

import os
import pickle
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import torch

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed. Install with: pip install faiss-cpu (or faiss-gpu)")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

from kilt_loader import KILTWikipediaArrow


class GTRRetriever:
    """
    GTR-T5-Base retriever for FiD-Light paper reproduction.

    Uses pre-built Faiss index over 5.9M KILT Wikipedia articles.

    Attributes:
        model: SentenceTransformer GTR-T5-Base model
        index: Faiss IndexFlatIP for similarity search
        idx_to_wiki_id: Mapping from Faiss index to Wikipedia ID
        wiki: KILTWikipediaArrow for article lookup
    """

    DEFAULT_INDEX_PATH = "kilt_data/gtr_faiss_index"

    def __init__(
        self,
        index_path: str = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_wiki: bool = True,
        use_multi_gpu: bool = False,
        preload_wiki: bool = False,
        use_mmap: bool = None,
        wiki_arrow_path: Optional[str] = None,
    ):
        """
        Initialize the GTR retriever.

        Args:
            index_path: Path to the Faiss index directory
            model_path: Path to model (default: GTR-T5-Base, or path to fine-tuned model)
            device: 'cuda' or 'cpu' (auto-detected if None)
            load_wiki: Whether to load Wikipedia for text lookup (set False for ID-only retrieval)
            use_multi_gpu: Use all available GPUs for encoding (recommended for large batch retrieval)
            preload_wiki: Preload all Wikipedia articles to memory (~10-15GB) to eliminate IO bottleneck
            use_mmap: Use memory-mapped loading for Faiss index (faster startup, but cannot use GPU index).
                      Default: None = auto (True for index > 1GB, False otherwise).
                      Set False explicitly to enable GPU index for batch precompute.
            wiki_arrow_path: Path to Wikipedia Arrow dataset (default: kilt_data/kilt_wikipedia_arrow)
        """
        self.wiki_arrow_path = wiki_arrow_path
        if index_path is None:
            index_path = self.DEFAULT_INDEX_PATH

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate index exists
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            raise FileNotFoundError(
                f"GTR Faiss index not found at {index_path}.\n"
                f"Please run: python build_gtr_index.py"
            )

        print(f"Loading GTR retriever from {index_path}...")

        # Load model (default or fine-tuned)
        if model_path is None:
            model_path = "sentence-transformers/gtr-t5-base"
        print(f"  Loading model from {model_path}...")
        self.model = SentenceTransformer(model_path, device=device)
        self.model_path = model_path
        self.device = device

        # Multi-GPU encoding pool
        self.pool = None
        self.use_multi_gpu = use_multi_gpu
        n_gpus = torch.cuda.device_count()

        if use_multi_gpu and n_gpus > 1:
            print(f"  Starting multi-GPU pool with {n_gpus} GPUs...")
            target_devices = [f"cuda:{i}" for i in range(n_gpus)]
            self.pool = self.model.start_multi_process_pool(target_devices=target_devices)
            print(f"  Multi-GPU pool ready: {target_devices}")

        # Load Faiss index
        index_file = os.path.join(index_path, "index.faiss")
        index_size_gb = os.path.getsize(index_file) / (1024**3)
        print(f"  Loading Faiss index ({index_size_gb:.1f} GB)...")

        # Determine whether to use mmap
        # - use_mmap=None (default): auto-detect based on size (>1GB uses mmap)
        # - use_mmap=True: force mmap (fast startup, no GPU)
        # - use_mmap=False: force full load (slow startup, can use GPU for batch precompute)
        if use_mmap is None:
            use_mmap = index_size_gb > 1.0
            if use_mmap:
                print("  Auto-selecting mmap for large index (use use_mmap=False to disable)")

        if use_mmap:
            print("  Using memory-mapped loading (fast startup, CPU search)...")
            cpu_index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        else:
            print("  Loading full index to memory (slow startup, can use GPU)...")
            cpu_index = faiss.read_index(index_file)

        # Try to move index to GPU for faster search (only for non-mmap indexes)
        self.use_gpu_index = False
        if not use_mmap and device == "cuda" and faiss.get_num_gpus() > 0:
            try:
                print("  Moving Faiss index to GPU...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.use_gpu_index = True
                print("  Faiss index on GPU!")
            except Exception as e:
                print(f"  Could not move index to GPU: {e}")
                print(f"  (This is normal if index is too large for GPU memory)")
                self.index = cpu_index
        else:
            self.index = cpu_index

        # Load ID mapping
        print("  Loading ID mapping...")
        with open(os.path.join(index_path, "idx_to_wiki_id.pkl"), 'rb') as f:
            self.idx_to_wiki_id = pickle.load(f)
        # Pre-create numpy array for fast vectorized lookup
        self.idx_to_wiki_id_array = np.array(self.idx_to_wiki_id)

        # Load metadata if available
        meta_path = os.path.join(index_path, "metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

        # Load Wikipedia for text lookup
        self.wiki = None
        if load_wiki:
            print("  Loading Wikipedia Arrow dataset...")
            try:
                self.wiki = KILTWikipediaArrow(arrow_path=self.wiki_arrow_path)
            except Exception as e:
                print(f"  Warning: Could not load Wikipedia: {e}")
                print("  Text lookup will be unavailable")

        # Preload Wikipedia to memory if requested
        if preload_wiki and self.wiki is not None:
            self.wiki.preload_to_memory()

        print(f"GTR retriever ready. Index size: {self.index.ntotal:,} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = 40,
        return_text: bool = True,
        max_paragraphs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k passages for a query.

        Args:
            query: Query string
            top_k: Number of passages to retrieve
            return_text: Whether to include full article text
            max_paragraphs: Max paragraphs to include per article

        Returns:
            List of passage dicts:
            [
                {
                    "wikipedia_id": str,
                    "title": str,
                    "text": str,        # Combined paragraphs
                    "score": float,     # Similarity score
                    "rank": int         # 1-based rank
                },
                ...
            ]
        """
        # Encode query
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Search Faiss index
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)

        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.idx_to_wiki_id):
                continue

            wiki_id = self.idx_to_wiki_id[idx]

            result = {
                "wikipedia_id": wiki_id,
                "title": "",
                "text": "",
                "score": float(score),
                "rank": rank + 1  # 1-based
            }

            # Get article details from Wikipedia
            if return_text and self.wiki is not None:
                article = self.wiki.get_by_id(wiki_id)
                if article:
                    result["title"] = article.get("wikipedia_title", "")
                    paragraphs = article.get("text", [])
                    if isinstance(paragraphs, list):
                        result["text"] = " ".join(paragraphs[:max_paragraphs])
                    else:
                        result["text"] = str(paragraphs)

            results.append(result)

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 40,
        return_text: bool = True,
        max_paragraphs: int = 5,
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve passages for multiple queries.

        More efficient than calling retrieve() multiple times.
        Uses batch Wikipedia lookups for better I/O performance.

        Args:
            queries: List of query strings
            top_k: Number of passages per query
            return_text: Whether to include full article text
            max_paragraphs: Max paragraphs per article
            batch_size: Encoding batch size

        Returns:
            List of results per query, each containing top_k passages
        """
        if not queries:
            return []

        # Encode all queries (multi-GPU if available)
        if self.pool is not None:
            # Multi-GPU encoding
            q_embs = self.model.encode_multi_process(
                queries,
                self.pool,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        else:
            # Single GPU encoding
            q_embs = self.model.encode(
                queries,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(queries) > 100
            )

        # Search Faiss index
        scores, indices = self.index.search(q_embs.astype(np.float32), top_k)

        # Collect all unique wikipedia_ids for batch lookup - VECTORIZED
        if return_text and self.wiki is not None:
            # Flatten indices and filter valid ones using numpy (much faster than Python loops)
            flat_indices = indices.flatten()
            valid_mask = (flat_indices >= 0) & (flat_indices < len(self.idx_to_wiki_id))
            valid_flat_indices = flat_indices[valid_mask]

            # Get unique indices for batch lookup
            unique_indices = np.unique(valid_flat_indices)

            # Map indices to wiki_ids using numpy advanced indexing (use pre-created array)
            unique_wiki_ids = self.idx_to_wiki_id_array[unique_indices].tolist()

            # Batch fetch all articles - use memory cache if available
            if unique_wiki_ids:
                if self.wiki.is_preloaded():
                    # Fast path: direct memory lookup
                    wiki_id_to_article = self.wiki.get_from_cache(unique_wiki_ids)
                else:
                    # Slow path: disk IO via Arrow
                    valid_arrow_indices = []
                    valid_wiki_ids = []
                    for wid in unique_wiki_ids:
                        arrow_idx = self.wiki.id_to_idx.get(wid)
                        if arrow_idx is not None:
                            valid_arrow_indices.append(arrow_idx)
                            valid_wiki_ids.append(wid)

                    if valid_arrow_indices:
                        # Single batch read from disk
                        batch_articles = self.wiki.dataset.select(valid_arrow_indices)
                        wiki_id_to_article = dict(zip(valid_wiki_ids, batch_articles))
                    else:
                        wiki_id_to_article = {}
            else:
                wiki_id_to_article = {}
        else:
            wiki_id_to_article = {}

        # Build results for each query - PARALLELIZED
        def process_single_query(q_idx):
            """Process results for a single query."""
            results = []
            for rank, (score, idx) in enumerate(zip(scores[q_idx], indices[q_idx])):
                if idx < 0 or idx >= len(self.idx_to_wiki_id):
                    continue

                wiki_id = self.idx_to_wiki_id[idx]

                result = {
                    "wikipedia_id": wiki_id,
                    "title": "",
                    "text": "",
                    "score": float(score),
                    "rank": rank + 1
                }

                if return_text and wiki_id in wiki_id_to_article:
                    article = wiki_id_to_article[wiki_id]
                    result["title"] = article.get("wikipedia_title", "")
                    paragraphs = article.get("text", [])
                    if isinstance(paragraphs, list):
                        result["text"] = " ".join(paragraphs[:max_paragraphs])
                    else:
                        result["text"] = str(paragraphs)

                results.append(result)
            return q_idx, results

        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(32, multiprocessing.cpu_count())
        all_results = [None] * len(queries)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_query, q_idx) for q_idx in range(len(queries))]
            for future in futures:
                q_idx, results = future.result()
                all_results[q_idx] = results

        return all_results

    def retrieve_ids_only(
        self,
        query: str,
        top_k: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Fast retrieval returning only IDs and scores (no text lookup).

        Useful for provenance matching where you only need wikipedia_ids.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of {"wikipedia_id": str, "score": float, "rank": int}
        """
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.idx_to_wiki_id):
                results.append({
                    "wikipedia_id": self.idx_to_wiki_id[idx],
                    "score": float(score),
                    "rank": rank + 1
                })

        return results

    def get_article(self, wikipedia_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a Wikipedia article by ID.

        Args:
            wikipedia_id: Wikipedia article ID

        Returns:
            Article dict or None if not found
        """
        if self.wiki is None:
            return None
        return self.wiki.get_by_id(wikipedia_id)

    def format_passage_for_fid(
        self,
        passage: Dict[str, Any],
        query: str,
        index: int
    ) -> str:
        """
        Format a passage for FiD-Light input (with source pointer).

        Paper format: "query: {Q} index: {i} context: {title} {text}"

        Args:
            passage: Passage dict from retrieve()
            query: Original query
            index: 1-based passage index for source pointer

        Returns:
            Formatted input string
        """
        title = passage.get("title", "")
        text = passage.get("text", "")
        return f"query: {query} index: {index} context: {title} {text}"

    def format_passages_for_fid(
        self,
        passages: List[Dict[str, Any]],
        query: str
    ) -> List[str]:
        """
        Format all passages for FiD-Light input.

        Args:
            passages: List of passages from retrieve()
            query: Original query

        Returns:
            List of formatted input strings (one per passage)
        """
        return [
            self.format_passage_for_fid(p, query, i + 1)
            for i, p in enumerate(passages)
        ]

    def format_passage_for_fid_pure(
        self,
        passage: Dict[str, Any],
        query: str
    ) -> str:
        """
        Format a single passage for FiD Pure (original FiD) input.

        Args:
            passage: Passage dict with 'title' and 'text'
            query: Original query

        Returns:
            Formatted input string in FiD Pure format: "question: {Q} title: {T} context: {P}"
        """
        title = passage.get("title", "")
        text = passage.get("text", "")
        return f"question: {query} title: {title} context: {text}"

    def format_passages_for_fid_pure(
        self,
        passages: List[Dict[str, Any]],
        query: str
    ) -> List[str]:
        """
        Format all passages for FiD Pure (original FiD) input.

        Args:
            passages: List of passages from retrieve()
            query: Original query

        Returns:
            List of formatted input strings (one per passage)
        """
        return [
            self.format_passage_for_fid_pure(p, query)
            for p in passages
        ]

    @property
    def num_passages(self) -> int:
        """Number of passages in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"GTRRetriever(num_passages={self.num_passages:,}, device={self.device}, multi_gpu={self.pool is not None})"

    def close(self):
        """Stop multi-GPU pool if running."""
        if self.pool is not None:
            print("Stopping multi-GPU pool...")
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None


def demo():
    """Demo the retriever with sample queries."""
    print("=" * 60)
    print("GTR Retriever Demo")
    print("=" * 60)

    # Initialize retriever
    retriever = GTRRetriever()
    print(f"\n{retriever}\n")

    # Test queries
    test_queries = [
        "Who is the president of the United States?",
        "What is the capital of France?",
        "When was the Eiffel Tower built?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        results = retriever.retrieve(query, top_k=5)

        for r in results:
            print(f"  [{r['rank']}] {r['title']} (score={r['score']:.4f})")
            if r['text']:
                text_preview = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
                print(f"      {text_preview}")

    # Demo FiD formatting
    print("\n" + "=" * 60)
    print("FiD-Light Input Format Demo")
    print("=" * 60)

    query = "Who discovered penicillin?"
    results = retriever.retrieve(query, top_k=3)
    formatted = retriever.format_passages_for_fid(results, query)

    for i, text in enumerate(formatted):
        print(f"\n[Passage {i+1}]")
        print(text[:200] + "..." if len(text) > 200 else text)


if __name__ == "__main__":
    demo()
