"""
GTR-T5-Base Faiss Index Builder for FiD-Light
==============================================

This script builds a Faiss index over 5.9M KILT Wikipedia articles using
the GTR-T5-Base model (768-dimensional embeddings).

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023)

Usage:
    python build_gtr_index.py                       # Full index (5.9M articles)
    python build_gtr_index.py --max_articles 10000  # Quick test with 10K articles
    python build_gtr_index.py --batch_size 64       # Smaller batch for low memory

Output:
    kilt_data/gtr_faiss_index/
    ├── index.faiss          # Faiss IndexFlatIP (768-dim)
    └── idx_to_wiki_id.pkl   # Mapping: faiss_index -> wikipedia_id
"""

import argparse
import os
import pickle
import time
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("Error: faiss not installed. Install with: pip install faiss-cpu (or faiss-gpu)")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Install with: pip install sentence-transformers")
    exit(1)

from kilt_loader import KILTWikipediaArrow


def format_article_text(article: dict, max_paragraphs: int = 5, max_chars: int = 2000) -> str:
    """
    Format a Wikipedia article for encoding.

    Following KILT official format:
    - Combine title with article text directly (no labels)
    - Format: "{title} {text}"

    Args:
        article: Wikipedia article dict with 'wikipedia_title' and 'text' (list of paragraphs)
        max_paragraphs: Max number of paragraphs to include
        max_chars: Max total characters

    Returns:
        Formatted text string: "{title} {paragraphs}"
    """
    title = article.get("wikipedia_title", "")
    paragraphs = article.get("text", [])

    # Join first N paragraphs
    if isinstance(paragraphs, list):
        text = " ".join(paragraphs[:max_paragraphs])
    else:
        text = str(paragraphs)

    # Format and truncate (no labels, just title + text)
    formatted = f"{title} {text}"
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars]

    return formatted


class GTRIndexBuilder:
    """
    Builds Faiss index over KILT Wikipedia using GTR-T5-Base embeddings.
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/gtr-t5-base",
        wiki_arrow_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the index builder.

        Args:
            model_name_or_path: SentenceTransformer model name or path to fine-tuned model
            wiki_arrow_path: Path to Wikipedia Arrow dataset
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_name_or_path} on {device}...")
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model_name_or_path = model_name_or_path
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        # Load Wikipedia Arrow dataset
        print("Loading KILT Wikipedia Arrow dataset...")
        self.wiki = KILTWikipediaArrow(arrow_path=wiki_arrow_path)
        print(f"Wikipedia loaded: {len(self.wiki)} articles")

    def build_index(
        self,
        output_dir: str = "kilt_data/gtr_faiss_index",
        batch_size: int = 128,
        max_articles: Optional[int] = None,
        save_every: int = 500000
    ) -> None:
        """
        Build Faiss index and save to disk.

        Args:
            output_dir: Directory to save index files
            batch_size: Batch size for encoding (adjust based on GPU memory)
            max_articles: Max articles to index (None = all 5.9M)
            save_every: Save checkpoint every N articles
        """
        os.makedirs(output_dir, exist_ok=True)

        total_articles = len(self.wiki)
        if max_articles is not None:
            total_articles = min(total_articles, max_articles)

        print(f"\nBuilding index for {total_articles:,} articles...")
        print(f"Batch size: {batch_size}")
        print(f"Output directory: {output_dir}")

        # Initialize Faiss index (IndexFlatIP = Inner Product for cosine similarity)
        # Note: For cosine similarity with normalized vectors, IP = cosine similarity
        index = faiss.IndexFlatIP(self.embedding_dim)

        # Mapping from Faiss index to Wikipedia ID
        idx_to_wiki_id = []

        # Process in batches
        batch_texts = []
        batch_wiki_ids = []

        start_time = time.time()

        with tqdm(total=total_articles, desc="Indexing articles") as pbar:
            for i, article in enumerate(self.wiki.dataset):
                if max_articles is not None and i >= max_articles:
                    break

                # Format article text
                text = format_article_text(article)
                wiki_id = article.get("wikipedia_id", str(i))

                batch_texts.append(text)
                batch_wiki_ids.append(wiki_id)

                # Process batch
                if len(batch_texts) >= batch_size:
                    self._add_batch_to_index(
                        index, idx_to_wiki_id,
                        batch_texts, batch_wiki_ids
                    )
                    batch_texts = []
                    batch_wiki_ids = []
                    pbar.update(batch_size)

                # Save checkpoint
                if (i + 1) % save_every == 0 and i > 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}")
                    self._save_index(index, idx_to_wiki_id, checkpoint_path)
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (total_articles - i - 1) / rate / 60
                    print(f"\nCheckpoint saved. Rate: {rate:.1f} articles/sec, ETA: {remaining:.1f} min")

        # Process remaining articles
        if batch_texts:
            self._add_batch_to_index(
                index, idx_to_wiki_id,
                batch_texts, batch_wiki_ids
            )
            pbar.update(len(batch_texts))

        # Save final index
        print(f"\nSaving final index ({index.ntotal:,} vectors)...")
        self._save_index(index, idx_to_wiki_id, output_dir)

        elapsed = time.time() - start_time
        print(f"\nIndex building complete!")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Index size: {index.ntotal:,} vectors")
        print(f"Files saved to: {output_dir}")

    def _add_batch_to_index(
        self,
        index: faiss.Index,
        idx_to_wiki_id: List[str],
        texts: List[str],
        wiki_ids: List[str]
    ) -> None:
        """Add a batch of texts to the index."""
        # Encode texts
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )

        # Add to Faiss index
        index.add(embeddings.astype(np.float32))

        # Update ID mapping
        idx_to_wiki_id.extend(wiki_ids)

    def _save_index(
        self,
        index: faiss.Index,
        idx_to_wiki_id: List[str],
        output_dir: str
    ) -> None:
        """Save index and ID mapping to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save Faiss index
        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)

        # Save ID mapping
        mapping_path = os.path.join(output_dir, "idx_to_wiki_id.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump(idx_to_wiki_id, f)

        # Save metadata
        meta = {
            "total_vectors": index.ntotal,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name_or_path
        }
        meta_path = os.path.join(output_dir, "metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)


def test_index(index_dir: str = "kilt_data/gtr_faiss_index"):
    """Test the built index with a sample query."""
    print(f"\nTesting index from {index_dir}...")

    # Load index
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "idx_to_wiki_id.pkl"), 'rb') as f:
        idx_to_wiki_id = pickle.load(f)

    print(f"Index loaded: {index.ntotal:,} vectors")

    # Load model for query encoding
    model = SentenceTransformer("sentence-transformers/gtr-t5-base")

    # Test query
    query = "Who is the president of the United States?"
    print(f"\nTest query: {query}")

    # Encode and search
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k=5)

    print(f"\nTop 5 results:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        wiki_id = idx_to_wiki_id[idx]
        print(f"  {i+1}. wiki_id={wiki_id}, score={score:.4f}")

    # Load Wikipedia to show titles
    try:
        wiki = KILTWikipediaArrow()
        print("\nWith titles:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            wiki_id = idx_to_wiki_id[idx]
            article = wiki.get_by_id(wiki_id)
            if article:
                title = article.get("wikipedia_title", "Unknown")
                print(f"  {i+1}. {title} (score={score:.4f})")
    except Exception as e:
        print(f"Could not load Wikipedia for titles: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Build GTR-T5-Base Faiss index over KILT Wikipedia"
    )
    parser.add_argument(
        "--wiki-arrow-path",
        type=str,
        default=None,
        help="Path to Wikipedia Arrow dataset (default: kilt_data/kilt_wikipedia_arrow)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kilt_data/gtr_faiss_index",
        help="Directory to save index files"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="sentence-transformers/gtr-t5-base",
        help="Path to model (default: GTR-T5-Base, or path to fine-tuned model)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (reduce for low memory)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Max articles to index (default: all 5.9M)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500000,
        help="Save checkpoint every N articles"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test existing index, don't build"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu' (auto-detected if not specified)"
    )

    args = parser.parse_args()

    if args.test_only:
        test_index(args.output_dir)
    else:
        builder = GTRIndexBuilder(
            model_name_or_path=args.model_path,
            wiki_arrow_path=args.wiki_arrow_path,
            device=args.device
        )
        builder.build_index(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_articles=args.max_articles,
            save_every=args.save_every
        )

        # Test after building
        test_index(args.output_dir)


if __name__ == "__main__":
    main()
