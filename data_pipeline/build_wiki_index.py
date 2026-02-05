"""
Build KILT Wikipedia Index
==========================
Convert JSON files to a format that supports fast lookup by wikipedia_id.

Option 1: SQLite database (recommended, small footprint, fast queries)
Option 2: Arrow format (HuggingFace datasets format)
Option 3: In-memory dictionary + Pickle (requires large memory)
"""

import json
import os
import argparse
from tqdm import tqdm

WIKI_JSON_PATH = "./kilt_data/kilt_knowledgesource.json"
TOTAL_ARTICLES = 5903530


def build_sqlite_index(output_path="./kilt_data/kilt_wikipedia.db"):
    """
    Option 1: Build SQLite index (recommended)
    - Storage size: ~40GB
    - Query speed: millisecond-level
    - Memory requirement: very low
    """
    import sqlite3

    print("Building SQLite index...")
    print(f"Input: {WIKI_JSON_PATH}")
    print(f"Output: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE articles (
            wikipedia_id TEXT PRIMARY KEY,
            wikipedia_title TEXT,
            text TEXT,
            categories TEXT,
            anchors TEXT,
            wikidata_info TEXT
        )
    ''')

    # Batch insert
    batch_size = 10000
    batch = []

    with open(WIKI_JSON_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=TOTAL_ARTICLES, desc="Building index"):
            article = json.loads(line.strip())
            batch.append((
                article.get('wikipedia_id', ''),
                article.get('wikipedia_title', ''),
                json.dumps(article.get('text', []), ensure_ascii=False),
                article.get('categories', ''),
                json.dumps(article.get('anchors', []), ensure_ascii=False),
                json.dumps(article.get('wikidata_info', {}), ensure_ascii=False)
            ))

            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)',
                    batch
                )
                conn.commit()
                batch = []

    # Insert remaining data
    if batch:
        cursor.executemany('INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)', batch)
        conn.commit()

    # Create index
    print("Creating index...")
    cursor.execute('CREATE INDEX idx_title ON articles(wikipedia_title)')
    conn.commit()
    conn.close()

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Done! File size: {size_gb:.2f} GB")


def build_arrow_index(output_path="./kilt_data/kilt_wikipedia_arrow", wiki_path=None):
    """
    Option 2: Convert to Arrow format (HuggingFace datasets)
    - Storage size: ~30GB
    - Supports dataset.filter(), dataset.select() operations
    - Requires additional ID mapping (for fast ID-based lookup)
    """
    from datasets import Dataset
    import pickle

    wiki_json_path = wiki_path or WIKI_JSON_PATH

    print("Building Arrow format index...")
    print(f"Input: {wiki_json_path}")
    print(f"Output: {output_path}")

    def load_articles():
        with open(wiki_json_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=TOTAL_ARTICLES, desc="Reading JSON"):
                article = json.loads(line.strip())
                yield {
                    'wikipedia_id': article.get('wikipedia_id', ''),
                    'wikipedia_title': article.get('wikipedia_title', ''),
                    'text': article.get('text', []),
                    'categories': article.get('categories', ''),
                    'anchors': article.get('anchors', []),
                }

    print("Loading data and converting to Arrow format...")
    dataset = Dataset.from_generator(load_articles)

    print(f"Dataset size: {len(dataset)} articles")

    # Build ID -> index mapping (for fast lookup)
    print("Building wikipedia_id -> row index mapping...")
    id_to_idx = {}
    for i in tqdm(range(len(dataset)), desc="Building ID mapping"):
        wiki_id = dataset[i]['wikipedia_id']
        id_to_idx[wiki_id] = i

    # Save dataset
    print("Saving Arrow dataset...")
    dataset.save_to_disk(output_path)

    # Save ID mapping
    mapping_path = os.path.join(output_path, 'id_to_idx.pkl')
    print(f"Saving ID mapping to: {mapping_path}")
    with open(mapping_path, 'wb') as f:
        pickle.dump(id_to_idx, f)

    print(f"Done!")
    print(f"  Arrow data: {output_path}")
    print(f"  ID mapping: {mapping_path}")
    print(f"  Mapping size: {len(id_to_idx)} entries")


def test_sqlite_index(db_path="./kilt_data/kilt_wikipedia.db"):
    """Test SQLite query speed."""
    import sqlite3
    import time

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Test a few queries
    test_ids = ['45267196', '21247', '9232']  # Super Bowl, Neil Armstrong, Eiffel Tower

    for wiki_id in test_ids:
        start = time.time()
        cursor.execute('SELECT wikipedia_title, text FROM articles WHERE wikipedia_id = ?', (wiki_id,))
        result = cursor.fetchone()
        elapsed = (time.time() - start) * 1000

        if result:
            title, text = result
            text_preview = json.loads(text)[0][:100] if text else ''
            print(f"ID {wiki_id}: {title} ({elapsed:.2f}ms)")
            print(f"  Preview: {text_preview}...")
        else:
            print(f"ID {wiki_id}: Not found")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KILT Wikipedia Index")
    parser.add_argument('--format', choices=['sqlite', 'arrow'], default='sqlite',
                        help='Index format: sqlite (recommended) or arrow')
    parser.add_argument('--wiki-path', type=str, default=WIKI_JSON_PATH,
                        help='Path to KILT Wikipedia JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--test', action='store_true', help='Test existing SQLite index')
    args = parser.parse_args()

    # Update global path if provided
    if args.wiki_path:
        WIKI_JSON_PATH = args.wiki_path

    if args.test:
        test_sqlite_index()
    elif args.format == 'sqlite':
        output_path = args.output_dir if args.output_dir else "./kilt_data/kilt_wikipedia.db"
        build_sqlite_index(output_path)
    else:
        output_path = args.output_dir if args.output_dir else "./kilt_data/kilt_wikipedia_arrow"
        build_arrow_index(output_path, wiki_path=args.wiki_path)
