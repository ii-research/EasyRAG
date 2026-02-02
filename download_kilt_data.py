"""
KILT Dataset Download Script
============================
Download facebook/kilt_tasks and facebook/kilt_wikipedia datasets to local storage.

Usage: python download_kilt_data.py
"""

from datasets import load_dataset
import os

# Set cache directory (optional, defaults to ~/.cache/huggingface/datasets)
CACHE_DIR = "./kilt_data"

def download_kilt_tasks():
    """
    Download KILT Tasks dataset.
    Contains 11 sub-tasks, each with train/validation/test splits.
    """
    print("=" * 60)
    print("Downloading facebook/kilt_tasks dataset...")
    print("=" * 60)

    # KILT tasks - 3 QA tasks only
    kilt_task_names = [
        "nq",                    # Natural Questions (91.7k rows)
        "hotpotqa",              # HotpotQA (100k rows)
        "triviaqa_support_only", # TriviaQA (73.8k rows) - ID only, requires additional processing
    ]

    for task_name in kilt_task_names:
        print(f"\nDownloading: {task_name}")
        try:
            # Download all splits (train, validation, test)
            dataset = load_dataset(
                "facebook/kilt_tasks",
                name=task_name,
                cache_dir=CACHE_DIR
            )
            print(f"  {task_name} downloaded successfully!")
            print(f"    Available splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"    - {split_name}: {len(split_data)} samples")
        except Exception as e:
            print(f"  {task_name} download failed: {e}")


def download_kilt_wikipedia():
    """
    Download KILT Wikipedia knowledge base.
    Note: This dataset is large (~37GB download, ~29GB after extraction).
    Contains 5,903,530 Wikipedia articles.

    Since HuggingFace no longer supports loading script, download directly from Facebook servers.
    """
    print("\n" + "=" * 60)
    print("Downloading facebook/kilt_wikipedia dataset...")
    print("Warning: This dataset is large (~35GB), download will take a while!")
    print("=" * 60)

    import urllib.request
    import sys

    # Facebook official download link
    url = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
    output_path = os.path.join(CACHE_DIR, "kilt_knowledgesource.json")

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        file_size = os.path.getsize(output_path) / (1024**3)
        print(f"File size: {file_size:.2f} GB")
        return output_path

    print(f"Download URL: {url}")
    print(f"Saving to: {output_path}")

    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = downloaded / total_size * 100 if total_size > 0 else 0
            downloaded_gb = downloaded / (1024**3)
            total_gb = total_size / (1024**3)
            sys.stdout.write(f"\rDownload progress: {percent:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, show_progress)
        print(f"\nkilt_wikipedia downloaded successfully!")
        print(f"  Saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"\nkilt_wikipedia download failed: {e}")
        print("\nYou can also download manually:")
        print(f"  wget {url} -O {output_path}")
        return None


def download_kilt_wikipedia_streaming():
    """
    Load KILT Wikipedia in streaming mode (without downloading complete data).
    Read directly from local JSON file.
    """
    print("\n" + "=" * 60)
    print("Checking local kilt_wikipedia file...")
    print("=" * 60)

    json_path = os.path.join(CACHE_DIR, "kilt_knowledgesource.json")

    if not os.path.exists(json_path):
        print(f"Local file does not exist: {json_path}")
        print("Please run first: python download_kilt_data.py --wikipedia-only")
        return

    import json

    try:
        print(f"Loading from local file: {json_path}")
        print("\nPreview first 3 entries:")

        with open(json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                article = json.loads(line.strip())
                print(f"\n--- Document {i+1} ---")
                print(f"  wikipedia_id: {article.get('wikipedia_id', 'N/A')}")
                print(f"  title: {article.get('wikipedia_title', 'N/A')}")
                text = article.get('text', [])
                if text and len(text) > 0:
                    preview = text[0][:200] if len(text[0]) > 200 else text[0]
                    print(f"  text preview: {preview}...")

        print("\nLocal file is available!")
    except Exception as e:
        print(f"Read failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download KILT Dataset")
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only download kilt_tasks, not wikipedia"
    )
    parser.add_argument(
        "--wikipedia-only",
        action="store_true",
        help="Only download kilt_wikipedia"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to preview wikipedia (without full download)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./kilt_data",
        help="Dataset cache directory (default: ./kilt_data)"
    )
    args = parser.parse_args()

    CACHE_DIR = args.cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Datasets will be saved to: {os.path.abspath(CACHE_DIR)}")

    if args.tasks_only:
        download_kilt_tasks()
    elif args.wikipedia_only:
        if args.streaming:
            download_kilt_wikipedia_streaming()
        else:
            download_kilt_wikipedia()
    else:
        # Download all by default
        download_kilt_tasks()
        if args.streaming:
            download_kilt_wikipedia_streaming()
        else:
            download_kilt_wikipedia()

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nExample usage in app.py:")
    print("""
from datasets import load_dataset

# ============ Load KILT Tasks ============
# Load a specific task (e.g., Natural Questions)
kilt_nq = load_dataset("facebook/kilt_tasks", "nq", cache_dir="./kilt_data")
train_data = kilt_nq["train"]
validation_data = kilt_nq["validation"]
test_data = kilt_nq["test"]

# Access data example
for example in train_data:
    question = example["input"]           # Question
    outputs = example["output"]           # Answer list
    for output in outputs:
        answer = output["answer"]         # Answer text
        provenance = output["provenance"] # Source passage information
    break

# ============ Load KILT Wikipedia ============
# Load from local JSON file (requires download first)
import json
wiki_path = "./kilt_data/kilt_knowledgesource.json"

# Stream reading (recommended, saves memory)
with open(wiki_path, 'r', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        wiki_id = article["wikipedia_id"]
        title = article["wikipedia_title"]
        paragraphs = article["text"]  # List of paragraphs
        break

# Or use the wrapper loader
from kilt_loader import load_kilt_wikipedia
for article in load_kilt_wikipedia():
    print(article["wikipedia_title"])
    break

# ============ 11 Available KILT Tasks ============
# nq, aidayago2, cweb, eli5, fever, hotpotqa,
# structured_zeroshot, trex, triviaqa_support_only, wned, wow
""")
