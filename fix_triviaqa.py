"""
Fix TriviaQA data by mapping KILT question IDs to actual questions.

Based on official HuggingFace code:
https://huggingface.co/datasets/kilt_tasks#triviaqa_support_only

This script outputs to triviaqa_fixed/ directory (intermediate step).
Run filter_kilt_data.py afterwards to filter samples without provenance.

Usage:
    python fix_triviaqa.py --output_dir /data/ZHOUXUANCHEN/kilt_data/triviaqa_fixed
"""

import argparse
import json
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Fix TriviaQA data")
    parser.add_argument("--output_dir", type=str, default="kilt_data/triviaqa_fixed",
                        help="Output directory for fixed data (intermediate, before filtering)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets (official code)
    print("Loading KILT TriviaQA...")
    kilt_triviaqa = load_dataset("facebook/kilt_tasks", name="triviaqa_support_only")

    print("Loading original TriviaQA...")
    trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')

    # Official mapping function
    def add_missing_data(x, trivia_qa_subset, triviaqa_map):
        i = triviaqa_map[x['id']]
        x['input'] = trivia_qa_subset[i]['question']
        # output is a list, add original_answer to first output if exists
        if x['output'] and len(x['output']) > 0:
            x['output'][0]['original_answer'] = trivia_qa_subset[i]['answer']['value']
        return x

    # Process each split (official code)
    for k in ['train', 'validation', 'test']:
        print(f"\nProcessing {k}...")

        # Build mapping
        triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])

        # Filter and map
        kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
        kilt_triviaqa[k] = kilt_triviaqa[k].map(
            add_missing_data,
            fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map)
        )

        print(f"  {k}: {len(kilt_triviaqa[k])} samples")

        # Save to JSONL
        output_path = os.path.join(args.output_dir, f"triviaqa_support_only_{k}.jsonl")
        with open(output_path, "w") as f:
            for sample in kilt_triviaqa[k]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"  Saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
