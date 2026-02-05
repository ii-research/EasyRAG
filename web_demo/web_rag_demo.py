"""
Web Search RAG Demo
====================

A simplified RAG demo using web search API instead of local GTR retriever + Wikipedia.
No API key required (uses DDGS library).

Modes:
- naive_rag: Search web -> concatenate results -> generate answer
- direct: No search, directly ask model (for comparison)

Models:
- t5base: T5-base (220M params)
- t5gemma: T5Gemma2-270M-270M (540M params, BF16)

Usage:
    # Naive RAG with T5-base
    python web_rag_demo.py --query "What is the capital of France?" --mode naive_rag --model t5base

    # Direct mode (no search)
    python web_rag_demo.py --query "What is the capital of France?" --mode direct --model t5base

    # With T5Gemma2
    python web_rag_demo.py --query "What is the capital of France?" --model t5gemma

    # Interactive mode
    python web_rag_demo.py --interactive --model t5base

    # Adjust search results count
    python web_rag_demo.py --query "..." --top_k 10

Requirements:
    pip install ddgs transformers torch
"""

import argparse
import re
import time
from typing import Dict, List, Optional, Any

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# DDGS for web search
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("Warning: ddgs not installed. Run: pip install ddgs")


class WebRAGDemo:
    """
    Web Search RAG Demo with T5-base or T5Gemma2.

    Uses DDGS library for web search (no API key required).
    """

    # Model configurations
    MODEL_CONFIGS = {
        "t5base": {
            "name": "t5-base",
            "tokenizer_class": T5Tokenizer,
            "model_class": T5ForConditionalGeneration,
            "dtype": None,  # FP32
            "generation_kwargs": {
                "max_length": 64,
                "num_beams": 4,
                "early_stopping": True,
            }
        },
        "t5gemma": {
            "name": "google/t5gemma-2-270m-270m",
            "tokenizer_class": AutoTokenizer,
            "model_class": AutoModelForSeq2SeqLM,
            "dtype": torch.bfloat16,
            "generation_kwargs": {
                "max_new_tokens": 64,
                "num_beams": 4,
                "do_sample": False,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
            }
        }
    }

    # Max input length per model
    MAX_INPUT_LENGTH = {
        "t5base": 512,      # T5-base limit
        "t5gemma": 2048,    # T5Gemma2 supports 128K, but 2048 is enough
    }

    # Max snippet length per model (characters)
    MAX_SNIPPET_LEN = {
        "t5base": 150,      # ~40 tokens each, 5 snippets = ~200 tokens
        "t5gemma": 500,     # ~130 tokens each, 5 snippets = ~650 tokens
    }

    def __init__(
        self,
        model_type: str = "t5base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_input_length: int = None,
    ):
        """
        Initialize the Web RAG Demo.

        Args:
            model_type: "t5base" or "t5gemma"
            device: "cuda" or "cpu"
            max_input_length: Maximum input tokens (auto-set based on model if None)
        """
        self.model_type = model_type
        self.device = device
        self.max_input_length = max_input_length or self.MAX_INPUT_LENGTH.get(model_type, 512)
        self.max_snippet_len = self.MAX_SNIPPET_LEN.get(model_type, 150)

        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        config = self.MODEL_CONFIGS[self.model_type]
        model_name = config["name"]

        print(f"\nLoading {model_name}...")
        start = time.time()

        # Load tokenizer
        self.tokenizer = config["tokenizer_class"].from_pretrained(model_name)

        # Load model
        if config["dtype"] is not None:
            self.model = config["model_class"].from_pretrained(
                model_name,
                torch_dtype=config["dtype"]
            )
        else:
            self.model = config["model_class"].from_pretrained(model_name)

        # Fix decoder_start_token_id for T5Gemma2
        if self.model_type == "t5gemma":
            if self.model.config.decoder_start_token_id is None:
                self.model.config.decoder_start_token_id = self.model.config.bos_token_id or 2

        self.model.to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        print(f"Model loaded in {elapsed:.1f}s")
        print(f"Device: {self.device}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {total_params:,}")

        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Search the web using DDGS.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with keys: title, href, body
        """
        if not HAS_DDGS:
            raise ImportError("ddgs not installed. Run: pip install ddgs")

        try:
            results = DDGS().text(query, max_results=top_k)
            return list(results) if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def format_input(self, query: str, search_results: List[Dict[str, str]]) -> str:
        """
        Format input in Naive RAG style (concatenated passages).

        Format:
            question: {query}
            context: [1] {body1}
            [2] {body2}
            ...
            answer:  (for T5Gemma2)
        """
        if not search_results:
            # Direct mode
            if self.model_type == "t5gemma":
                return f"question: {query}\nanswer:"
            return f"question: {query}"

        contexts = []
        for i, result in enumerate(search_results):
            body = result.get("body", "")
            if body:
                # Truncate long snippets based on model type
                if len(body) > self.max_snippet_len:
                    body = body[:self.max_snippet_len] + "..."
                contexts.append(f"[{i+1}] {body}")

        context_str = "\n".join(contexts)

        # Add "answer:" prompt for T5Gemma2 to signal it should answer now
        if self.model_type == "t5gemma":
            return f"question: {query}\ncontext: {context_str}\nanswer:"
        return f"question: {query}\ncontext: {context_str}"

    def generate(
        self,
        query: str,
        mode: str = "naive_rag",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate answer for a query.

        Args:
            query: User question
            mode: "naive_rag" (search + generate) or "direct" (no search)
            top_k: Number of search results (only for naive_rag mode)

        Returns:
            Dict with: query, answer, search_results, input_text, latency_ms
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Search (only for naive_rag mode)
        if mode == "naive_rag":
            search_start = time.time()
            search_results = self.search(query, top_k)
            search_latency = (time.time() - search_start) * 1000
            input_text = self.format_input(query, search_results)
        else:  # direct mode
            search_results = []
            search_latency = 0
            input_text = f"question: {query}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        config = self.MODEL_CONFIGS[self.model_type]
        gen_kwargs = config["generation_kwargs"].copy()

        # Add eos_token_id for T5Gemma2
        if self.model_type == "t5gemma":
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Filter out <unused...> tokens that T5Gemma2 sometimes generates
        if self.model_type == "t5gemma":
            answer = re.sub(r'<unused\d+>', '', answer).strip()
            # Remove duplicate "answer:" prefix if present
            answer = re.sub(r'^answer:\s*', '', answer, flags=re.IGNORECASE).strip()

        total_latency = (time.time() - start_time) * 1000

        return {
            "query": query,
            "answer": answer,
            "search_results": search_results,
            "input_text": input_text,
            "mode": mode,
            "model": self.model_type,
            "top_k": top_k if mode == "naive_rag" else 0,
            "latency_ms": {
                "total": total_latency,
                "search": search_latency,
                "generation": total_latency - search_latency,
            },
            "input_tokens": len(input_ids[0]),
        }


def print_result(result: Dict[str, Any], verbose: bool = True) -> None:
    """Pretty print the result."""
    print("\n" + "=" * 60)
    print(f"Query: {result['query']}")
    print(f"Mode: {result['mode']} | Model: {result['model']}", end="")
    if result['mode'] == 'naive_rag':
        print(f" | Top-K: {result['top_k']}")
    else:
        print()
    print("=" * 60)

    # Search results
    if result['search_results'] and verbose:
        print("\nSearch Results:")
        for i, r in enumerate(result['search_results']):
            title = r.get('title', 'No title')
            body = r.get('body', '')[:150] + "..." if len(r.get('body', '')) > 150 else r.get('body', '')
            print(f"  [{i+1}] {title}")
            print(f"      {body}")
        print()

    # Answer
    print(f"Answer: {result['answer']}")

    # Latency
    latency = result['latency_ms']
    print(f"\nLatency: {latency['total']:.0f}ms total", end="")
    if result['mode'] == 'naive_rag':
        print(f" (search: {latency['search']:.0f}ms, generation: {latency['generation']:.0f}ms)")
    else:
        print()

    print(f"Input tokens: {result['input_tokens']}")
    print("=" * 60)


def interactive_mode(demo: WebRAGDemo, mode: str, top_k: int, verbose: bool) -> None:
    """Run interactive Q&A loop."""
    print("\n" + "=" * 60)
    print("Interactive Web RAG Demo")
    print(f"Model: {demo.model_type} | Mode: {mode} | Top-K: {top_k}")
    print("Type 'quit' or 'exit' to stop, 'mode' to switch modes")
    print("=" * 60)

    current_mode = mode

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if query.lower() == 'mode':
                current_mode = 'direct' if current_mode == 'naive_rag' else 'naive_rag'
                print(f"Switched to {current_mode} mode")
                continue

            result = demo.generate(query, mode=current_mode, top_k=top_k)
            print_result(result, verbose=verbose)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Web Search RAG Demo - No API key required",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Query
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Question to answer (if not interactive)")

    # Mode
    parser.add_argument("--mode", "-m", type=str, default="naive_rag",
                        choices=["naive_rag", "direct"],
                        help="naive_rag: search + generate, direct: no search")

    # Model
    parser.add_argument("--model", type=str, default="t5base",
                        choices=["t5base", "t5gemma"],
                        help="Model to use")

    # Search
    parser.add_argument("--top_k", "-k", type=int, default=5,
                        help="Number of search results to use")

    # Input length
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum input tokens")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Interactive
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive Q&A mode")

    # Verbose
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Show search results")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide search results")

    args = parser.parse_args()

    # Check DDGS
    if not HAS_DDGS and args.mode == "naive_rag":
        print("Error: ddgs not installed. Run: pip install ddgs")
        print("Or use --mode direct to skip search.")
        return

    # Check query or interactive
    if not args.interactive and not args.query:
        print("Error: Please provide --query or use --interactive mode")
        parser.print_help()
        return

    # Initialize demo
    demo = WebRAGDemo(
        model_type=args.model,
        device=args.device,
        max_input_length=args.max_input_length,
    )

    # Load model
    demo.load()

    verbose = not args.quiet

    if args.interactive:
        interactive_mode(demo, args.mode, args.top_k, verbose)
    else:
        result = demo.generate(args.query, mode=args.mode, top_k=args.top_k)
        print_result(result, verbose=verbose)


if __name__ == "__main__":
    main()
