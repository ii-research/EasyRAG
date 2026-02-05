"""
Web RAG Panel Component
========================

Simple RAG demo using web search (no local retriever/Wikipedia needed).
Compares Direct (no search) vs Naive RAG (search + generate) side-by-side.

Features:
- No API key required (uses DDGS)
- Left: Direct answer (no search)
- Right: Naive RAG answer (with search results)
- Supports T5-base and T5Gemma2
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from nicegui import ui

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


# Model options
MODEL_OPTIONS = {
    "t5base": "T5-base (220M params)",
    "t5gemma": "T5Gemma2 (540M params)",
}

# Model configurations
MODEL_CONFIGS = {
    "t5base": {
        "name": "t5-base",
        "tokenizer_class": T5Tokenizer,
        "model_class": T5ForConditionalGeneration,
        "dtype": None,
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


@dataclass
class WebRAGResult:
    """Result from web RAG inference."""
    answer: str = ""
    latency_ms: float = 0
    search_results: List[Dict[str, str]] = field(default_factory=list)
    input_text: str = ""
    input_tokens: int = 0
    error: str = ""


class WebRAGEngine:
    """Lightweight engine for web RAG demo."""

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

    def __init__(self, model_type: str = "t5base"):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_input_length = self.MAX_INPUT_LENGTH.get(model_type, 512)
        self.max_snippet_len = self.MAX_SNIPPET_LEN.get(model_type, 150)
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return

        config = MODEL_CONFIGS[self.model_type]
        model_name = config["name"]

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
        self._loaded = True

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search web using DDGS."""
        if not HAS_DDGS:
            return []
        try:
            results = DDGS().text(query, max_results=top_k)
            return list(results) if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def format_input(self, query: str, search_results: List[Dict[str, str]]) -> str:
        """Format input in Naive RAG style with length control."""
        if not search_results:
            # Direct mode: just the question
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

    def generate(self, input_text: str) -> tuple:
        """Generate answer from input text. Returns (answer, input_tokens)."""
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        config = MODEL_CONFIGS[self.model_type]
        gen_kwargs = config["generation_kwargs"].copy()

        if self.model_type == "t5gemma":
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up T5Gemma2 output
        if self.model_type == "t5gemma":
            answer = re.sub(r'<unused\d+>', '', answer).strip()
            answer = re.sub(r'^answer:\s*', '', answer, flags=re.IGNORECASE).strip()

        return answer, len(input_ids[0])


class WebRAGPanel:
    """
    Web RAG comparison panel.

    Compares Direct (no search) vs Naive RAG (with web search) side-by-side.
    """

    def __init__(self):
        """Initialize the web RAG panel."""
        self.model_type = "t5base"
        self.engine: Optional[WebRAGEngine] = None
        self.is_loading = False
        self.is_generating = False

        # Fixed top_k (no need to configure)
        self.top_k = 5

        # Results
        self.direct_result = WebRAGResult()
        self.rag_result = WebRAGResult()

        self._build_ui()

    def _build_ui(self):
        """Build the panel UI."""
        with ui.column().classes("w-full"):
            # Title and description
            ui.label("Live Demo").classes("text-h5 mb-2")
            ui.label(
                "Compare Direct (no retrieval) vs Naive RAG (web search). "
                "No local data needed - uses web search for retrieval."
            ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            if not HAS_DDGS:
                ui.label(
                    "Warning: ddgs not installed. Run: pip install ddgs"
                ).classes("text-red-500 mb-4")

            # Model selection and load button
            with ui.card().classes("w-full mb-4"):
                with ui.row().classes("items-center gap-4"):
                    ui.label("Model:").classes("font-medium")
                    self.model_select = ui.select(
                        options=MODEL_OPTIONS,
                        value=self.model_type,
                        on_change=self._on_model_change,
                    ).classes("w-64")

                    self.load_btn = ui.button(
                        "Load Model",
                        on_click=self._load_model,
                    ).props("color=primary")

                    self.status_label = ui.label("Not loaded").classes(
                        "text-sm text-gray-500"
                    )

            # Query input
            with ui.card().classes("w-full mb-4"):
                ui.label("Question").classes("font-medium mb-2")
                self.query_input = ui.textarea(
                    placeholder="Enter your question here...",
                    value="What is the capital of France?",
                ).classes("w-full").props("rows=2")

                with ui.row().classes("mt-2 gap-2"):
                    self.generate_btn = ui.button(
                        "Generate",
                        on_click=self._generate,
                    ).props("color=primary").bind_enabled_from(
                        self, "is_generating", backward=lambda x: not x
                    )

                    self.spinner = ui.spinner(size="sm").bind_visibility_from(
                        self, "is_generating"
                    )

            # Results - side by side
            with ui.row().classes("w-full gap-4"):
                # Left: Direct (no search)
                with ui.card().classes("flex-1"):
                    ui.label("Direct (No Search)").classes(
                        "text-h6 mb-2 text-blue-600 dark:text-blue-400"
                    )
                    ui.label(
                        "Model answers directly without any context"
                    ).classes("text-xs text-gray-500 mb-2")

                    self.direct_answer_label = ui.label("").classes(
                        "text-lg font-medium p-3 bg-blue-50 dark:bg-blue-900 rounded"
                    )

                    with ui.row().classes("mt-2 text-xs text-gray-500"):
                        self.direct_latency = ui.label("")
                        self.direct_tokens = ui.label("")

                # Right: Naive RAG (with search)
                with ui.card().classes("flex-1"):
                    ui.label("Naive RAG (Web Search)").classes(
                        "text-h6 mb-2 text-green-600 dark:text-green-400"
                    )
                    ui.label(
                        f"Top-{self.top_k} web search results concatenated as context"
                    ).classes("text-xs text-gray-500 mb-2")

                    self.rag_answer_label = ui.label("").classes(
                        "text-lg font-medium p-3 bg-green-50 dark:bg-green-900 rounded"
                    )

                    with ui.row().classes("mt-2 text-xs text-gray-500"):
                        self.rag_latency = ui.label("")
                        self.rag_tokens = ui.label("")

            # Search results display (below)
            with ui.card().classes("w-full mt-4"):
                ui.label("Retrieved Web Content").classes("text-h6 mb-2")
                self.search_results_container = ui.column().classes("w-full")

    def _on_model_change(self, e):
        """Handle model selection change."""
        new_model = e.value
        if new_model != self.model_type:
            self.model_type = new_model
            # Unload current model
            if self.engine is not None:
                self.engine.unload()
                self.engine = None
            self.status_label.text = "Not loaded"
            self.status_label.classes(remove="text-green-500", add="text-gray-500")

    async def _load_model(self):
        """Load the selected model."""
        if self.is_loading:
            return

        self.is_loading = True
        self.load_btn.disable()
        self.status_label.text = "Loading..."
        self.status_label.classes(remove="text-gray-500 text-green-500", add="text-yellow-500")

        try:
            # Unload previous engine
            if self.engine is not None:
                self.engine.unload()

            # Create and load new engine
            self.engine = WebRAGEngine(model_type=self.model_type)

            # Run in thread to not block UI
            await asyncio.get_event_loop().run_in_executor(
                None, self.engine.load
            )

            self.status_label.text = f"Loaded ({self.engine.device})"
            self.status_label.classes(remove="text-yellow-500", add="text-green-500")

        except Exception as e:
            self.status_label.text = f"Error: {str(e)[:50]}"
            self.status_label.classes(remove="text-yellow-500", add="text-red-500")
            self.engine = None

        finally:
            self.is_loading = False
            self.load_btn.enable()

    async def _generate(self):
        """Generate answers for both modes."""
        if self.engine is None or not self.engine._loaded:
            ui.notify("Please load a model first", type="warning")
            return

        query = self.query_input.value.strip()
        if not query:
            ui.notify("Please enter a question", type="warning")
            return

        self.is_generating = True

        try:
            # Clear previous results
            self.direct_answer_label.text = "Generating..."
            self.rag_answer_label.text = "Searching & generating..."
            self.direct_latency.text = ""
            self.direct_tokens.text = ""
            self.rag_latency.text = ""
            self.rag_tokens.text = ""
            self.search_results_container.clear()

            # Run both in parallel using executor
            loop = asyncio.get_event_loop()

            # Direct mode
            direct_start = time.time()
            # Add "answer:" prompt for T5Gemma2
            if self.engine.model_type == "t5gemma":
                direct_input = f"question: {query}\nanswer:"
            else:
                direct_input = f"question: {query}"
            direct_answer, direct_tokens = await loop.run_in_executor(
                None, self.engine.generate, direct_input
            )
            direct_latency = (time.time() - direct_start) * 1000

            # Update direct result
            self.direct_answer_label.text = direct_answer or "(no answer)"
            self.direct_latency.text = f"Latency: {direct_latency:.0f}ms"
            self.direct_tokens.text = f"Input: {direct_tokens} tokens"

            # RAG mode - search first
            rag_start = time.time()
            search_results = await loop.run_in_executor(
                None, self.engine.search, query, self.top_k
            )
            search_latency = (time.time() - rag_start) * 1000

            # Format and generate
            rag_input = self.engine.format_input(query, search_results)
            gen_start = time.time()
            rag_answer, rag_tokens = await loop.run_in_executor(
                None, self.engine.generate, rag_input
            )
            gen_latency = (time.time() - gen_start) * 1000
            total_latency = (time.time() - rag_start) * 1000

            # Update RAG result
            self.rag_answer_label.text = rag_answer or "(no answer)"
            self.rag_latency.text = f"Latency: {total_latency:.0f}ms (search: {search_latency:.0f}ms)"
            self.rag_tokens.text = f"Input: {rag_tokens} tokens"

            # Display search results
            self._display_search_results(search_results)

        except Exception as e:
            ui.notify(f"Error: {str(e)}", type="negative")
            self.direct_answer_label.text = f"Error: {str(e)}"
            self.rag_answer_label.text = f"Error: {str(e)}"

        finally:
            self.is_generating = False

    def _display_search_results(self, results: List[Dict[str, str]]):
        """Display search results in the container."""
        self.search_results_container.clear()

        if not results:
            with self.search_results_container:
                ui.label("No search results found").classes("text-gray-500")
            return

        with self.search_results_container:
            for i, result in enumerate(results):
                title = result.get("title", "No title")
                body = result.get("body", "")
                href = result.get("href", "")

                with ui.card().classes("w-full mb-2 p-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.badge(f"[{i+1}]").props("color=primary")
                        if href:
                            ui.link(title, href, new_tab=True).classes(
                                "font-medium text-blue-600 dark:text-blue-400"
                            )
                        else:
                            ui.label(title).classes("font-medium")

                    ui.label(body).classes(
                        "text-sm text-gray-600 dark:text-gray-400 mt-1"
                    )
