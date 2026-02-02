"""
Inference Panel Component
==========================

Interactive Q&A panel for FiD-Light/FiD Pure/Stochastic RAG inference.

Features:
- Algorithm selection (FiD-Light, FiD Pure, Stochastic RAG)
- Model backbone selection (T5-base, T5Gemma2)
- Adjustable parameters (compression_k, num_passages, n_selected)
- Model/retriever/index path configuration
- Question input
- Answer display with source highlighting (for FiD-Light/SR)
- Retrieved passages view with reranker scores (for SR)
"""

from nicegui import ui
from typing import Optional, Callable
import asyncio


# Algorithm options with descriptions
ALGORITHM_OPTIONS = {
    "fidlight": "FiD-Light (source pointer)",
    "fid_pure": "FiD Pure (answer only)",
    "stochastic_rag": "Stochastic RAG (reranker)",
}

# Model backbone options
MODEL_OPTIONS = {
    "t5base": "T5-base (220M params)",
    "t5gemma": "T5Gemma2 (270M params, BF16)",
}

# Default parameters per algorithm
ALGORITHM_DEFAULTS = {
    "fidlight": {"k": 64, "n_passages": 40, "n_selected": 40},
    "fid_pure": {"k": 250, "n_passages": 100, "n_selected": 100},
    "stochastic_rag": {"k": 64, "n_passages": 40, "n_selected": 10},
}


class InferencePanel:
    """
    Interactive inference demonstration panel.

    Supports all 6 model types:
    - FiD-Light (T5-base, T5Gemma2)
    - FiD Pure (T5-base, T5Gemma2)
    - Stochastic RAG (T5-base, T5Gemma2)
    """

    def __init__(
        self,
        default_checkpoint: str = "checkpoints/fidlight_paper/final",
        default_index: str = "kilt_data/gtr_faiss_index",
        default_wiki: str = "",  # Deprecated
    ):
        """
        Initialize the inference panel.

        Args:
            default_checkpoint: Default checkpoint path
            default_index: Default index path
            default_wiki: Deprecated, not used
        """
        self.checkpoint_path = default_checkpoint
        self.index_path = default_index
        self.retriever_path = ""
        self.wiki_arrow_path = ""
        self.algorithm = "fidlight"
        self.model_type = "t5base"

        # Adjustable parameters
        self.compression_k = 64
        self.num_passages = 40
        self.n_selected = 10

        self.engine = None
        self.is_loading = False
        self.is_loaded = False

        self._build_ui()

    def _build_ui(self):
        """Build the inference panel UI."""
        with ui.column().classes("w-full"):
            # Title
            ui.label("Inference Demo").classes("text-h5 mb-4")

            # Configuration card
            with ui.card().classes("w-full mb-4"):
                ui.label("Model Configuration").classes("text-h6 mb-2")

                with ui.column().classes("w-full gap-3"):
                    # Algorithm selection
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Algorithm:").classes("w-28 font-bold")
                        self.algorithm_select = ui.select(
                            options=ALGORITHM_OPTIONS,
                            value="fidlight",
                            on_change=self._on_algorithm_change,
                        ).classes("flex-grow")

                    # Algorithm description
                    self.algorithm_desc = ui.label(
                        self._get_algorithm_description("fidlight")
                    ).classes("text-xs text-gray-500 dark:text-gray-400 ml-32 -mt-2")

                    # Model backbone selection
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Model:").classes("w-28 font-bold")
                        self.model_select = ui.select(
                            options=MODEL_OPTIONS,
                            value="t5base",
                            on_change=self._on_model_change,
                        ).classes("flex-grow")

                    ui.separator().classes("my-2")

                    # Checkpoint path
                    with ui.row().classes("w-full items-center"):
                        ui.label("Checkpoint:").classes("w-28")
                        self.checkpoint_input = ui.input(
                            value=self.checkpoint_path,
                            placeholder="checkpoints/fidlight_paper/final",
                        ).classes("flex-grow")

                    # Retriever path (optional)
                    with ui.row().classes("w-full items-center"):
                        ui.label("Retriever:").classes("w-28")
                        self.retriever_input = ui.input(
                            value="",
                            placeholder="(optional) checkpoints/gtr_finetuned",
                        ).classes("flex-grow")
                    ui.label(
                        "Leave empty to use default GTR-T5-Base"
                    ).classes("text-xs text-gray-500 dark:text-gray-400 ml-32 -mt-2")

                    # Index path
                    with ui.row().classes("w-full items-center"):
                        ui.label("Index:").classes("w-28")
                        self.index_input = ui.input(
                            value=self.index_path,
                            placeholder="kilt_data/gtr_faiss_index",
                        ).classes("flex-grow")

                    # Wiki Arrow path (optional)
                    with ui.row().classes("w-full items-center"):
                        ui.label("Wiki Arrow:").classes("w-28")
                        self.wiki_input = ui.input(
                            value="",
                            placeholder="(optional) kilt_data/wiki_arrow",
                        ).classes("flex-grow")
                    ui.label(
                        "Leave empty for default path, or specify custom wiki_arrow location"
                    ).classes("text-xs text-gray-500 dark:text-gray-400 ml-32 -mt-2")

                    ui.separator().classes("my-2")

                    # Advanced parameters section
                    with ui.expansion("Advanced Parameters", icon="tune").classes("w-full"):
                        with ui.column().classes("w-full gap-2 p-2"):
                            # Compression k
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label("Compression k:").classes("w-32")
                                self.k_input = ui.number(
                                    value=64,
                                    min=1,
                                    max=512,
                                    step=1,
                                    on_change=self._on_param_change,
                                ).classes("w-24")
                                ui.label("tokens per passage (64 for FiD-Light, 250 for FiD Pure)").classes(
                                    "text-xs text-gray-500 dark:text-gray-400"
                                )

                            # Num passages
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label("Num Passages:").classes("w-32")
                                self.n_passages_input = ui.number(
                                    value=40,
                                    min=1,
                                    max=200,
                                    step=1,
                                    on_change=self._on_param_change,
                                ).classes("w-24")
                                ui.label("passages to retrieve (40 for FiD-Light, 100 for FiD Pure)").classes(
                                    "text-xs text-gray-500 dark:text-gray-400"
                                )

                            # N selected (only for Stochastic RAG)
                            self.n_selected_row = ui.row().classes("w-full items-center gap-2")
                            with self.n_selected_row:
                                ui.label("N Selected:").classes("w-32")
                                self.n_selected_input = ui.number(
                                    value=10,
                                    min=1,
                                    max=100,
                                    step=1,
                                    on_change=self._on_param_change,
                                ).classes("w-24")
                                ui.label("passages after reranking (Stochastic RAG only)").classes(
                                    "text-xs text-gray-500 dark:text-gray-400"
                                )
                            self.n_selected_row.set_visibility(False)  # Hidden by default

                            # Reset to defaults button
                            ui.button(
                                "Reset to Algorithm Defaults",
                                icon="refresh",
                                on_click=self._reset_params_to_defaults,
                            ).props("flat dense size=sm")

                    # Load button
                    with ui.row().classes("w-full justify-end mt-2 items-center"):
                        self.status_label = ui.label("Not loaded").classes(
                            "mr-4 text-gray-500 dark:text-gray-400"
                        )
                        self.load_button = ui.button(
                            "Load Model",
                            icon="download",
                            on_click=self._on_load_model,
                        ).props("color=primary")

            # Question input card
            with ui.card().classes("w-full mb-4"):
                ui.label("Ask a Question").classes("text-h6 mb-2")

                self.question_input = ui.textarea(
                    placeholder="Enter your question here...",
                ).classes("w-full")

                # Sample questions
                with ui.row().classes("w-full flex-wrap gap-2 mt-2"):
                    ui.label("Try:").classes("text-gray-500 dark:text-gray-400")
                    sample_questions = [
                        "Who discovered penicillin?",
                        "What is the capital of France?",
                        "When was the Eiffel Tower built?",
                    ]
                    for q in sample_questions:
                        ui.button(
                            q[:25] + "..." if len(q) > 25 else q,
                            on_click=lambda q=q: self._set_question(q),
                        ).props("flat dense size=sm")

                with ui.row().classes("w-full justify-end mt-2"):
                    self.generate_button = ui.button(
                        "Generate Answer",
                        icon="send",
                        on_click=self._on_generate,
                    ).props("color=primary")
                    self.generate_button.disable()

            # Results card
            with ui.card().classes("w-full"):
                ui.label("Results").classes("text-h6 mb-2")

                with ui.column().classes("w-full"):
                    # Answer section
                    ui.label("Answer:").classes("font-bold")
                    self.answer_container = ui.column().classes("w-full")
                    with self.answer_container:
                        self.answer_label = ui.label("...").classes(
                            "text-lg p-2 bg-gray-100 dark:bg-gray-800 rounded w-full"
                        )
                        with ui.row().classes("w-full items-center gap-4"):
                            self.latency_label = ui.label("").classes(
                                "text-xs text-gray-500 dark:text-gray-400"
                            )
                            self.raw_output_label = ui.label("").classes(
                                "text-xs text-gray-400 dark:text-gray-500 italic"
                            )

                    ui.separator().classes("my-4")

                    # Source indices (hidden for FiD Pure)
                    self.source_section = ui.column().classes("w-full")
                    with self.source_section:
                        ui.label("Source Passages:").classes("font-bold")
                        self.source_container = ui.row().classes("w-full flex-wrap gap-2")

                    ui.separator().classes("my-4")

                    # Retrieved passages
                    ui.label("Retrieved Passages:").classes("font-bold")
                    with ui.scroll_area().classes("h-64"):
                        self.passages_container = ui.column().classes("w-full")

    def _get_algorithm_description(self, algo: str) -> str:
        """Get detailed description for algorithm."""
        defaults = ALGORITHM_DEFAULTS.get(algo, {})
        descriptions = {
            "fidlight": f"Compresses to k={defaults.get('k', 64)} tokens. Output: 'index: X text: answer'",
            "fid_pure": f"Uses k={defaults.get('k', 250)} tokens (full). Output: answer only (no source)",
            "stochastic_rag": f"Reranks {defaults.get('n_passages', 40)} to {defaults.get('n_selected', 10)}. Output: 'index: X text: answer'",
        }
        return descriptions.get(algo, "")

    def _on_algorithm_change(self, e):
        """Handle algorithm selection change."""
        self.algorithm = e.value
        self.algorithm_desc.text = self._get_algorithm_description(e.value)

        # Update source section visibility
        has_source = e.value != "fid_pure"
        self.source_section.set_visibility(has_source)

        # Show/hide n_selected input based on algorithm
        self.n_selected_row.set_visibility(e.value == "stochastic_rag")

        # Update default parameters
        self._reset_params_to_defaults()

        # Reset loaded state when config changes
        if self.is_loaded:
            self._mark_reload_required()

    def _on_model_change(self, e):
        """Handle model type selection change."""
        self.model_type = e.value

        # Reset loaded state when config changes
        if self.is_loaded:
            self._mark_reload_required()

    def _on_param_change(self, e):
        """Handle parameter change."""
        if self.is_loaded:
            self._mark_reload_required()

    def _mark_reload_required(self):
        """Mark that a reload is required due to config change."""
        self.is_loaded = False
        self.status_label.text = "Config changed - reload required"
        self.status_label.classes(remove="text-green-600")
        self.status_label.classes(add="text-orange-500")
        self.generate_button.disable()

    def _reset_params_to_defaults(self):
        """Reset parameters to algorithm defaults."""
        algo = self.algorithm_select.value if hasattr(self, 'algorithm_select') else "fidlight"
        defaults = ALGORITHM_DEFAULTS.get(algo, ALGORITHM_DEFAULTS["fidlight"])

        if hasattr(self, 'k_input'):
            self.k_input.value = defaults["k"]
        if hasattr(self, 'n_passages_input'):
            self.n_passages_input.value = defaults["n_passages"]
        if hasattr(self, 'n_selected_input'):
            self.n_selected_input.value = defaults["n_selected"]

    def _set_question(self, question: str):
        """Set the question input."""
        self.question_input.value = question

    async def _on_load_model(self):
        """Handle load model button click."""
        if self.is_loading:
            return

        self.is_loading = True
        self.load_button.disable()
        self.status_label.text = "Loading..."
        self.status_label.classes(
            remove="text-gray-500 dark:text-gray-400 text-green-600 dark:text-green-400 text-red-600 dark:text-red-400 text-orange-500"
        )
        self.status_label.classes(add="text-blue-600")

        # Get values from inputs
        self.checkpoint_path = self.checkpoint_input.value
        self.index_path = self.index_input.value
        self.retriever_path = self.retriever_input.value.strip() or None
        self.wiki_arrow_path = self.wiki_input.value.strip() or None
        self.algorithm = self.algorithm_select.value
        self.model_type = self.model_select.value

        # Get advanced parameters
        self.compression_k = int(self.k_input.value) if self.k_input.value else 64
        self.num_passages = int(self.n_passages_input.value) if self.n_passages_input.value else 40
        self.n_selected = int(self.n_selected_input.value) if self.n_selected_input.value else 10

        # Load in background to not block UI
        try:
            # Import here to avoid circular imports
            from ..inference_demo import InferenceEngine, reset_inference_engine

            # Reset existing engine
            reset_inference_engine()

            # Create new engine with selected configuration
            self.engine = InferenceEngine(
                checkpoint_path=self.checkpoint_path,
                algorithm=self.algorithm,
                model_type=self.model_type,
                retriever_path=self.retriever_path,
                index_path=self.index_path if self.index_path else None,
                wiki_arrow_path=self.wiki_arrow_path,
                compression_k=self.compression_k,
                num_passages=self.num_passages,
                n_selected=self.n_selected,
            )

            # Load model (this can be slow)
            await asyncio.get_event_loop().run_in_executor(
                None, self.engine.load
            )

            if self.engine.is_loaded:
                self.is_loaded = True
                params_str = f"k={self.compression_k}, n={self.num_passages}"
                if self.algorithm == "stochastic_rag":
                    params_str += f", sel={self.n_selected}"
                self.status_label.text = f"Loaded ({self.algorithm}, {params_str})"
                self.status_label.classes(remove="text-blue-600")
                self.status_label.classes(add="text-green-600")
                self.generate_button.enable()
                ui.notify("Model loaded successfully", type="positive")
            else:
                # Show detailed error message
                error_msg = getattr(self.engine, 'load_error', None) or "Unknown error"
                self.status_label.text = f"Load failed: {error_msg[:50]}..."
                self.status_label.classes(remove="text-blue-600")
                self.status_label.classes(add="text-red-600")
                ui.notify(f"Failed to load model: {error_msg}", type="negative")

        except Exception as e:
            self.status_label.text = f"Error: {str(e)[:40]}..."
            self.status_label.classes(remove="text-blue-600")
            self.status_label.classes(add="text-red-600")
            ui.notify(f"Error loading model: {e}", type="negative")

        finally:
            self.is_loading = False
            self.load_button.enable()

    async def _on_generate(self):
        """Handle generate button click."""
        if not self.is_loaded or self.engine is None:
            ui.notify("Please load the model first", type="warning")
            return

        question = self.question_input.value.strip()
        if not question:
            ui.notify("Please enter a question", type="warning")
            return

        self.generate_button.disable()
        self.answer_label.text = "Generating..."

        try:
            # Run inference in background
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.engine.answer_question, question
            )

            # Update answer
            self.answer_label.text = result.answer if result.answer else "(No answer generated)"
            self.latency_label.text = f"Latency: {result.latency_ms:.0f}ms"
            self.raw_output_label.text = f"Raw: {result.raw_output[:50]}..." if len(result.raw_output) > 50 else f"Raw: {result.raw_output}"

            # Update source indices (only for algorithms with source pointer)
            self.source_container.clear()
            if self.engine.has_source_pointer:
                self.source_section.set_visibility(True)
                with self.source_container:
                    if result.source_indices:
                        for idx in result.source_indices:
                            ui.badge(f"Passage {idx}").props("color=primary")
                    else:
                        ui.label("No sources identified").classes("text-gray-500 dark:text-gray-400")
            else:
                self.source_section.set_visibility(False)

            # Update passages
            self.passages_container.clear()
            with self.passages_container:
                for i, passage in enumerate(result.passages):  # Show all passages
                    is_source = (i + 1) in result.source_indices
                    with ui.card().classes(
                        f"w-full mb-2 p-2 {'border-2 border-primary' if is_source else ''}"
                    ):
                        with ui.row().classes("items-center"):
                            ui.badge(f"#{i + 1}").classes("mr-2")
                            if is_source:
                                ui.badge("SOURCE").props("color=positive")

                            # Show reranker score for Stochastic RAG
                            if result.reranker_scores and i < len(result.reranker_scores):
                                ui.label(f"Rerank: {result.reranker_scores[i]:.3f}").classes(
                                    "text-xs text-purple-500 dark:text-purple-400 ml-2"
                                )

                            ui.label(f"Retrieval: {passage.get('score', 0):.3f}").classes(
                                "text-xs text-gray-500 dark:text-gray-400 ml-auto"
                            )

                        title = passage.get("title", "Untitled")
                        ui.label(title).classes("font-bold text-sm")

                        text = passage.get("text", "")[:300]
                        if len(passage.get("text", "")) > 300:
                            text += "..."
                        ui.label(text).classes("text-xs text-gray-600 dark:text-gray-400")

        except Exception as e:
            self.answer_label.text = f"Error: {e}"
            ui.notify(f"Error during inference: {e}", type="negative")

        finally:
            self.generate_button.enable()


class InferencePanelCompact:
    """
    Compact version of inference panel for embedding in main dashboard.
    """

    def __init__(self):
        """Initialize the compact inference panel."""
        self.engine = None
        self.is_loaded = False
        self._build_ui()

    def _build_ui(self):
        """Build the compact UI."""
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Quick Inference").classes("text-h6")
                self.status_badge = ui.badge("Not loaded").props("color=grey")

            with ui.row().classes("w-full items-end gap-2 mt-2"):
                self.question_input = ui.input(
                    placeholder="Ask a question...",
                ).classes("flex-grow")

                self.ask_button = ui.button(
                    icon="send",
                    on_click=self._on_ask,
                ).props("color=primary").disable()

            self.answer_label = ui.label("").classes(
                "text-sm mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded w-full"
            )
            self.answer_label.set_visibility(False)

    def set_engine(self, engine):
        """Set the inference engine."""
        self.engine = engine
        if engine and engine.is_loaded:
            self.is_loaded = True
            self.status_badge.text = f"Ready ({engine.algorithm})"
            self.status_badge.props(remove="color=grey")
            self.status_badge.props(add="color=positive")
            self.ask_button.enable()

    async def _on_ask(self):
        """Handle ask button click."""
        if not self.is_loaded or self.engine is None:
            return

        question = self.question_input.value.strip()
        if not question:
            return

        self.ask_button.disable()
        self.answer_label.set_visibility(True)
        self.answer_label.text = "Thinking..."

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.engine.answer_question, question
            )
            self.answer_label.text = result.answer if result.answer else "(No answer)"
        except Exception as e:
            self.answer_label.text = f"Error: {e}"
        finally:
            self.ask_button.enable()
