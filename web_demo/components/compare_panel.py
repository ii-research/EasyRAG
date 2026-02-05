"""
Compare Panel Component
========================

Three-way model comparison for FiD-Light/FiD Pure/Stochastic RAG.

Features:
- Shared retriever and index for fair comparison
- Up to 3 models compared side-by-side
- Adjustable parameters per model
- Parallel result display with latency metrics
"""

from nicegui import ui
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import asyncio


# Algorithm options
ALGORITHM_OPTIONS = {
    "fidlight": "FiD-Light",
    "fid_pure": "FiD Pure",
    "stochastic_rag": "Stochastic RAG",
}

# Model backbone options
MODEL_OPTIONS = {
    "t5base": "T5-base (220M)",
    "t5gemma": "T5Gemma2 (270M)",
}

# Default parameters per algorithm
ALGORITHM_DEFAULTS = {
    "fidlight": {"k": 64, "n_passages": 40, "n_selected": 40},
    "fid_pure": {"k": 250, "n_passages": 100, "n_selected": 100},
    "stochastic_rag": {"k": 64, "n_passages": 40, "n_selected": 10},
}


@dataclass
class ModelConfig:
    """Configuration for one model in comparison."""
    algorithm: str = "fidlight"
    model_type: str = "t5base"
    checkpoint_path: str = ""
    compression_k: int = 64
    num_passages: int = 40
    n_selected: int = 10
    enabled: bool = True


@dataclass
class CompareResult:
    """Result from one model."""
    answer: str = ""
    raw_output: str = ""
    source_indices: List[int] = field(default_factory=list)
    latency_ms: float = 0
    error: str = ""
    reranker_scores: List[float] = field(default_factory=list)


class ComparePanel:
    """
    Three-way model comparison panel.

    Allows comparing up to 3 models with the same question,
    using shared retriever and index for fair comparison.
    """

    def __init__(
        self,
        default_retriever: str = "",
        default_index: str = "kilt_data/gtr_faiss_index",
    ):
        """
        Initialize the compare panel.

        Args:
            default_retriever: Default retriever path (empty for default GTR)
            default_index: Default index path
        """
        self.shared_retriever_path = default_retriever
        self.shared_index_path = default_index
        self.shared_wiki_arrow_path = ""

        # Three model configurations with different defaults
        self.model_configs = [
            ModelConfig(
                algorithm="fidlight",
                compression_k=64,
                num_passages=40,
                checkpoint_path="checkpoints/fidlight_paper/final",
                enabled=True,
            ),
            ModelConfig(
                algorithm="fid_pure",
                compression_k=250,
                num_passages=100,
                checkpoint_path="checkpoints/fid_pure/final",
                enabled=True,
            ),
            ModelConfig(
                algorithm="stochastic_rag",
                compression_k=64,
                num_passages=40,
                n_selected=10,
                checkpoint_path="checkpoints/stochastic_rag/final",
                enabled=False,
            ),
        ]

        # Loaded engines (lazy)
        self.engines = [None, None, None]
        self.is_loading = [False, False, False]

        # Shared retriever (loaded once, shared by all models)
        self._shared_retriever = None

        # Results
        self.results = [CompareResult(), CompareResult(), CompareResult()]
        self.passages = []  # Shared passages from retrieval

        self._build_ui()

    def _build_ui(self):
        """Build the compare panel UI."""
        with ui.column().classes("w-full"):
            # Title
            ui.label("Model Comparison").classes("text-h5 mb-4")

            # Shared retriever/index configuration
            with ui.card().classes("w-full mb-4"):
                ui.label("Shared Retriever & Index").classes("text-h6 mb-2")
                ui.label(
                    "All models will use the same retriever and index for fair comparison."
                ).classes("text-xs text-gray-500 dark:text-gray-400 mb-2")

                with ui.row().classes("w-full items-center gap-4"):
                    ui.label("Retriever:").classes("w-24")
                    self.retriever_input = ui.input(
                        value=self.shared_retriever_path,
                        placeholder="(optional) checkpoints/gtr_finetuned",
                    ).classes("flex-grow")

                with ui.row().classes("w-full items-center gap-4 mt-2"):
                    ui.label("Index:").classes("w-24")
                    self.index_input = ui.input(
                        value=self.shared_index_path,
                        placeholder="kilt_data/gtr_faiss_index",
                    ).classes("flex-grow")

                with ui.row().classes("w-full items-center gap-4 mt-2"):
                    ui.label("Wiki Arrow:").classes("w-24")
                    self.wiki_input = ui.input(
                        value="",
                        placeholder="(optional) kilt_data/wiki_arrow",
                    ).classes("flex-grow")

            # Three model configurations in a row
            with ui.row().classes("w-full gap-4"):
                for i in range(3):
                    self._build_model_config_card(i)

            # Query input
            with ui.card().classes("w-full my-4"):
                ui.label("Query").classes("text-h6 mb-2")

                self.query_input = ui.textarea(
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
                            on_click=lambda q=q: self._set_query(q),
                        ).props("flat dense size=sm")

                with ui.row().classes("w-full justify-end mt-2"):
                    self.compare_button = ui.button(
                        "Compare Models",
                        icon="compare_arrows",
                        on_click=self._on_compare,
                    ).props("color=primary")

            # Results section
            with ui.card().classes("w-full"):
                ui.label("Results").classes("text-h6 mb-2")

                with ui.row().classes("w-full gap-4"):
                    for i in range(3):
                        self._build_result_card(i)

    def _build_model_config_card(self, index: int):
        """Build configuration card for one model."""
        config = self.model_configs[index]
        labels = ["Model A", "Model B", "Model C"]
        colors = ["blue", "green", "purple"]

        with ui.card().classes("flex-1 min-w-[250px]"):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(labels[index]).classes(f"text-h6 text-{colors[index]}-600")

                # Enable checkbox
                checkbox = ui.checkbox(
                    "Enable",
                    value=config.enabled,
                    on_change=lambda e, i=index: self._on_enable_change(i, e.value),
                )
                setattr(self, f"enable_checkbox_{index}", checkbox)

            # Algorithm selection
            algo_select = ui.select(
                options=ALGORITHM_OPTIONS,
                value=config.algorithm,
                label="Algorithm",
                on_change=lambda e, i=index: self._on_algo_change(i, e.value),
            ).classes("w-full")
            setattr(self, f"algo_select_{index}", algo_select)

            # Model type selection
            model_select = ui.select(
                options=MODEL_OPTIONS,
                value=config.model_type,
                label="Model Backbone",
                on_change=lambda e, i=index: self._on_model_type_change(i, e.value),
            ).classes("w-full mt-2")
            setattr(self, f"model_select_{index}", model_select)

            # Checkpoint path
            checkpoint_input = ui.input(
                value=config.checkpoint_path,
                label="Checkpoint",
                placeholder="checkpoints/...",
                on_change=lambda e, i=index: self._on_checkpoint_change(i, e.value),
            ).classes("w-full mt-2")
            setattr(self, f"checkpoint_input_{index}", checkpoint_input)

            # Parameters row
            with ui.row().classes("w-full gap-2 mt-2"):
                k_input = ui.number(
                    value=config.compression_k,
                    label="k",
                    min=1,
                    max=512,
                    step=1,
                    on_change=lambda e, i=index: self._on_k_change(i, e.value),
                ).classes("w-16")
                setattr(self, f"k_input_{index}", k_input)

                n_input = ui.number(
                    value=config.num_passages,
                    label="n_pass",
                    min=1,
                    max=200,
                    step=1,
                    on_change=lambda e, i=index: self._on_n_passages_change(i, e.value),
                ).classes("w-20")
                setattr(self, f"n_input_{index}", n_input)

                # n_selected (for SR only)
                n_sel_input = ui.number(
                    value=config.n_selected,
                    label="n_sel",
                    min=1,
                    max=100,
                    step=1,
                    on_change=lambda e, i=index: self._on_n_selected_change(i, e.value),
                ).classes("w-16")
                n_sel_input.set_visibility(config.algorithm == "stochastic_rag")
                setattr(self, f"n_sel_input_{index}", n_sel_input)

            # Status indicator
            status_label = ui.label("Not loaded").classes(
                "text-xs text-gray-500 dark:text-gray-400 mt-2"
            )
            setattr(self, f"status_label_{index}", status_label)

    def _build_result_card(self, index: int):
        """Build result display card for one model."""
        labels = ["Model A", "Model B", "Model C"]
        colors = ["blue", "green", "purple"]

        result_container = ui.column().classes("flex-1 min-w-[250px]")
        with result_container:
            with ui.card().classes("w-full h-full").style("min-height: 200px"):
                ui.label(labels[index]).classes(f"font-bold text-{colors[index]}-600")

                # Answer
                answer_label = ui.label("...").classes(
                    "text-lg p-2 bg-gray-100 dark:bg-gray-800 rounded w-full mt-2"
                )
                setattr(self, f"answer_label_{index}", answer_label)

                # Latency
                latency_label = ui.label("").classes(
                    "text-xs text-gray-500 dark:text-gray-400 mt-1"
                )
                setattr(self, f"latency_label_{index}", latency_label)

                # Raw output
                raw_label = ui.label("").classes(
                    "text-xs text-gray-400 dark:text-gray-500 italic mt-1"
                )
                setattr(self, f"raw_label_{index}", raw_label)

                # Sources
                source_container = ui.row().classes("w-full flex-wrap gap-1 mt-2")
                setattr(self, f"source_container_{index}", source_container)

        setattr(self, f"result_container_{index}", result_container)

    def _set_query(self, query: str):
        """Set the query input."""
        self.query_input.value = query

    def _on_enable_change(self, index: int, value: bool):
        """Handle enable checkbox change."""
        self.model_configs[index].enabled = value
        # Reset engine when disabled/enabled
        self.engines[index] = None

    def _on_algo_change(self, index: int, value: str):
        """Handle algorithm change."""
        self.model_configs[index].algorithm = value

        # Update defaults
        defaults = ALGORITHM_DEFAULTS.get(value, ALGORITHM_DEFAULTS["fidlight"])
        self.model_configs[index].compression_k = defaults["k"]
        self.model_configs[index].num_passages = defaults["n_passages"]
        self.model_configs[index].n_selected = defaults["n_selected"]

        # Update UI
        k_input = getattr(self, f"k_input_{index}")
        k_input.value = defaults["k"]

        n_input = getattr(self, f"n_input_{index}")
        n_input.value = defaults["n_passages"]

        n_sel_input = getattr(self, f"n_sel_input_{index}")
        n_sel_input.value = defaults["n_selected"]
        n_sel_input.set_visibility(value == "stochastic_rag")

        # Reset engine
        self.engines[index] = None
        status_label = getattr(self, f"status_label_{index}")
        status_label.text = "Not loaded"

    def _on_model_type_change(self, index: int, value: str):
        """Handle model type change."""
        self.model_configs[index].model_type = value
        self.engines[index] = None

    def _on_checkpoint_change(self, index: int, value: str):
        """Handle checkpoint path change."""
        self.model_configs[index].checkpoint_path = value
        self.engines[index] = None

    def _on_k_change(self, index: int, value):
        """Handle compression k change."""
        if value:
            self.model_configs[index].compression_k = int(value)
            self.engines[index] = None

    def _on_n_passages_change(self, index: int, value):
        """Handle num passages change."""
        if value:
            self.model_configs[index].num_passages = int(value)
            self.engines[index] = None

    def _on_n_selected_change(self, index: int, value):
        """Handle n_selected change."""
        if value:
            self.model_configs[index].n_selected = int(value)
            self.engines[index] = None

    async def _on_compare(self):
        """Handle compare button click."""
        query = self.query_input.value.strip()
        if not query:
            ui.notify("Please enter a question", type="warning")
            return

        # Check if at least one model is enabled
        enabled_count = sum(1 for c in self.model_configs if c.enabled)
        if enabled_count == 0:
            ui.notify("Please enable at least one model", type="warning")
            return

        self.compare_button.disable()

        # Clear previous results
        for i in range(3):
            answer_label = getattr(self, f"answer_label_{i}")
            latency_label = getattr(self, f"latency_label_{i}")
            raw_label = getattr(self, f"raw_label_{i}")
            source_container = getattr(self, f"source_container_{i}")

            if self.model_configs[i].enabled:
                answer_label.text = "Loading..."
            else:
                answer_label.text = "(disabled)"

            latency_label.text = ""
            raw_label.text = ""
            source_container.clear()

        try:
            # Get shared config
            retriever_path = self.retriever_input.value.strip() or None
            index_path = self.index_input.value.strip()
            wiki_arrow_path = self.wiki_input.value.strip() or None

            # Load shared retriever once (for all models to share)
            if not hasattr(self, '_shared_retriever') or self._shared_retriever is None:
                if index_path:
                    print("Loading shared retriever for comparison...")
                    from utils.gtr_retriever import GTRRetriever
                    self._shared_retriever = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: GTRRetriever(
                            index_path=index_path,
                            model_path=retriever_path,
                            load_wiki=True,
                            wiki_arrow_path=wiki_arrow_path,
                        )
                    )
                    print("Shared retriever loaded!")
                else:
                    self._shared_retriever = None

            # Load and run each enabled model sequentially
            for i in range(3):
                if not self.model_configs[i].enabled:
                    continue

                config = self.model_configs[i]
                status_label = getattr(self, f"status_label_{i}")
                answer_label = getattr(self, f"answer_label_{i}")

                # Load engine if needed
                if self.engines[i] is None:
                    status_label.text = "Loading model..."
                    answer_label.text = "Loading model..."

                    try:
                        from ..inference_demo import InferenceEngine

                        self.engines[i] = InferenceEngine(
                            checkpoint_path=config.checkpoint_path,
                            algorithm=config.algorithm,
                            model_type=config.model_type,
                            retriever_path=retriever_path,
                            index_path=index_path if index_path else None,
                            wiki_arrow_path=wiki_arrow_path,
                            compression_k=config.compression_k,
                            num_passages=config.num_passages,
                            n_selected=config.n_selected,
                            shared_retriever=self._shared_retriever,  # Share retriever
                        )

                        await asyncio.get_event_loop().run_in_executor(
                            None, self.engines[i].load
                        )

                        if self.engines[i].is_loaded:
                            status_label.text = "Loaded"
                        else:
                            status_label.text = "Load failed"
                            answer_label.text = "Failed to load model"
                            continue

                    except Exception as e:
                        status_label.text = f"Error: {str(e)[:30]}..."
                        answer_label.text = f"Error: {e}"
                        continue

                # Run inference
                answer_label.text = "Generating..."

                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.engines[i].answer_question, query
                    )

                    # Update UI
                    answer_label.text = result.answer if result.answer else "(No answer)"

                    latency_label = getattr(self, f"latency_label_{i}")
                    latency_label.text = f"Latency: {result.latency_ms:.0f}ms"

                    raw_label = getattr(self, f"raw_label_{i}")
                    raw_text = result.raw_output[:40] + "..." if len(result.raw_output) > 40 else result.raw_output
                    raw_label.text = f"Raw: {raw_text}"

                    # Show sources
                    source_container = getattr(self, f"source_container_{i}")
                    source_container.clear()
                    with source_container:
                        if self.engines[i].has_source_pointer and result.source_indices:
                            for idx in result.source_indices:
                                ui.badge(f"[{idx}]").props("color=primary size=sm")
                        elif not self.engines[i].has_source_pointer:
                            ui.label("(no source pointer)").classes("text-xs text-gray-400")

                except Exception as e:
                    answer_label.text = f"Error: {e}"

        finally:
            self.compare_button.enable()


def create_compare_page():
    """Create the compare page with panel."""
    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        ComparePanel()
