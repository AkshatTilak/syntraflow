"""SyntraFlow Embedder Registry with Harrier 0.6B support (sub_07_02)."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EmbedderSpec:
    model_id: str
    display_name: str
    dimension: int
    modality: str
    provider: str
    vram_mb: int
    is_local: bool


EMBEDDER_REGISTRY: Dict[str, EmbedderSpec] = {
    "jina-clip-v2": EmbedderSpec(
        model_id="jinaai/jina-clip-v2",
        display_name="Jina CLIP v2 (Multimodal)",
        dimension=1024,
        modality="text+image",
        provider="huggingface",
        vram_mb=1000,
        is_local=True,
    ),
    "harrier-0.6b": EmbedderSpec(
        model_id="microsoft/harrier-oss-v1-0.6b",
        display_name="Harrier 0.6B (Local Text)",
        dimension=1024,
        modality="text",
        provider="huggingface",
        vram_mb=800,
        is_local=True,
    ),
    "harrier-270m": EmbedderSpec(
        model_id="microsoft/harrier-oss-v1-270m",
        display_name="Harrier 270M (Local Text)",
        dimension=640,
        modality="text",
        provider="huggingface",
        vram_mb=400,
        is_local=True,
    ),
    "nomic-embed-vision-v1.5": EmbedderSpec(
        model_id="nomic-ai/nomic-embed-vision-v1.5",
        display_name="Nomic Embed Vision v1.5",
        dimension=768,
        modality="image",
        provider="huggingface",
        vram_mb=500,
        is_local=True,
    ),
}


def get_embedder_spec(model_id: str) -> EmbedderSpec:
    """Retrieve embedder specification by model ID or raise ValueError."""
    spec = EMBEDDER_REGISTRY.get(model_id.lower())
    if not spec:
        raise ValueError(
            f"Unsupported embedding model '{model_id}'. Available models: {list(EMBEDDER_REGISTRY.keys())}"
        )
    return spec


def list_supported_embedders() -> List[Dict[str, Any]]:
    """List all supported embedder models with metadata."""
    return [
        {
            "model_id": k,
            "display_name": spec.display_name,
            "dimension": spec.dimension,
            "modality": spec.modality,
            "provider": spec.provider,
            "vram_mb": spec.vram_mb,
            "is_local": spec.is_local,
        }
        for k, spec in EMBEDDER_REGISTRY.items()
    ]
