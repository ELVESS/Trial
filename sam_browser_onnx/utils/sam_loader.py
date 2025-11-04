"""
SAM model loader with caching.
Loads and caches SAM backbones per model type.
"""
from pathlib import Path
from typing import Optional, Dict
import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMLoader:
    """Manages loading and caching of SAM models."""

    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self._cache: Dict[str, SamPredictor] = {}
        self._current_model_type: Optional[str] = None

    def load_model(self, model_type: str) -> SamPredictor:
        """
        Load SAM model and return predictor.
        Caches models to avoid reloading.

        Args:
            model_type: One of 'vit_b', 'vit_l', 'vit_h'

        Returns:
            SamPredictor instance
        """
        if model_type not in ["vit_b", "vit_l", "vit_h"]:
            raise ValueError(f"Invalid model_type: {model_type}")

        # Return cached model if available
        if model_type in self._cache:
            self._current_model_type = model_type
            return self._cache[model_type]

        # Load checkpoint
        checkpoint_path = self.checkpoints_dir / f"sam_{model_type}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Please run: python models/download_models.py"
            )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        sam.eval()

        # Create predictor
        predictor = SamPredictor(sam)

        # Cache
        self._cache[model_type] = predictor
        self._current_model_type = model_type

        return predictor

    def get_current_model(self) -> Optional[SamPredictor]:
        """Get the currently loaded model."""
        if self._current_model_type is None:
            return None
        return self._cache.get(self._current_model_type)

    def get_current_model_type(self) -> Optional[str]:
        """Get the current model type."""
        return self._current_model_type

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._cache.clear()
        self._current_model_type = None
