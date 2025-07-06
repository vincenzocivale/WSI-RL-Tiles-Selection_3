"""
Simplified inference pipeline combining HEST dataset management with TRIDENT-style model inference.
"""

from .file_utils import save_hdf5, read_assets_from_h5, save_pkl, load_pkl
from .dataset import PatchDataset, get_default_transforms
from .models import patch_encoder_factory, slide_encoder_factory, PatchEncoder, SlideEncoder
from .inference import run_inference, extract_patch_features, extract_slide_features, discover_samples

# Optional CONCH v1.5 support
try:
    from .conch_v15 import create_conch_v15_encoder, CONCHv15Model
    __all__ = [
        'save_hdf5', 'read_assets_from_h5', 'save_pkl', 'load_pkl',
        'PatchDataset', 'get_default_transforms',
        'patch_encoder_factory', 'slide_encoder_factory', 'PatchEncoder', 'SlideEncoder',
        'run_inference', 'extract_patch_features', 'extract_slide_features', 'discover_samples',
        'create_conch_v15_encoder', 'CONCHv15Model'
    ]
except ImportError:
    __all__ = [
        'save_hdf5', 'read_assets_from_h5', 'save_pkl', 'load_pkl',
        'PatchDataset', 'get_default_transforms',
        'patch_encoder_factory', 'slide_encoder_factory', 'PatchEncoder', 'SlideEncoder',
        'run_inference', 'extract_patch_features', 'extract_slide_features', 'discover_samples'
    ]
