"""
Trident Pipeline Manager - A module for managing WSI processing workflows.

This module provides a PipelineManager class that tracks the state of WSI processing
tasks (segmentation, coordinate extraction, feature extraction) and only performs
missing operations for each sample.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import h5py
import pandas as pd
from trident import Processor
from trident.segmentation_models.load import segmentation_model_factory
from trident.patch_encoder_models.load import encoder_factory


@dataclass
class SampleState:
    """Track processing state for a single WSI sample."""
    sample_id: str
    wsi_path: str
    segmentation_done: bool = False
    coordinates_done: bool = False
    features_done: bool = False
    segmentation_path: Optional[str] = None
    coordinates_path: Optional[str] = None
    features_path: Optional[str] = None
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    # Segmentation parameters
    segmenter: str = 'hest'
    seg_conf_thresh: float = 0.5
    remove_holes: bool = False
    remove_artifacts: bool = False
    remove_penmarks: bool = False
    
    # Patching parameters
    mag: int = 20
    patch_size: int = 512
    overlap: int = 0
    min_tissue_proportion: float = 0.0
    
    # Feature extraction parameters
    patch_encoder: str = 'conch_v15'
    patch_encoder_ckpt_path: Optional[str] = None
    
    # Processing parameters
    batch_size: int = 64
    gpu: int = 0
    max_workers: Optional[int] = None
    skip_errors: bool = True


class PipelineManager:
    """
    Manages WSI processing pipeline state and execution.
    
    This class tracks which processing steps have been completed for each sample
    and only performs missing operations, avoiding redundant computation.
    """
    
    def __init__(self, 
                 job_dir: str,
                 wsi_dir: str,
                 config: Optional[PipelineConfig] = None,
                 wsi_ext: Optional[List[str]] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            job_dir: Directory to store all outputs
            wsi_dir: Directory containing WSI files
            config: Pipeline configuration (uses defaults if None)
            wsi_ext: List of allowed WSI file extensions
        """
        self.job_dir = Path(job_dir)
        self.wsi_dir = Path(wsi_dir)
        self.config = config or PipelineConfig()
        self.wsi_ext = wsi_ext
        
        # Create output directory structure
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.job_dir / "pipeline_state.json"
        
        # Initialize subdirectories
        self.seg_dir = self.job_dir / "segmentation"
        self.coords_dir = self.job_dir / f"coordinates_{self.config.mag}x_{self.config.patch_size}px"
        self.features_dir = self.job_dir / f"features_{self.config.mag}x_{self.config.patch_size}px_{self.config.patch_encoder}"
        
        for directory in [self.seg_dir, self.coords_dir, self.features_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = Processor(
            job_dir=str(self.job_dir),
            wsi_source=str(self.wsi_dir),
            wsi_ext=self.wsi_ext,
            skip_errors=self.config.skip_errors,
            max_workers=self.config.max_workers,
        )
        
        # Load or initialize state
        self.samples: Dict[str, SampleState] = self._load_state()
    
    def _generate_sample_id(self, wsi_path: str) -> str:
        """Generate unique sample ID from WSI path."""
        return Path(wsi_path).stem
    
    def _load_state(self) -> Dict[str, SampleState]:
        """Load pipeline state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return {k: SampleState(**v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
        return {}
    
    def _save_state(self):
        """Save pipeline state to JSON file."""
        data = {k: asdict(v) for k, v in self.samples.items()}
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_sample_state(self, sample_id: str, **kwargs):
        """Update sample state and save to disk."""
        if sample_id not in self.samples:
            # Initialize new sample
            wsi_files = list(self.wsi_dir.glob("*"))
            if self.wsi_ext:
                wsi_files = [f for f in wsi_files if f.suffix.lower() in [ext.lower() for ext in self.wsi_ext]]
            
            matching_files = [f for f in wsi_files if f.stem == sample_id]
            if not matching_files:
                raise ValueError(f"No WSI file found for sample_id: {sample_id}")
            
            self.samples[sample_id] = SampleState(
                sample_id=sample_id,
                wsi_path=str(matching_files[0])
            )
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(self.samples[sample_id], key):
                setattr(self.samples[sample_id], key, value)
        
        self.samples[sample_id].last_updated = datetime.now().isoformat()
        self._save_state()
    
    def discover_samples(self) -> List[str]:
        """Discover all WSI samples in the input directory."""
        wsi_files = list(self.wsi_dir.glob("*"))
        if self.wsi_ext:
            wsi_files = [f for f in wsi_files if f.suffix.lower() in [ext.lower() for ext in self.wsi_ext]]
        
        if not wsi_files:
            raise FileNotFoundError(f"No WSI files with extensions {self.wsi_ext} found in {self.wsi_dir}")
        
        sample_ids = []
        for wsi_file in wsi_files:
            if wsi_file.is_file():
                sample_id = self._generate_sample_id(str(wsi_file))
                sample_ids.append(sample_id)
                
                # Initialize sample state if not exists
                if sample_id not in self.samples:
                    self.samples[sample_id] = SampleState(
                        sample_id=sample_id,
                        wsi_path=str(wsi_file)
                    )
        
        # Scan for existing output files and update states
        self._scan_existing_outputs()
        
        self._save_state()
        return sample_ids
    
    def _scan_existing_outputs(self):
        """Scan for existing output files and update sample states."""
        for sample_id in self.samples.keys():
            # Check for segmentation files - including .h5ad files
            if not self.samples[sample_id].segmentation_done:
                seg_patterns = [
                    f"{sample_id}_segmentation.h5",
                    f"{sample_id}_segmentation.h5ad",
                    f"{sample_id}.h5",
                    f"{sample_id}.h5ad",
                    f"*{sample_id}*segmentation*.h5",
                    f"*{sample_id}*segmentation*.h5ad",
                    f"*{sample_id}*seg*.h5",
                    f"*{sample_id}*seg*.h5ad"
                ]
                
                for pattern in seg_patterns:
                    seg_files = list(self.job_dir.glob(f"**/{pattern}"))
                    if seg_files:
                        self._update_sample_state(
                            sample_id,
                            segmentation_done=True,
                            segmentation_path=str(seg_files[0])
                        )
                        break
            
            # Check for coordinate files - including .h5ad files
            if not self.samples[sample_id].coordinates_done:
                coord_patterns = [
                    f"{sample_id}_coordinates.h5",
                    f"{sample_id}_coordinates.h5ad",
                    f"{sample_id}_coords.h5",
                    f"{sample_id}_coords.h5ad",
                    f"{sample_id}.h5",
                    f"{sample_id}.h5ad",
                    f"*{sample_id}*coord*.h5",
                    f"*{sample_id}*coord*.h5ad",
                    f"*{sample_id}*patch*.h5",
                    f"*{sample_id}*patch*.h5ad"
                ]
                
                for pattern in coord_patterns:
                    coord_files = list(self.job_dir.glob(f"**/{pattern}"))
                    if coord_files:
                        self._update_sample_state(
                            sample_id,
                            coordinates_done=True,
                            coordinates_path=str(coord_files[0])
                        )
                        break
            
            # Check for feature files - including .h5ad files
            if not self.samples[sample_id].features_done:
                feat_patterns = [
                    f"{sample_id}_features.h5",
                    f"{sample_id}_features.h5ad",
                    f"{sample_id}_feat.h5",
                    f"{sample_id}_feat.h5ad",
                    f"{sample_id}.h5",
                    f"{sample_id}.h5ad",
                    f"*{sample_id}*feature*.h5",
                    f"*{sample_id}*feature*.h5ad",
                    f"*{sample_id}*feat*.h5",
                    f"*{sample_id}*feat*.h5ad"
                ]
                
                for pattern in feat_patterns:
                    feat_files = list(self.job_dir.glob(f"**/{pattern}"))
                    if feat_files:
                        self._update_sample_state(
                            sample_id,
                            features_done=True,
                            features_path=str(feat_files[0])
                        )
                        break
    
    def get_pending_samples(self, task: str) -> List[str]:
        """Get samples that need processing for a specific task."""
        pending = []
        for sample_id, state in self.samples.items():
            if task == "segmentation" and not state.segmentation_done:
                pending.append(sample_id)
            elif task == "coordinates" and state.segmentation_done and not state.coordinates_done:
                pending.append(sample_id)
            elif task == "features" and state.coordinates_done and not state.features_done:
                pending.append(sample_id)
        return pending
    
    def run_segmentation(self, sample_ids: Optional[List[str]] = None) -> int:
        """Run segmentation for specified samples."""
        if sample_ids is None:
            sample_ids = self.get_pending_samples("segmentation")
        
        if not sample_ids:
            print("No samples need segmentation.")
            return 0
        
        print(f"Running segmentation for {len(sample_ids)} samples...")
        
        # Initialize segmentation model
        segmentation_model = segmentation_model_factory(
            self.config.segmenter,
            confidence_thresh=self.config.seg_conf_thresh,
        )
        
        artifact_remover_model = None
        if self.config.remove_artifacts or self.config.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=self.config.remove_penmarks and not self.config.remove_artifacts
            )
        
        # Run segmentation
        self.processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not self.config.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=self.config.batch_size,
            device=f'cuda:{self.config.gpu}',
        )
        
        # Update state for processed samples - check for various possible output formats
        processed_count = 0
        for sample_id in sample_ids:
            # Check for common segmentation output patterns
            possible_paths = [
                self.seg_dir / f"{sample_id}_segmentation.h5",
                self.seg_dir / f"{sample_id}.h5",
                self.job_dir / "segmentation" / f"{sample_id}_segmentation.h5",
                self.job_dir / "segmentation" / f"{sample_id}.h5",
            ]
            
            # Also check in job_dir subdirectories
            for seg_file in self.job_dir.glob(f"**/segmentation/**/*{sample_id}*"):
                if seg_file.is_file():
                    possible_paths.append(seg_file)
            
            found_path = None
            for path in possible_paths:
                if path.exists():
                    found_path = str(path)
                    break
            
            if found_path:
                self._update_sample_state(
                    sample_id,
                    segmentation_done=True,
                    segmentation_path=found_path
                )
                processed_count += 1
        
        return processed_count
    
    def run_coordinate_extraction(self, sample_ids: Optional[List[str]] = None) -> int:
        """Run coordinate extraction for specified samples."""
        if sample_ids is None:
            sample_ids = self.get_pending_samples("coordinates")
        
        if not sample_ids:
            print("No samples need coordinate extraction.")
            return 0
        
        for sample_id in sample_ids:
            state = self.samples[sample_id]
            if not state.segmentation_done or not state.segmentation_path or not Path(state.segmentation_path).exists():
                raise FileNotFoundError(f"Segmentation output for {sample_id} not found. Cannot extract coordinates.")

        print(f"Running coordinate extraction for {len(sample_ids)} samples...")
        
        # Run patching job
        self.processor.run_patching_job(
            target_magnification=self.config.mag,
            patch_size=self.config.patch_size,
            overlap=self.config.overlap,
            saveto=str(self.coords_dir),
            min_tissue_proportion=self.config.min_tissue_proportion,
        )
        
        # Update state for processed samples - check for various possible output formats
        processed_count = 0
        for sample_id in sample_ids:
            # Check for common coordinate output patterns
            possible_paths = [
                self.coords_dir / f"{sample_id}_coordinates.h5",
                self.coords_dir / f"{sample_id}.h5",
                self.coords_dir / f"{sample_id}_coords.h5",
                self.coords_dir / f"{sample_id}.h5ad",
            ]
            
            # Also check in job_dir subdirectories
            for coord_file in self.job_dir.glob(f"**/coordinates*/**/*{sample_id}*"):
                if coord_file.is_file():
                    possible_paths.append(coord_file)
            
            found_path = None
            for path in possible_paths:
                if path.exists():
                    found_path = str(path)
                    break
            
            if found_path:
                self._update_sample_state(
                    sample_id,
                    coordinates_done=True,
                    coordinates_path=found_path
                )
                processed_count += 1
        
        return processed_count
    
    def run_feature_extraction(self, sample_ids: Optional[List[str]] = None) -> int:
        """Run feature extraction for specified samples."""
        if sample_ids is None:
            sample_ids = self.get_pending_samples("features")
        
        if not sample_ids:
            print("No samples need feature extraction.")
            return 0
        
        for sample_id in sample_ids:
            state = self.samples[sample_id]
            if not state.coordinates_done or not state.coordinates_path or not Path(state.coordinates_path).exists():
                raise FileNotFoundError(f"Coordinates for {sample_id} not found. Cannot extract features.")

        print(f"Running feature extraction for {len(sample_ids)} samples...")
        
        # Initialize patch encoder
        encoder = encoder_factory(
            self.config.patch_encoder, 
            weights_path=self.config.patch_encoder_ckpt_path
        )
        
        # Run feature extraction
        self.processor.run_patch_feature_extraction_job(
            coords_dir=str(self.coords_dir),
            patch_encoder=encoder,
            device=f'cuda:{self.config.gpu}',
            saveas='h5',
            batch_limit=self.config.batch_size,
        )
        
        # Update state for processed samples - check for various possible output formats
        processed_count = 0
        for sample_id in sample_ids:
            # Check for common feature output patterns
            possible_paths = [
                self.features_dir / f"{sample_id}_features.h5",
                self.features_dir / f"{sample_id}.h5",
                self.features_dir / f"{sample_id}_feat.h5",
                self.features_dir / f"{sample_id}.h5ad",
            ]
            
            # Also check in job_dir subdirectories for feature files
            for feat_file in self.job_dir.glob(f"**/features*/**/*{sample_id}*"):
                if feat_file.is_file():
                    possible_paths.append(feat_file)
            
            # Check for files in the default Trident output structure
            for feat_file in self.job_dir.glob(f"**/*{sample_id}*_features.h5"):
                possible_paths.append(feat_file)
            
            found_path = None
            for path in possible_paths:
                if path.exists():
                    found_path = str(path)
                    break
            
            if found_path:
                self._update_sample_state(
                    sample_id,
                    features_done=True,
                    features_path=found_path
                )
                processed_count += 1
        
        return processed_count
    
    def run_full_pipeline(self, sample_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """Run the complete pipeline for specified samples."""
        if sample_ids is None:
            sample_ids = self.discover_samples()
        
        results = {}
        
        # Run segmentation
        seg_pending = [s for s in sample_ids if s in self.get_pending_samples("segmentation")]
        results['segmentation'] = self.run_segmentation(seg_pending)
        
        # Run coordinate extraction
        coords_pending = [s for s in sample_ids if s in self.get_pending_samples("coordinates")]
        results['coordinates'] = self.run_coordinate_extraction(coords_pending)
        
        # Run feature extraction
        feat_pending = [s for s in sample_ids if s in self.get_pending_samples("features")]
        results['features'] = self.run_feature_extraction(feat_pending)
        
        return results
    
    def get_summary(self) -> Dict[str, int]:
        """Get processing summary statistics."""
        total = len(self.samples)
        seg_done = sum(1 for s in self.samples.values() if s.segmentation_done)
        coords_done = sum(1 for s in self.samples.values() if s.coordinates_done)
        feat_done = sum(1 for s in self.samples.values() if s.features_done)
        
        return {
            'total_samples': total,
            'segmentation_completed': seg_done,
            'coordinates_completed': coords_done,
            'features_completed': feat_done,
            'segmentation_pending': total - seg_done,
            'coordinates_pending': total - coords_done,
            'features_pending': total - feat_done,
        }
    
    def print_summary(self):
        """Print processing summary."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Segmentation: {summary['segmentation_completed']}/{summary['total_samples']} completed")
        print(f"Coordinates: {summary['coordinates_completed']}/{summary['total_samples']} completed")
        print(f"Features: {summary['features_completed']}/{summary['total_samples']} completed")
        print("="*50)
    
    def debug_output_files(self):
        """Debug method to show what files actually exist in the output directories."""
        print("\n" + "="*60)
        print("DEBUG: OUTPUT FILES")
        print("="*60)
        
        print(f"\nJob directory: {self.job_dir}")
        print(f"Segmentation directory: {self.seg_dir}")
        print(f"Coordinates directory: {self.coords_dir}")
        print(f"Features directory: {self.features_dir}")
        
        # Show all files in job directory
        print(f"\nAll files in job directory:")
        all_files = list(self.job_dir.glob("**/*"))
        for f in sorted(all_files):
            if f.is_file():
                print(f"  {f.relative_to(self.job_dir)}")
        
        # Show sample-specific files
        print(f"\nFirst 3 samples file analysis:")
        sample_ids = list(self.samples.keys())[:3]
        for sample_id in sample_ids:
            print(f"\n  Sample: {sample_id}")
            
            # Search for files containing this sample ID
            matching_files = list(self.job_dir.glob(f"**/*{sample_id}*"))
            if matching_files:
                print(f"    Found {len(matching_files)} files:")
                for f in sorted(matching_files):
                    if f.is_file():
                        print(f"      {f.relative_to(self.job_dir)}")
            else:
                print(f"    No files found for this sample")
        
        print("="*60)
    
    def force_refresh_state(self):
        """Force refresh the state by scanning all output directories."""
        print("Force refreshing state by scanning all output files...")
        
        # Reset all states
        for sample_id in self.samples.keys():
            self.samples[sample_id].segmentation_done = False
            self.samples[sample_id].coordinates_done = False
            self.samples[sample_id].features_done = False
            self.samples[sample_id].segmentation_path = None
            self.samples[sample_id].coordinates_path = None
            self.samples[sample_id].features_path = None
        
        # Scan for all possible output patterns
        self._scan_existing_outputs()
        
        # Also check common Trident output patterns
        print("Checking common Trident output patterns...")
        
        # Check for various output directory structures that Trident might create
        possible_output_dirs = [
            self.job_dir,
            self.job_dir / "segmentation",
            self.job_dir / "coordinates",
            self.job_dir / "features",
            self.job_dir / "patches",
            self.job_dir / "tissue_coordinates",
            self.job_dir / "patch_features",
        ]
        
        for sample_id in self.samples.keys():
            # More comprehensive file search
            for output_dir in possible_output_dirs:
                if output_dir.exists():
                    # Look for any file containing the sample ID
                    for file_path in output_dir.glob(f"*{sample_id}*"):
                        if file_path.is_file():
                            filename = file_path.name.lower()
                            
                            # Determine file type based on name and content
                            if any(keyword in filename for keyword in ['seg', 'mask', 'contour']):
                                if not self.samples[sample_id].segmentation_done:
                                    self._update_sample_state(
                                        sample_id,
                                        segmentation_done=True,
                                        segmentation_path=str(file_path)
                                    )
                            elif any(keyword in filename for keyword in ['coord', 'patch', 'tile']):
                                if not self.samples[sample_id].coordinates_done:
                                    self._update_sample_state(
                                        sample_id,
                                        coordinates_done=True,
                                        coordinates_path=str(file_path)
                                    )
                            elif any(keyword in filename for keyword in ['feat', 'embedding', 'feature']):
                                if not self.samples[sample_id].features_done:
                                    self._update_sample_state(
                                        sample_id,
                                        features_done=True,
                                        features_path=str(file_path)
                                    )
        
        self._save_state()
        print("State refresh completed.")
        self.print_summary()
    
    def convert_h5ad_to_h5(self, h5ad_path: str, h5_path: str) -> bool:
        """Convert .h5ad file to .h5 format."""
        try:
            # Try to import anndata
            try:
                import anndata as ad
                use_anndata = True
            except ImportError:
                use_anndata = False
                print(f"Warning: anndata not available, attempting simple conversion for {h5ad_path}")
            
            if use_anndata:
                # Read the .h5ad file
                adata = ad.read_h5ad(h5ad_path)
                
                # Save as .h5 format
                with h5py.File(h5_path, 'w') as h5_file:
                    # Save the main data matrix
                    if hasattr(adata, 'X') and adata.X is not None:
                        h5_file.create_dataset('X', data=adata.X)
                    
                    # Save observations (cell/patch metadata)
                    if hasattr(adata, 'obs') and len(adata.obs) > 0:
                        obs_group = h5_file.create_group('obs')
                        for col in adata.obs.columns:
                            obs_group.create_dataset(col, data=adata.obs[col].values)
                    
                    # Save variables (feature metadata)
                    if hasattr(adata, 'var') and len(adata.var) > 0:
                        var_group = h5_file.create_group('var')
                        for col in adata.var.columns:
                            var_group.create_dataset(col, data=adata.var[col].values)
                    
                    # Save unstructured metadata
                    if hasattr(adata, 'uns') and len(adata.uns) > 0:
                        uns_group = h5_file.create_group('uns')
                        for key, value in adata.uns.items():
                            try:
                                uns_group.create_dataset(key, data=value)
                            except:
                                # Skip if data can't be serialized
                                pass
            else:
                # Simple approach: just copy the file with .h5 extension
                # .h5ad files are actually HDF5 files, so they can often be read as .h5
                import shutil
                shutil.copy2(h5ad_path, h5_path)
                print(f"Copied {h5ad_path} to {h5_path} (simple copy)")
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not convert {h5ad_path} to .h5 format: {e}")
            return False
    
    def ensure_h5_format(self):
        """Convert any .h5ad files to .h5 format by renaming (they're compatible)."""
        converted_count = 0
        
        for sample_id, sample_state in self.samples.items():
            # Check segmentation path
            if sample_state.segmentation_path and sample_state.segmentation_path.endswith('.h5ad'):
                h5_path = sample_state.segmentation_path.replace('.h5ad', '.h5')
                try:
                    import shutil
                    shutil.copy2(sample_state.segmentation_path, h5_path)
                    self._update_sample_state(sample_id, segmentation_path=h5_path)
                    converted_count += 1
                    print(f"Converted segmentation for {sample_id}: .h5ad -> .h5")
                except Exception as e:
                    print(f"Warning: Could not convert segmentation for {sample_id}: {e}")
            
            # Check coordinates path
            if sample_state.coordinates_path and sample_state.coordinates_path.endswith('.h5ad'):
                h5_path = sample_state.coordinates_path.replace('.h5ad', '.h5')
                try:
                    import shutil
                    shutil.copy2(sample_state.coordinates_path, h5_path)
                    self._update_sample_state(sample_id, coordinates_path=h5_path)
                    converted_count += 1
                    print(f"Converted coordinates for {sample_id}: .h5ad -> .h5")
                except Exception as e:
                    print(f"Warning: Could not convert coordinates for {sample_id}: {e}")
            
            # Check features path
            if sample_state.features_path and sample_state.features_path.endswith('.h5ad'):
                h5_path = sample_state.features_path.replace('.h5ad', '.h5')
                try:
                    import shutil
                    shutil.copy2(sample_state.features_path, h5_path)
                    self._update_sample_state(sample_id, features_path=h5_path)
                    converted_count += 1
                    print(f"Converted features for {sample_id}: .h5ad -> .h5")
                except Exception as e:
                    print(f"Warning: Could not convert features for {sample_id}: {e}")
        
        if converted_count > 0:
            print(f"Successfully converted {converted_count} files from .h5ad to .h5 format")
        else:
            print("No .h5ad files found to convert")
        
        return converted_count
