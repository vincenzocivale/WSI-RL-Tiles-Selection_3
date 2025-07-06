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
    mpp: Optional[float] = None
    
    # Processing parameters
    batch_size: int = 64
    gpu: int = 0
    max_workers: Optional[int] = None
    skip_errors: bool = True
    reader_type: Optional[str] = None

class PipelineManager:
    """
    Manages WSI processing pipeline state and execution.
    
    This class tracks which processing steps have been completed for each sample
    and only performs missing operations, avoiding redundant computation.
    """
    
    def __init__(self, 
                 job_dir: str,
                 wsi_dir: str,
                 hest_dataset_root_dir: Optional[str] = None, # NEW: Root of the HEST dataset
                 config: Optional[PipelineConfig] = None,
                 wsi_ext: Optional[List[str]] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            job_dir: Directory to store all outputs
            wsi_dir: Directory containing WSI files
            hest_dataset_root_dir: Optional. Root directory of the MahmoodLab/hest dataset.
                                   If provided, it will scan for existing outputs here.
            config: Pipeline configuration (uses defaults if None)
            wsi_ext: List of allowed WSI file extensions
        """
        self.job_dir = Path(job_dir)
        self.wsi_dir = Path(wsi_dir)
        self.hest_dataset_root_dir = Path(hest_dataset_root_dir) if hest_dataset_root_dir else None # Store the HEST root
        self.config = config or PipelineConfig()
        self.wsi_ext = wsi_ext or ['.tif', '.tiff', '.svs', '.ndpi'] # Default common WSI extensions
        
        # Create output directory structure
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.job_dir / "pipeline_state.json"
        
        # Initialize subdirectories for pipeline generated outputs
        self.seg_dir = self.job_dir / "segmentation"
        self.coords_dir = self.job_dir / f"coordinates_{self.config.mag}x_{self.config.patch_size}px"
        self.features_dir = self.job_dir / f"features_{self.config.mag}x_{self.config.patch_size}px_{self.config.patch_encoder}"
        
        for directory in [self.seg_dir, self.coords_dir, self.features_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = Processor(
            job_dir=str(self.job_dir), # Trident will save outputs relative to this
            wsi_source=str(self.wsi_dir),
            wsi_ext=self.wsi_ext,
            skip_errors=self.config.skip_errors,
            max_workers=self.config.max_workers,
            reader_type=self.config.reader_type,
            mpp=self.config.mpp,
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
                print(f"Warning: Could not load state file {self.state_file}: {e}")
        return {}
    
    def _save_state(self):
        """Save pipeline state to JSON file."""
        data = {k: asdict(v) for k, v in self.samples.items()}
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_sample_state(self, sample_id: str, **kwargs):
        """Update sample state and save to disk."""
        if sample_id not in self.samples:
            # If sample_id not in self.samples, it means discover_samples hasn't run or failed for it
            # Try to find the WSI path for the new sample_id
            wsi_file_found = False
            for ext in self.wsi_ext:
                possible_wsi_path = self.wsi_dir / f"{sample_id}{ext}"
                if possible_wsi_path.is_file():
                    self.samples[sample_id] = SampleState(
                        sample_id=sample_id,
                        wsi_path=str(possible_wsi_path)
                    )
                    wsi_file_found = True
                    break
            if not wsi_file_found:
                # Fallback: search all WSIs for a matching stem
                all_wsi_files = list(self.wsi_dir.glob(f"*{sample_id}*"))
                if self.wsi_ext:
                    all_wsi_files = [f for f in all_wsi_files if f.suffix.lower() in [ext.lower() for ext in self.wsi_ext]]
                
                matching_wsi = [f for f in all_wsi_files if f.stem == sample_id and f.is_file()]
                if matching_wsi:
                    self.samples[sample_id] = SampleState(
                        sample_id=sample_id,
                        wsi_path=str(matching_wsi[0])
                    )
                else:
                    print(f"Warning: WSI file not found for new sample_id: {sample_id}. State will be incomplete.")
                    # Still create a placeholder state
                    self.samples[sample_id] = SampleState(sample_id=sample_id, wsi_path="")

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
            print(f"Warning: No WSI files with extensions {self.wsi_ext} found in {self.wsi_dir}")
        
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
        """Scan for existing output files and update sample states, including HEST dataset specific paths."""
        print("Scanning for existing outputs...")
        # Iterate over a copy of keys as _update_sample_state modifies self.samples
        for sample_id in list(self.samples.keys()): 
            sample_state = self.samples[sample_id]

            # --- Check for Segmentation files ---
            if not sample_state.segmentation_done:
                # Common patterns from Trident/general
                seg_patterns = [
                    f"{sample_id}_segmentation.h5", f"{sample_id}_segmentation.h5ad",
                    f"{sample_id}.h5", f"{sample_id}.h5ad",
                    f"*{sample_id}*seg*.h5", f"*{sample_id}*seg*.h5ad"
                ]
                # HEST specific segmentation patterns
                if self.hest_dataset_root_dir and (self.hest_dataset_root_dir / "tissue_seg").exists():
                    seg_patterns.extend([
                        f"{sample_id}_mask.jpg",
                        f"{sample_id}_mask.pkl",
                    ])
                
                found_seg_path = None
                # Search in job_dir and hest_dataset_root_dir subdirectories
                search_dirs = [self.job_dir, self.seg_dir]
                if self.hest_dataset_root_dir:
                    search_dirs.append(self.hest_dataset_root_dir / "tissue_seg")

                for pattern in seg_patterns:
                    for s_dir in search_dirs:
                        if not s_dir.exists():
                            continue
                        # Use rglob for recursive search within the target directory
                        for f_path in s_dir.rglob(f"**/{pattern}"):
                            if f_path.is_file():
                                found_seg_path = str(f_path)
                                break
                        if found_seg_path:
                            break
                    if found_seg_path:
                        break
            
                if found_seg_path:
                    self._update_sample_state(
                        sample_id,
                        segmentation_done=True,
                        segmentation_path=found_seg_path
                    )
                    # print(f"Found existing segmentation for {sample_id}: {found_seg_path}")


            # --- Check for Coordinate/Patch files ---
            if not sample_state.coordinates_done:
                # Common patterns from Trident/general
                coord_patterns = [
                    f"{sample_id}_coordinates.h5", f"{sample_id}_coordinates.h5ad",
                    f"{sample_id}_coords.h5", f"{sample_id}_coords.h5ad",
                    f"{sample_id}.h5", f"{sample_id}.h5ad", # HEST patches are often {id}.h5
                    f"*{sample_id}*coord*.h5", f"*{sample_id}*coord*.h5ad",
                    f"*{sample_id}*patch*.h5", f"*{sample_id}*patch*.h5ad",
                ]
                
                found_coord_path = None
                # Search in job_dir, coords_dir, and hest_dataset_root_dir/patches
                search_dirs = [self.job_dir, self.coords_dir]
                if self.hest_dataset_root_dir:
                    search_dirs.append(self.hest_dataset_root_dir / "patches") # HEST patches directory

                for pattern in coord_patterns:
                    for s_dir in search_dirs:
                        if not s_dir.exists():
                            continue
                        for f_path in s_dir.rglob(f"**/{pattern}"):
                            if f_path.is_file():
                                found_coord_path = str(f_path)
                                break
                        if found_coord_path:
                            break
                    if found_coord_path:
                        break
            
                if found_coord_path:
                    self._update_sample_state(
                        sample_id,
                        coordinates_done=True,
                        coordinates_path=found_coord_path
                    )
                    # print(f"Found existing coordinates/patches for {sample_id}: {found_coord_path}")

            # --- Check for Feature files ---
            # Even if coordinates are found (e.g., HEST patches), features might still need extraction
            if not sample_state.features_done:
                # Common patterns from Trident/general
                feat_patterns = [
                    f"{sample_id}_features.h5", f"{sample_id}_features.h5ad",
                    f"{sample_id}_feat.h5", f"{sample_id}_feat.h5ad",
                    f"*{sample_id}*feature*.h5", f"*{sample_id}*feature*.h5ad",
                    f"*{sample_id}*feat*.h5", f"*{sample_id}*feat*.h5ad",
                ]
                # HEST 'patches' directory might contain pre-extracted features
                # if so, their filename might follow some pattern
                if self.hest_dataset_root_dir and (self.hest_dataset_root_dir / "patches").exists():
                    feat_patterns.append(f"{sample_id}.h5") # HEST patches themselves might be considered features if they already contain embeddings

                found_feat_path = None
                # Search in job_dir, features_dir, and hest_dataset_root_dir/patches
                search_dirs = [self.job_dir, self.features_dir]
                if self.hest_dataset_root_dir:
                    search_dirs.append(self.hest_dataset_root_dir / "patches") # Search HEST patches directory for feature files too

                for pattern in feat_patterns:
                    for s_dir in search_dirs:
                        if not s_dir.exists():
                            continue
                        for f_path in s_dir.rglob(f"**/{pattern}"):
                            if f_path.is_file():
                                # Check if it's explicitly a feature file or if the HEST patch .h5 contains features
                                # This check might need refinement based on HEST .h5 content
                                if "feature" in f_path.name.lower() or "feat" in f_path.name.lower():
                                    found_feat_path = str(f_path)
                                elif "patches" in str(f_path).lower() and self.hest_dataset_root_dir:
                                    # If it's a HEST patch file, check if it contains 'features' group/dataset
                                    try:
                                        with h5py.File(f_path, 'r') as hf:
                                            if 'features' in hf or 'embeddings' in hf: # Assuming HEST patches might contain these
                                                found_feat_path = str(f_path)
                                    except Exception as e:
                                        pass # Not a valid H5 or no features found
                                if found_feat_path:
                                    break
                        if found_feat_path:
                            break
                    if found_feat_path:
                        break
            
                if found_feat_path:
                    self._update_sample_state(
                        sample_id,
                        features_done=True,
                        features_path=found_feat_path
                    )
                    # print(f"Found existing features for {sample_id}: {found_feat_path}")
        self._save_state() # Ensure state is saved after scan
        print("Scan complete.")
    
    def get_pending_samples(self, task: str) -> List[str]:
        """Get samples that need processing for a specific task."""
        pending = []
        for sample_id, state in self.samples.items():
            if not state.wsi_path or not Path(state.wsi_path).exists():
                print(f"Skipping {sample_id}: WSI file not found at {state.wsi_path}")
                continue # Skip samples where WSI is missing
            
            if task == "segmentation" and not state.segmentation_done:
                pending.append(sample_id)
            # For coordinates, we now check if HEST patches exist as an alternative
            elif task == "coordinates" and not state.coordinates_done:
                # Check if a HEST patch file exists that can serve as coordinates
                hest_patch_file = None
                if self.hest_dataset_root_dir and (self.hest_dataset_root_dir / "patches" / f"{sample_id}.h5").exists():
                    hest_patch_file = self.hest_dataset_root_dir / "patches" / f"{sample_id}.h5"
                
                if not state.segmentation_done: # Segmentation must be done or existing
                    continue # Cannot do coordinate extraction without segmentation (or pre-existing masks/patches)
                elif hest_patch_file and hest_patch_file.is_file():
                    # If HEST patches exist, consider coordinates done
                    # This implies we will use HEST patches for feature extraction
                    # We update the state here to reflect this
                    self._update_sample_state(
                        sample_id,
                        coordinates_done=True,
                        coordinates_path=str(hest_patch_file)
                    )
                    # This sample is now technically "done" for coordinates, so it won't be in pending
                else:
                    # If no HEST patch file, and internal coordinates not done, then it's pending
                    pending.append(sample_id)
            elif task == "features" and not state.features_done:
                if not state.coordinates_done:
                    continue # Cannot extract features without coordinates/patches
                pending.append(sample_id)
        return pending
 
    def run_feature_extraction(self, sample_ids: Optional[List[str]] = None) -> int:
        """Run feature extraction for specified samples.
        Prioritizes HEST patch files if available for input.
        """
        if sample_ids is None:
            sample_ids = self.get_pending_samples("features")
        
        if not sample_ids:
            print("No samples need feature extraction or all are already done.")
            return 0
        
        print(f"Running feature extraction for {len(sample_ids)} samples...")
        
        # Initialize patch encoder
        encoder = encoder_factory(
            self.config.patch_encoder, 
            weights_path=self.config.patch_encoder_ckpt_path
        )
        
        processed_count = 0
        for sample_id in sample_ids:
            state = self.samples[sample_id]
            
            # Determine the input path for patch features (either HEST patches or pipeline-generated coords)
            input_patch_path: Optional[Path] = None
            if state.coordinates_path and Path(state.coordinates_path).exists():
                input_patch_path = Path(state.coordinates_path)
            
            if not input_patch_path:
                print(f"Skipping feature extraction for {sample_id}: No valid coordinates/patches found.")
                continue

            # Ensure the input path is an H5 file (or anndata compatible)
            if not str(input_patch_path).lower().endswith(('.h5', '.h5ad')):
                print(f"Skipping feature extraction for {sample_id}: Input path {input_patch_path} is not a recognized H5/H5AD format.")
                continue

            # Determine output path for features
            # Trident's `run_patch_feature_extraction_job` will save to a subdirectory under `job_dir`
            # based on its internal logic, typically `job_dir/patch_features/` or similar.
            # We will then search for the generated file.
            
            print(f"  Extracting features for {sample_id} from {input_patch_path}...")
            try:
                # The processor expects coords_dir to be a directory
                # If input_patch_path is a specific file, we need to pass its parent directory
                coords_input_dir_for_trident = input_patch_path.parent 
                
                # If Trident's processor was initialized with job_dir, it will save
                # new features to job_dir/patch_features or similar
                self.processor.run_patch_feature_extraction_job(
                    coords_dir=str(coords_input_dir_for_trident), # Pass the directory where the input .h5 is
                    patch_encoder=encoder,
                    device=f'cuda:{self.config.gpu}',
                    saveas='h5', # Ensure saving as H5
                    batch_limit=self.config.batch_size,
                )
                
                # After running, check for the output feature file.
                # Trident typically saves features under `job_dir/patch_features/` or `job_dir/features_<encoder_name>/`
                # or similar. We need to be robust in finding it.
                
                found_feat_output_path = None
                # Patterns for output files generated by Trident based on its internal saving logic
                output_patterns = [
                    self.features_dir / f"{sample_id}_features.h5", # Our pre-defined features_dir
                    self.features_dir / f"{sample_id}.h5",
                    self.job_dir / "patch_features" / f"{sample_id}_features.h5", # Common Trident default
                    self.job_dir / "patch_features" / f"{sample_id}.h5",
                    self.job_dir / f"features_{self.config.mag}x_{self.config.patch_size}px_{self.config.patch_encoder}" / f"{sample_id}_features.h5",
                    self.job_dir / f"features_{self.config.mag}x_{self.config.patch_size}px_{self.config.patch_encoder}" / f"{sample_id}.h5",
                ]

                # Also search more broadly in the job_dir for any .h5 output matching the sample_id
                for p_path in self.job_dir.rglob(f"**/*{sample_id}*.h5"):
                    if "features" in p_path.name.lower() or "feat" in p_path.name.lower():
                        output_patterns.append(p_path)
                
                # Check for the first existing path
                for path_candidate in output_patterns:
                    if path_candidate.exists() and path_candidate.is_file():
                        found_feat_output_path = str(path_candidate)
                        break

                if found_feat_output_path:
                    self._update_sample_state(
                        sample_id,
                        features_done=True,
                        features_path=found_feat_output_path
                    )
                    processed_count += 1
                    print(f"  Successfully extracted and saved features for {sample_id} to {found_feat_output_path}")
                else:
                    print(f"  Warning: Feature output not found for {sample_id} after running job. Please check Trident logs.")

            except Exception as e:
                print(f"  Error extracting features for {sample_id}: {e}")
        
        return processed_count
    


    

        """Debug method to show what files actually exist in the output directories."""
        print("\n" + "="*60)
        print("DEBUG: OUTPUT FILES")
        print("="*60)
        
        print(f"\nJob directory: {self.job_dir}")
        print(f"Segmentation directory: {self.seg_dir}")
        print(f"Coordinates directory: {self.coords_dir}")
        print(f"Features directory: {self.features_dir}")
        if self.hest_dataset_root_dir:
            print(f"HEST Dataset Root: {self.hest_dataset_root_dir}")
            print(f"HEST Patches directory: {self.hest_dataset_root_dir / 'patches'}")
            print(f"HEST Tissue Seg directory: {self.hest_dataset_root_dir / 'tissue_seg'}")

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
            
            # Search for files containing this sample ID in job_dir and HEST dirs
            search_roots = [self.job_dir]
            if self.hest_dataset_root_dir:
                search_roots.extend([self.hest_dataset_root_dir / "tissue_seg", self.hest_dataset_root_dir / "patches"])

            found_files = []
            for root in search_roots:
                if root.exists():
                    found_files.extend(list(root.glob(f"**/*{sample_id}*")))
            
            if found_files:
                print(f"    Found {len(found_files)} files:")
                for f in sorted(found_files):
                    if f.is_file():
                        # Try to show relative path, fall back to absolute
                        try:
                            print(f"      {f.relative_to(self.job_dir)}")
                        except ValueError:
                            try:
                                if self.hest_dataset_root_dir:
                                    print(f"      (HEST) {f.relative_to(self.hest_dataset_root_dir)}")
                                else:
                                    print(f"      {f}")
                            except ValueError:
                                print(f"      {f}")
            else:
                print(f"    No relevant files found for this sample in job or HEST directories.")
        
        print("="*60)
  
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
        
        self._save_state()
        print("State refresh completed.")
        self.print_summary()
    
   
        """Convert .h5ad file to .h5 format. (Simplified to copy if anndata not present)"""
        try:
            import anndata as ad
            
            print(f"Converting {h5ad_path} to {h5_path} using anndata...")
            adata = ad.read_h5ad(h5ad_path)
            
            # Save as .h5 format that is compatible with Trident/h5py expectations
            # Trident's H5 structure for patches/coords usually has 'coords' dataset
            # and for features 'coords' and 'features' datasets.
            with h5py.File(h5_path, 'w') as h5_file:
                # Assuming 'X' in adata contains features if it's a feature file, or image data if it's a patch file
                if hasattr(adata, 'X') and adata.X is not None:
                    h5_file.create_dataset('X', data=adata.X, compression="gzip")
                
                # Try to map common anndata obs to 'coords' or other relevant datasets
                if 'coords' in adata.obs.columns:
                    h5_file.create_dataset('coords', data=adata.obs['coords'].values, compression="gzip")
                elif 'x_coord' in adata.obs.columns and 'y_coord' in adata.obs.columns:
                    # Create a 'coords' dataset from x,y
                    coords_data = adata.obs[['x_coord', 'y_coord']].values
                    h5_file.create_dataset('coords', data=coords_data, compression="gzip")
                
                # If it's a feature file, save features
                if hasattr(adata, 'obsm') and 'X_spatial' in adata.obsm: # Example for spatial embeddings
                    h5_file.create_dataset('features', data=adata.obsm['X_spatial'], compression="gzip")
                elif 'features' in adata.obs.columns: # If features are directly in obs
                     h5_file.create_dataset('features', data=adata.obs['features'].values, compression="gzip")
                elif 'X' in h5_file and h5_file['X'].shape[1] > 2: # Heuristic: if X is not just coords
                     # If X is already deemed features, copy it directly
                    pass
                    
                # Store any relevant metadata as attributes
                for key, value in adata.uns.items():
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        try:
                            h5_file.attrs[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
                        except TypeError:
                            pass # Skip non-serializable attributes
            
            return True
            
        except ImportError:
            print(f"Warning: anndata not available. Attempting simple file copy for {h5ad_path}.")
            try:
                import shutil
                shutil.copy2(h5ad_path, h5_path)
                print(f"Copied {h5ad_path} to {h5_path} (simple copy). Note: Full anndata to Trident H5 conversion skipped.")
                return True
            except Exception as e:
                print(f"Error during simple copy of {h5ad_path} to {h5_path}: {e}")
                return False
        except Exception as e:
            print(f"Warning: Could not convert {h5ad_path} to .h5 format (Trident-compatible): {e}")
            return False
    
  
        """Convert any .h5ad files found in state to .h5 format.
        This will attempt a proper conversion if anndata is available,
        otherwise it will perform a simple file copy/rename.
        """
        converted_count = 0
        
        # Iterate over a copy of keys as _update_sample_state modifies self.samples
        for sample_id in list(self.samples.keys()):
            sample_state = self.samples[sample_id]

            # Check and convert segmentation path
            if sample_state.segmentation_path and sample_state.segmentation_path.lower().endswith('.h5ad'):
                original_path = Path(sample_state.segmentation_path)
                h5_path = original_path.with_suffix('.h5')
                if not h5_path.exists() or original_path.stat().st_mtime > h5_path.stat().st_mtime:
                    print(f"Attempting to convert segmentation for {sample_id}: {original_path.name}")
                    if self.convert_h5ad_to_h5(str(original_path), str(h5_path)):
                        self._update_sample_state(sample_id, segmentation_path=str(h5_path))
                        converted_count += 1
                else:
                    print(f"Skipping segmentation conversion for {sample_id}: .h5 version exists and is up-to-date.")
                    self._update_sample_state(sample_id, segmentation_path=str(h5_path)) # Ensure state points to H5
            
            # Check and convert coordinates path
            if sample_state.coordinates_path and sample_state.coordinates_path.lower().endswith('.h5ad'):
                original_path = Path(sample_state.coordinates_path)
                h5_path = original_path.with_suffix('.h5')
                if not h5_path.exists() or original_path.stat().st_mtime > h5_path.stat().st_mtime:
                    print(f"Attempting to convert coordinates for {sample_id}: {original_path.name}")
                    if self.convert_h5ad_to_h5(str(original_path), str(h5_path)):
                        self._update_sample_state(sample_id, coordinates_path=str(h5_path))
                        converted_count += 1
                else:
                    print(f"Skipping coordinates conversion for {sample_id}: .h5 version exists and is up-to-date.")
                    self._update_sample_state(sample_id, coordinates_path=str(h5_path)) # Ensure state points to H5

            # Check and convert features path
            if sample_state.features_path and sample_state.features_path.lower().endswith('.h5ad'):
                original_path = Path(sample_state.features_path)
                h5_path = original_path.with_suffix('.h5')
                if not h5_path.exists() or original_path.stat().st_mtime > h5_path.stat().st_mtime:
                    print(f"Attempting to convert features for {sample_id}: {original_path.name}")
                    if self.convert_h5ad_to_h5(str(original_path), str(h5_path)):
                        self._update_sample_state(sample_id, features_path=str(h5_path))
                        converted_count += 1
                else:
                    print(f"Skipping features conversion for {sample_id}: .h5 version exists and is up-to-date.")
                    self._update_sample_state(sample_id, features_path=str(h5_path)) # Ensure state points to H5

        if converted_count > 0:
            print(f"Successfully converted/updated state for {converted_count} files from .h5ad to .h5 format.")
        else:
            print("No .h5ad files found needing conversion or all are up-to-date.")
        
        return converted_count