import torch
import os
import h5py
import numpy as np
from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SlideInferenceExtractor:
    """
    A class to perform slide-level inference and extract attention matrices.
    """
    def __init__(self, model_name, output_dir, device=None):
        """
        Initializes the SlideInferenceExtractor.

        Args:
            model_name (str): The name of the slide-level model to use.
            output_dir (str): Directory to save attention matrices.
            device (torch.device, optional): The device to run the model on. Defaults to None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = slide_encoder_factory(model_name).to(self.device)
        self.model.eval()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.attention_outputs = []
        self.hook_handles = []

    def _attention_hook(self, module, input, output):
        """A hook to capture attention module outputs."""
        self.attention_outputs.append(output.detach().cpu())

    def _register_hooks(self):
        """Registers forward hooks to the attention modules of the model."""
        self._clear_hooks()
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_encoder'):
            for block in self.model.model.vision_encoder.blocks.modules_list:
                handle = block.attn.register_forward_hook(self._attention_hook)
                self.hook_handles.append(handle)

    def _clear_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.attention_outputs = []

    def run_inference(self, embeddings_path, patch_size_lv0=256, max_patches=None):
        """
        Runs slide-level inference on embeddings from an H5 file.

        Args:
            embeddings_path (str): Path to the H5 file containing embeddings and coordinates.
            patch_size_lv0 (int): The patch size at level 0. Defaults to 256.
            max_patches (int, optional): Maximum number of patches to use per slide. Defaults to None (no limit).
        """
        try:
            with h5py.File(embeddings_path, 'r') as hf:
                features = torch.from_numpy(hf['embeddings'][:]).float()
                coords = torch.from_numpy(hf['coords'][:])
        except Exception as e:
            print(f"Error reading data from {embeddings_path}: {e}")
            return None

        if max_patches and features.shape[0] > max_patches:
            print(f"Warning: {os.path.basename(embeddings_path)} has {features.shape[0]} patches. Subsampling to {max_patches}.")
            indices = np.random.choice(features.shape[0], max_patches, replace=False)
            features = features[indices]
            coords = coords[indices]

        self._register_hooks()

        batch = {
            'features': features,
            'coords': coords,
            'attributes': {'patch_size_level0': patch_size_lv0}
        }
        
        try:
            with torch.inference_mode():
                slide_embedding = self.model(batch, device=self.device)
        except torch.cuda.OutOfMemoryError:
            print(f"\nCUDA Out of Memory on {os.path.basename(embeddings_path)}. Skipping this slide as it has too many patches for the available GPU memory.\n")
            self._clear_hooks()
            return None
        except Exception as e:
            print(f"\nAn unexpected error occurred on {os.path.basename(embeddings_path)}: {e}. Skipping.\n")
            self._clear_hooks()
            return None

        slide_name = os.path.splitext(os.path.basename(embeddings_path))[0]
        attention_dir = os.path.join(self.output_dir, slide_name)
        os.makedirs(attention_dir, exist_ok=True)

        if not self.attention_outputs:
            print(f"Warning: No attention maps were captured for {slide_name}.")
        else:
            # Salva embeddings patch-wise (output dell'attention)
            patch_embeddings_dir = os.path.join(attention_dir, 'patch_embeddings')
            os.makedirs(patch_embeddings_dir, exist_ok=True)
            for i, patch_embeds in enumerate(self.attention_outputs):
                np.save(os.path.join(patch_embeddings_dir, f'embeddings_block_{i}.npy'), patch_embeds.numpy())
        
        self._clear_hooks()

        return slide_embedding.cpu()

    def extract_attention_from_folder(self, input_folder, attention_output_folder, slide_embedding_output_folder, patch_size_lv0=256, max_patches=None):
        """
        Extracts attention maps and slide embeddings from a folder of H5 files.

        Args:
            input_folder (str): Path to the folder containing H5 files with embeddings and coordinates.
            attention_output_folder (str): Path to the folder where attention maps will be saved.
            slide_embedding_output_folder (str): Path to the folder where slide embeddings will be saved.
            patch_size_lv0 (int): The patch size at level 0. Defaults to 256.
            max_patches (int, optional): Maximum number of patches to use per slide. Defaults to None (no limit).
        """
        os.makedirs(attention_output_folder, exist_ok=True)
        os.makedirs(slide_embedding_output_folder, exist_ok=True)

        file_list = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
        for filename in tqdm(file_list, desc="Processing Slides"):
            embeddings_path = os.path.join(input_folder, filename)
            slide_embedding = self.run_inference(embeddings_path, patch_size_lv0, max_patches=max_patches)

            if slide_embedding is not None:
                slide_name = os.path.splitext(filename)[0]
                embedding_output_path = os.path.join(slide_embedding_output_folder, f"{slide_name}_embedding.npy")
                np.save(embedding_output_path, slide_embedding.numpy())

    