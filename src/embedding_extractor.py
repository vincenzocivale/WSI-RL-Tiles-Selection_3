import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from trident.patch_encoder_models.load import encoder_factory
import os

class EmbeddingExtractor:
    """
    A class to extract embeddings from patches stored in an H5 file.
    """
    def __init__(self, model_name, weights_path=None, device=None, patches_key='patches'):
        """
        Initializes the EmbeddingExtractor.

        Args:
            model_name (str): The name of the foundation model to use.
            weights_path (str, optional): Path to custom model weights. Defaults to None.
            device (torch.device, optional): The device to run the model on. Defaults to None.
            patches_key (str, optional): The key for the patches dataset in the H5 file. Defaults to 'patches'.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.patches_key = patches_key
        self.model = encoder_factory(model_name, weights_path=weights_path)
        
        # Get the transformations and precision from the loaded model
        self.transforms = self.model.eval_transforms
        self.precision = self.model.precision

        self.model.to(self.device)
        self.model.to(self.precision)
        self.model.eval()

    def extract_embeddings(self, h5_path, output_path, batch_size=64):
        """
        Extracts embeddings from an H5 file containing patches and saves them to another H5 file.

        Args:
            h5_path (str): Path to the input H5 file with patches.
            output_path (str): Path to save the output H5 file with embeddings.
            batch_size (int, optional): The batch size for inference. Defaults to 64.
        """
        try:
            with h5py.File(h5_path, 'r') as hf:
                patches = hf[self.patches_key][:]
        except Exception as e:
            print(f"Error reading patches from {h5_path}: {e}")
            return

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(patches), batch_size), desc="Extracting Embeddings"):
                batch = patches[i:i+batch_size]
                
                # Convert to PIL images and apply transforms
                pil_images = [Image.fromarray(p) for p in batch]
                
                # Ensure the transform is not None
                if self.transforms:
                    transformed_batch = torch.stack([self.transforms(p) for p in pil_images]).to(self.device, dtype=self.precision)
                else:
                    # Fallback to a basic ToTensor if no transforms are provided
                    transformed_batch = torch.stack([transforms.ToTensor()(p) for p in pil_images]).to(self.device)

                # Get embeddings
                embedding_batch = self.model(transformed_batch)
                embeddings.append(embedding_batch.cpu().numpy())
        
        if not embeddings:
            print("No embeddings were generated.")
            return

        all_embeddings = np.vstack(embeddings)
        
        try:
            with h5py.File(output_path, 'w') as hf:
                hf.create_dataset('embeddings', data=all_embeddings)
            print(f"Embeddings saved to {output_path}")
        except Exception as e:
            print(f"Error saving embeddings to {output_path}: {e}")

    def extract_embeddings_from_folder(self, input_folder, output_folder, batch_size=64):
        """
        Extracts embeddings from all H5 files in a folder and saves them to an output folder.

        Args:
            input_folder (str): Path to the folder containing H5 files with patches.
            output_folder (str): Path to the folder where embedding H5 files will be saved.
            batch_size (int, optional): The batch size for inference. Defaults to 64.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith('.h5'):
                h5_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                print(f"Processing {h5_path}...")
                self.extract_embeddings(h5_path, output_path, batch_size)
