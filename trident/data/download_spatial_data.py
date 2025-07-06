import os
from huggingface_hub import login, hf_hub_download, snapshot_download
import pandas as pd
from tqdm import tqdm
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_huggingface(token):
    """
    Setup connection to Hugging Face using your token
    Args:
        token (str): Your Hugging Face token
    """
    try:
        login(token=token)
        logging.info("Successfully logged into Hugging Face.")
    except Exception as e:
        logging.error(f"Failed to log into Hugging Face: {e}")
        raise # Rilancia l'eccezione per fermare l'esecuzione se il login fallisce

def get_metadata(repo_id="nonchev/TCGA_digital_spatial_transcriptomics", filename="metadata_2025-05-21.csv"):
    """
    Download and return the metadata for the dataset
    Args:
        repo_id (str): Repository ID on Hugging Face
        filename (str): Name of the metadata file
    Returns:
        pd.DataFrame: Metadata information
    """
    logging.info(f"Downloading metadata file: {filename} from {repo_id}")
    try:
        file_path = hf_hub_download(repo_id=repo_id, 
                                     filename=filename, 
                                     repo_type="dataset")
        logging.info(f"Metadata downloaded to: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to download or read metadata: {e}")
        raise

def download_data(repo_id, local_dir, allow_patterns=None, repo_type="dataset"):
    """
    Generic function to download data using snapshot_download.
    Handles progress and ensures correct directory structure.
    Args:
        repo_id (str): Repository ID on Hugging Face
        local_dir (str): Directory to save the downloaded data
        allow_patterns (list or str): Optional list of patterns to filter files.
        repo_type (str): Type of Hugging Face repository ('dataset', 'model', 'space')
    Returns:
        str: Path to the downloaded local directory.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    logging.info(f"Starting download from {repo_id} to {local_dir} with patterns: {allow_patterns}")
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            repo_type=repo_type,
            # tqdm_class=tqdm # Puoi abilitare questo per una progress bar interna a snapshot_download
        )
        logging.info(f"Download completed successfully. Data saved to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        raise

def download_cancer_type(cancer_type, slide_type=None, output_base_dir='data/raw', repo_id="nonchev/TCGA_digital_spatial_transcriptomics"):
    """
    Download data for a specific cancer type and optionally slide type.
    This will download all files matching the pattern into the appropriate subdirectories.
    Args:
        cancer_type (str): Cancer type (e.g., 'TCGA_KIRC', 'TCGA_SKCM')
        slide_type (str): Optional - 'FF' or 'FFPE'
        output_base_dir (str): Base directory to save the downloaded data.
                               The data will be in output_base_dir/cancer_type/slide_type/...
        repo_id (str): Repository ID on Hugging Face
    """
    patterns = []
    if slide_type:
        patterns.append(f"{cancer_type}/{slide_type}/*")
    else:
        # Se non specifichi slide_type, scarica entrambi FF e FFPE per quel cancer_type
        patterns.append(f"{cancer_type}/FF/*")
        patterns.append(f"{cancer_type}/FFPE/*")
    
    # snapshot_download scaricherà direttamente nella struttura: output_base_dir/TCGA_KIRC/FF/file.h5ad.gz
    logging.info(f"Downloading for cancer type: {cancer_type}, slide type: {slide_type if slide_type else 'All'}")
    download_data(repo_id=repo_id, local_dir=output_base_dir, allow_patterns=patterns)


def download_first_n_samples(n, output_base_dir='data/raw', repo_id="nonchev/TCGA_digital_spatial_transcriptomics"):
    """
    Download the first n samples from the dataset using metadata.
    This is now more efficient as it uses allow_patterns for snapshot_download once.
    Args:
        n (int): Number of samples to download
        output_base_dir (str): Base directory to save the downloaded data.
        repo_id (str): Repository ID on Hugging Face
    """
    metadata = get_metadata(repo_id=repo_id)
    sample_paths = metadata['file_path'].head(n).tolist()
    
    logging.info(f"Preparing to download first {n} samples. This might take a while if many files.")
    download_data(repo_id=repo_id, local_dir=output_base_dir, allow_patterns=sample_paths)

def download_all_samples(output_base_dir='data/raw', repo_id="nonchev/TCGA_digital_spatial_transcriptomics"):
    """
    Download all samples from the dataset using metadata.
    This is the most efficient way to download all known files by pattern.
    Args:
        output_base_dir (str): Base directory to save the downloaded data.
        repo_id (str): Repository ID on Hugging Face
    """
    metadata = get_metadata(repo_id=repo_id)
    sample_paths = metadata['file_path'].tolist()
    
    logging.info(f"Preparing to download ALL samples. This might take a very long time depending on dataset size.")
    download_data(repo_id=repo_id, local_dir=output_base_dir, allow_patterns=sample_paths)