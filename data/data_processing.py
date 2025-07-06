import os
import numpy as np
import scanpy as sc
import tifffile

def extract_wsi_from_h5ad(file_h5ad_path, slide_key='20x_slide'):
    """
    Estrae l'immagine WSI dal campo `uns` di un AnnData.

    Parametri:
    - file_h5ad_path: path al file h5ad
    - slide_key: chiave nel campo `uns` dove è salvata l'immagine (default '20x_slide')

    Ritorna:
    - np.ndarray: array immagine della WSI
    """
    adata = sc.read_h5ad(file_h5ad_path)

    if slide_key not in adata.uns:
        raise KeyError(f"Key '{slide_key}' not found in adata.uns")

    wsi_data = adata.uns[slide_key]

    return wsi_data

def save_wsi(wsi_image, output_path, verbose=False):
    """
    Salva WSI come BigTIFF compatibile con OpenSlide.
    
    Args:
        wsi_image: Array immagine WSI
        output_path: Path di output per il file TIFF
        verbose: Se True, stampa informazioni dettagliate
    
    Returns:
        str: Path del file salvato o None se fallisce
    """
    if not isinstance(wsi_image, np.ndarray):
        wsi_image = np.array(wsi_image)
    
    # Preprocessing standard
    if wsi_image.dtype != np.uint8:
        if wsi_image.dtype in [np.float64, np.float32]:
            if wsi_image.max() <= 1.0:
                wsi_image = (wsi_image * 255).astype(np.uint8)
            else:
                wsi_image = ((wsi_image - wsi_image.min()) / (wsi_image.max() - wsi_image.min()) * 255).astype(np.uint8)
        else:
            wsi_image = ((wsi_image - wsi_image.min()) / (wsi_image.max() - wsi_image.min()) * 255).astype(np.uint8)
    
    # Converti RGBA a RGB
    if len(wsi_image.shape) == 3 and wsi_image.shape[2] == 4:
        if verbose:
            print("Converting RGBA to RGB for OpenSlide compatibility...")
        wsi_image = wsi_image[:, :, :3]
    
    if verbose:
        print(f"Image shape: {wsi_image.shape}, dtype: {wsi_image.dtype}")
    
    try:
        # Salva come BigTIFF con parametri ottimizzati per OpenSlide
        tifffile.imwrite(
            output_path,
            wsi_image,
            bigtiff=True,
            compression='jpeg',  # JPEG spesso più compatibile di LZW
            photometric='rgb',
            planarconfig='contig',  # Importante per OpenSlide
            tile=(512, 512),  # Tiled TIFF per performance migliori
            metadata={'Software': 'tifffile'}
        )
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if verbose:
            print(f"BigTIFF saved: {output_path} ({file_size_mb:.1f} MB)")
        return output_path
        
    except Exception as e:
        if verbose:
            print(f"BigTIFF error: {e}")
        
        # Fallback: prova senza compressione JPEG
        try:
            tifffile.imwrite(
                output_path,
                wsi_image,
                bigtiff=True,
                compression='lzw',
                photometric='rgb',
                planarconfig='contig'
            )
            if verbose:
                print(f"Saved with LZW compression: {output_path}")
            return output_path
        except Exception as e2:
            if verbose:
                print(f"Error with LZW compression: {e2}")
            return None

def process_h5ad_file(
    file_h5ad_path,
    output_raw_dir=None,
    slide_key='20x_slide',
    config=None,
    verbose=False
):
    """
    Estrae la WSI e i dati spatial, salvandoli come TIFF e pickle.
    
    Args:
        file_h5ad_path (str): Path al file h5ad di input
        output_raw_dir (str, optional): Directory di output. Se None, usa config['interim_data_dir']
        slide_key (str): Chiave nel campo `uns` dove è salvata l'immagine
        config (dict, optional): Dizionario di configurazione con i path
        verbose (bool): Se True, stampa informazioni dettagliate
    
    Returns:
        tuple: (raw_wsi_path, spatial_data_path) - Path del file TIFF e dei dati spatial salvati
    
    Raises:
        RuntimeError: Se l'estrazione WSI o spatial data fallisce
    """
    # Determina la directory di output usando CONFIG se disponibile
    if output_raw_dir is None:
        if config is not None and 'interim_data_dir' in config:
            output_raw_dir = os.path.join(config['interim_data_dir'], 'raw_wsi')
        else:
            # Fallback al path di default se CONFIG non è disponibile
            output_raw_dir = 'data/interim/raw_wsi'
    
    os.makedirs(output_raw_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(file_h5ad_path))[0]
    raw_wsi_path = os.path.join(output_raw_dir, filename + '.tiff')
    spatial_data_path = os.path.join(output_raw_dir, filename + '_spatial.pkl')

    # Salva WSI se non esiste
    if not os.path.exists(raw_wsi_path):
        if verbose:
            print(f"Extracting WSI for {filename}...")
        try:
            wsi_data = extract_wsi_from_h5ad(file_h5ad_path, slide_key=slide_key)
            result_path = save_wsi(wsi_data, raw_wsi_path)
            if result_path is None:
                raise RuntimeError(f"Failed to save WSI for {filename}")
        except Exception as e:
            raise RuntimeError(f"WSI extraction failed for {filename}: {str(e)}")

    # Salva dati spatial se non esistono
    if not os.path.exists(spatial_data_path):
        if verbose:
            print(f"Extracting spatial data for {filename}...")
        try:
            spatial_data = extract_spatial_data(file_h5ad_path)
            import pickle
            with open(spatial_data_path, 'wb') as f:
                pickle.dump(spatial_data, f)
        except Exception as e:
            raise RuntimeError(f"Spatial data extraction failed for {filename}: {str(e)}")

    return raw_wsi_path, spatial_data_path




def extract_spatial_data(h5ad_path):
    """
    Estrae i dati spatial dal file h5ad per il calcolo dei reward biologici.
    
    Args:
        h5ad_path (str): Path al file .h5ad
        
    Returns:
        dict: Dizionario contenente coordinate e probabilità dei tessuti
    """
    adata = sc.read_h5ad(h5ad_path)
    
    # Estrai coordinate spaziali (preferisci pixel coordinates)
    if 'x_pixel' in adata.obs and 'y_pixel' in adata.obs:
        coords = adata.obs[['x_pixel', 'y_pixel']].values.astype(np.float32)
    else:
        coords = adata.obsm['spatial'].astype(np.float32)
    
    # Estrai probabilità dei tessuti per il calcolo del reward
    tissue_probs = {}
    prob_mapping = {
        'tumor': 'Tumor Probability',
        'stroma': 'Stroma Probability', 
        'blood_necrosis': 'Blood and necrosis Probability'
    }
    
    for key, col_name in prob_mapping.items():
        if col_name in adata.obs.columns:
            tissue_probs[key] = adata.obs[col_name].values.astype(np.float32)
        else:
            print(f"Warning: {col_name} not found in obs columns")
            tissue_probs[key] = np.zeros(len(coords), dtype=np.float32)
    
    spatial_data = {
        'coords': coords,
        'tissue_probs': tissue_probs,
        'shape': coords.shape,
        'coord_range': {
            'x_min': coords[:, 0].min(),
            'x_max': coords[:, 0].max(),
            'y_min': coords[:, 1].min(),
            'y_max': coords[:, 1].max()
        }
    }
    
    return spatial_data