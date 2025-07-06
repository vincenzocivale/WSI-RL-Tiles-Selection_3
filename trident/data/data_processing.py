import os
import numpy as np
import scanpy as sc
import tifffile
import pickle
import logging

# Configura il logging per visualizzare i messaggi di stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Costante per il diametro fisico dello spot Visium in micron (standard per le slide Visium)
VISIUM_SPOT_DIAMETER_MICRONS = 65.0 

def extract_wsi_and_spatial_from_h5ad(file_h5ad_path: str, slide_key: str = '20x_slide'):
    """
    Carica un AnnData, estrae l'immagine WSI, i dati spaziali e il MPP da esso.
    Questo evita di leggere il file h5ad più volte e fornisce tutti i dati necessari.

    Args:
        file_h5ad_path (str): Path al file h5ad.
        slide_key (str): Chiave nel campo `uns` dove è salvata l'immagine WSI.

    Returns:
        tuple: (adata: AnnData, wsi_image: np.ndarray | None, spatial_data: dict, mpp: float | None)
               Restituisce l'oggetto AnnData completo, l'array immagine della WSI (o None se non trovata),
               i dati spaziali elaborati e il MPP calcolato (o None se non determinabile).
    
    Raises:
        FileNotFoundError: Se il file h5ad non esiste.
        Exception: Per altri errori durante la lettura o l'estrazione.
    """
    logging.info(f"Tentativo di leggere AnnData da {file_h5ad_path}")
    try:
        adata = sc.read_h5ad(file_h5ad_path)
        logging.info(f"Lettura AnnData da {file_h5ad_path} completata con successo.")
    except FileNotFoundError:
        logging.error(f"File AnnData non trovato: {file_h5ad_path}")
        raise
    except Exception as e:
        logging.error(f"Errore durante la lettura di AnnData {file_h5ad_path}: {e}")
        raise

    # --- Estrazione WSI ---
    wsi_image = None
    if slide_key in adata.uns:
        wsi_candidate = adata.uns[slide_key]
        if isinstance(wsi_candidate, np.ndarray):
            wsi_image = wsi_candidate
            logging.info(f"Immagine WSI trovata in adata.uns['{slide_key}'] con shape {wsi_image.shape} e dtype {wsi_image.dtype}")
            if wsi_image.shape[2] == 4:
                logging.info("L'immagine WSI sembra essere RGBA. Verrà convertita in RGB durante il salvataggio.")
        else:
            logging.warning(f"I dati per '{slide_key}' in adata.uns non sono un array NumPy (tipo: {type(wsi_candidate)}). L'immagine WSI non può essere estratta nel formato previsto.")
    else:
        logging.warning(f"Chiave '{slide_key}' non trovata in adata.uns. L'immagine WSI non verrà estratta.")
    
    # --- Estrazione dati spaziali ---
    spatial_data = extract_spatial_data_from_adata(adata)

    # --- Estrazione o calcolo del MPP ---
    calculated_mpp = None
    if 'scaled_slide_info' in adata.uns and isinstance(adata.uns['scaled_slide_info'], dict):
        slide_info = adata.uns['scaled_slide_info']
        if 'spot_diameter_20x' in slide_info and slide_info['spot_diameter_20x'] > 0:
            spot_diameter_pixels = float(slide_info['spot_diameter_20x'])
            calculated_mpp = VISIUM_SPOT_DIAMETER_MICRONS / spot_diameter_pixels
            logging.info(f"MPP calcolato da 'spot_diameter_20x' ({spot_diameter_pixels} pixel) in 'scaled_slide_info': {calculated_mpp:.4f} µm/pixel.")
        else:
            logging.warning("'spot_diameter_20x' non trovato o è zero in adata.uns['scaled_slide_info']. Impossibile determinare automaticamente il MPP.")
    else:
        logging.warning("'scaled_slide_info' non trovato in adata.uns. Impossibile determinare automaticamente il MPP.")

    return adata, wsi_image, spatial_data, calculated_mpp


def save_wsi(wsi_image: np.ndarray, output_path: str, mpp: float, verbose: bool = False) -> str | None:
    """
    Salva la WSI come BigTIFF compatibile con OpenSlide, includendo metadati MPP.
    Include il preprocessing per convertire tipi di dati e formati di canale.
    
    Args:
        wsi_image (np.ndarray): Array immagine WSI.
        output_path (str): Path di output per il file TIFF.
        mpp (float): Microns Per Pixel dell'immagine WSI. Cruciale per i metadati OpenSlide.
        verbose (bool): Se True, stampa informazioni dettagliate.
    
    Returns:
        str: Path del file salvato o None se fallisce.
    """
    if not isinstance(wsi_image, np.ndarray) or wsi_image.size == 0:
        logging.error(f"L'input wsi_image non è un array NumPy valido o è vuoto: {type(wsi_image)}. Impossibile salvare la WSI.")
        raise ValueError("Impossibile salvare la WSI: l'immagine di input non è valida.")
    
    if mpp <= 0:
        logging.error(f"Valore MPP non valido: {mpp}. MPP deve essere un float positivo. Impossibile salvare la WSI con i metadati di risoluzione corretti.")
        return None

    # Lavoriamo su una copia per non modificare l'array originale in memoria
    processed_image = wsi_image.copy()

    # --- Preprocessing per assicurare uint8 e RGB ---
    if processed_image.dtype != np.uint8:
        if verbose:
            logging.info(f"Conversione immagine da {processed_image.dtype} a uint8.")
        if np.issubdtype(processed_image.dtype, np.floating):
            if processed_image.max() <= 1.0 + 1e-6 and processed_image.min() >= 0.0 - 1e-6:
                processed_image = (processed_image * 255).astype(np.uint8)
            else:
                min_val = processed_image.min()
                max_val = processed_image.max()
                if max_val - min_val == 0:
                    processed_image = np.zeros_like(processed_image, dtype=np.uint8)
                else:
                    processed_image = ((processed_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            min_val = processed_image.min()
            max_val = processed_image.max()
            if max_val - min_val == 0:
                processed_image = np.zeros_like(processed_image, dtype=np.uint8)
            else:
                processed_image = ((processed_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Converti RGBA a RGB (rimuovendo il canale alpha)
    if len(processed_image.shape) == 3 and processed_image.shape[2] == 4:
        if verbose:
            logging.info("Conversione RGBA a RGB (rimozione canale alpha).")
        processed_image = processed_image[:, :, :3]
    elif len(processed_image.shape) == 2: # Se in scala di grigi (2D), converti a RGB replicando i canali
        if verbose:
            logging.info("Conversione immagine in scala di grigi (2D) a RGB replicando i canali.")
        processed_image = np.stack([processed_image, processed_image, processed_image], axis=-1)
    
    if not (len(processed_image.shape) == 3 and processed_image.shape[2] == 3):
        logging.error(f"L'immagine processata ha una shape inaspettata {processed_image.shape} per RGB. Attesa (H, W, 3). Annullamento salvataggio WSI.")
        return None

    if verbose:
        logging.info(f"Immagine finale per il salvataggio: shape={processed_image.shape}, dtype={processed_image.dtype}")
    
    try:
        # Calculate resolution in pixels per centimeter
        # 1 cm = 10000 microns
        resolution_value = 10000 / mpp 
        
        tifffile.imwrite(
            output_path,
            processed_image,
            bigtiff=True,          # Necessario per immagini grandi (>4GB)
            compression='jpeg',    # Ottima compressione per immagini istologiche
            photometric='rgb',     # Indica che l'immagine ha canali RGB
            planarconfig='contig', # Canali interfoliati (R1G1B1R2G2B2...) preferito per OpenSlide
            tile=(512, 512),       # Immagine divisa in tiles, migliora la lettura random-access
            resolution=(resolution_value, resolution_value), # Risoluzione in pixel (senza unità)
            resolutionunit='cm',   # Unità di risoluzione specificata separatamente
            metadata={'Software': 'Python tifffile', 'MPP_microns_per_pixel': mpp, 'Source': 'AnnData_uns_extract'}
        )
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"BigTIFF salvato con successo: {output_path} ({file_size_mb:.1f} MB) con MPP={mpp:.4f} µm/pixel.")
        return output_path
        
    except Exception as e:
        logging.error(f"Errore durante il salvataggio BigTIFF con compressione JPEG a {output_path}: {e}")
        logging.warning("Tentativo di salvare senza compressione come fallback.")
        try:
            tifffile.imwrite(
                output_path,
                processed_image,
                bigtiff=True,
                compression=None, # Nessuna compressione
                photometric='rgb',
                planarconfig='contig',
                tile=(512, 512),
                resolution=(resolution_value, resolution_value), # Risoluzione in pixel (senza unità)
                resolutionunit='cm',   # Unità di risoluzione specificata separatamente
                metadata={'Software': 'Python tifffile', 'MPP_microns_per_pixel': mpp, 'Source': 'AnnData_uns_extract'}
            )
            file_size_mb_fallback = os.path.getsize(output_path) / (1024 * 1024)
            logging.info(f"BigTIFF salvato SENZA compressione: {output_path} ({file_size_mb_fallback:.1f} MB)")
            return output_path
        except Exception as e2:
            logging.error(f"Errore con fallback (nessuna compressione) per {output_path}: {e2}")
            return None

def extract_spatial_data_from_adata(adata: sc.AnnData) -> dict:
    """
    Estrae i dati spaziali e le probabilità dei tessuti da un oggetto AnnData.
    
    Args:
        adata (sc.AnnData): L'oggetto AnnData da cui estrarre i dati.
        
    Returns:
        dict: Dizionario contenente coordinate e probabilità dei tessuti.
    """
    logging.info("Estrazione delle coordinate spaziali e delle probabilità dei tessuti.")
    
    # Estrai coordinate spaziali (preferisci le coordinate in pixel se disponibili)
    if 'x_pixel' in adata.obs.columns and 'y_pixel' in adata.obs.columns:
        coords = adata.obs[['x_pixel', 'y_pixel']].values.astype(np.float32)
        logging.info("Utilizzo di 'x_pixel' e 'y_pixel' da adata.obs per le coordinate.")
    elif 'spatial' in adata.obsm:
        coords = adata.obsm['spatial'].astype(np.float32)
        logging.info("Utilizzo di 'spatial' da adata.obsm per le coordinate.")
    else:
        logging.error("Nessuna coordinata spaziale trovata in adata.obs (x_pixel, y_pixel) o adata.obsm['spatial']. Restituzione di coordinate vuote.")
        coords = np.array([], dtype=np.float32).reshape(0, 2)
    
    # Estrai probabilità dei tessuti
    tissue_probs = {}
    prob_mapping = {
        'tumor': 'Tumor Probability',
        'stroma': 'Stroma Probability', 
        'blood_necrosis': 'Blood and necrosis Probability'
    }
    
    for key, col_name in prob_mapping.items():
        if col_name in adata.obs.columns:
            tissue_probs[key] = adata.obs[col_name].values.astype(np.float32)
            logging.info(f"Trovata probabilità del tessuto: '{col_name}' per la chiave '{key}'.")
        else:
            logging.warning(f"Colonna '{col_name}' non trovata nelle colonne di adata.obs. Impostazione delle probabilità per '{key}' a zero.")
            tissue_probs[key] = np.zeros(len(coords), dtype=np.float32)
    
    spatial_data = {
        'coords': coords,
        'tissue_probs': tissue_probs,
        'shape': coords.shape,
        'coord_range': {
            'x_min': coords[:, 0].min() if coords.size > 0 else 0,
            'x_max': coords[:, 0].max() if coords.size > 0 else 0,
            'y_min': coords[:, 1].min() if coords.size > 0 else 0,
            'y_max': coords[:, 1].max() if coords.size > 0 else 0
        }
    }
    
    logging.info("Estrazione dei dati spaziali completata.")
    return spatial_data


def process_h5ad_file(
    file_h5ad_path: str,
    output_raw_dir: str | None = None,
    slide_key: str = '20x_slide',
    default_mpp: float = 0.40625, # Nuovo default basato sul calcolo dal tuo esempio
    config: dict | None = None,
    verbose: bool = False
) -> tuple[str | None, str | None]:
    """
    Estrae la WSI e i dati spaziali da un file h5ad, salvandoli come TIFF e pickle.
    Legge il file h5ad una sola volta e tenta di estrarre il MPP automaticamente.

    Args:
        file_h5ad_path (str): Path al file h5ad di input.
        output_raw_dir (str, optional): Directory di output per i file estratti.
                                        Se None, usa config['interim_data_dir'] o un default.
        slide_key (str): Chiave nel campo `uns` dove è salvata l'immagine WSI.
        default_mpp (float): Valore MPP da usare se non può essere determinato automaticamente dal file.
        config (dict, optional): Dizionario di configurazione con i path.
        verbose (bool): Se True, abilita logging più dettagliato nelle funzioni di salvataggio.

    Returns:
        tuple: (raw_wsi_path, spatial_data_path) - Path del file TIFF e dei dati spatial salvati.
                                                  Possono essere None se l'estrazione o il salvataggio falliscono.
    
    Raises:
        RuntimeError: Se l'estrazione/salvataggio WSI o spatial data fallisce in modo critico.
    """
    # Determina la directory di output
    if output_raw_dir is None:
        if config is not None and 'interim_data_dir' in config:
            output_raw_dir = os.path.join(config['interim_data_dir'], 'raw_wsi')
        else:
            output_raw_dir = 'data/interim/raw_wsi' # Fallback
    
    os.makedirs(output_raw_dir, exist_ok=True)
    logging.info(f"Directory di output impostata su: {output_raw_dir}")

    filename_base = os.path.splitext(os.path.basename(file_h5ad_path))[0]
    if filename_base.endswith('.h5ad'): # Per gestire nomi come 'file.h5ad.gz'
        filename_base = os.path.splitext(filename_base)[0]

    raw_wsi_output_path = os.path.join(output_raw_dir, filename_base + '.tiff')
    spatial_data_output_path = os.path.join(output_raw_dir, filename_base + '_spatial.pkl')

    raw_wsi_saved_path = None
    spatial_data_saved_path = None

    try:
        # Leggi AnnData e estrai tutti i dati, incluso il MPP calcolato
        adata, wsi_data, spatial_data, calculated_mpp = extract_wsi_and_spatial_from_h5ad(file_h5ad_path, slide_key=slide_key)

        # Usa il MPP calcolato, altrimenti il default
        mpp_to_use = calculated_mpp if calculated_mpp is not None else default_mpp
        if calculated_mpp is None:
            logging.warning(f"Impossibile determinare il MPP automaticamente. Utilizzo del MPP predefinito: {mpp_to_use:.4f} µm/pixel. Assicurarsi che questo sia corretto.")

        # --- Salva WSI se disponibile e non esiste già ---
        if wsi_data is not None:
            if not os.path.exists(raw_wsi_output_path):
                logging.info(f"Salvataggio WSI su {raw_wsi_output_path} con MPP {mpp_to_use:.4f}...")
                raw_wsi_saved_path = save_wsi(wsi_data, raw_wsi_output_path, mpp=mpp_to_use, verbose=verbose)
                if raw_wsi_saved_path is None:
                    logging.error(f"Salvataggio WSI per {filename_base} fallito. Nessun file TIFF sarà disponibile.")
            else:
                logging.info(f"Il file WSI esiste già: {raw_wsi_output_path}. Saltando il salvataggio.")
                raw_wsi_saved_path = raw_wsi_output_path
        else:
            logging.warning(f"Nessun dato immagine WSI trovato o estratto correttamente per {filename_base}. Saltando il salvataggio WSI.")

        # --- Salva dati spaziali se non esistono già ---
        if spatial_data is not None and spatial_data['coords'].size > 0:
            if not os.path.exists(spatial_data_output_path):
                logging.info(f"Salvataggio dei dati spaziali su {spatial_data_output_path}...")
                try:
                    with open(spatial_data_output_path, 'wb') as f:
                        pickle.dump(spatial_data, f)
                    spatial_data_saved_path = spatial_data_output_path
                    logging.info(f"Dati spaziali salvati: {spatial_data_saved_path}")
                except Exception as e:
                    logging.error(f"Salvataggio dei dati spaziali per {filename_base} fallito: {e}")
                    spatial_data_saved_path = None
            else:
                logging.info(f"Il file dei dati spaziali esiste già: {spatial_data_output_path}. Saltando il salvataggio.")
                spatial_data_saved_path = spatial_data_output_path
        else:
            logging.warning(f"Nessun dato spaziale trovato o estratto per {filename_base}. Saltando il salvataggio dei dati spaziali.")

    except Exception as e:
        logging.critical(f"Errore critico durante l'elaborazione di {file_h5ad_path}: {e}", exc_info=True)
        raise RuntimeError(f"Elaborazione fallita per {filename_base}: {str(e)}")

    return raw_wsi_saved_path, spatial_data_saved_path