import datasets
import pandas as pd

local_dir='/equilibrium/datasets/TCGA-histological-data/' # hest will be dowloaded to this folder

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

# Filter the dataframe by organ, oncotree code...
meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
meta_df = meta_df[meta_df['organ'] == 'Breast']

ids_to_query = meta_df['id'].values

list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
dataset = datasets.load_dataset(
    'MahmoodLab/hest', 
    cache_dir=local_dir,
    patterns=list_patterns
)