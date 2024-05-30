from tqdm.auto import tqdm
import os
from mulan.pdb_utils import AnglesFromStructure, getStructureObject


afdb_path = '<path to folder with protein structures>' # path with all pdb structures
save_angles_path = '<path to where to save angles>' # path where all data files with angles for each structure are stored


all_ids = sorted(os.listdir(afdb_path))

for pdb_id in tqdm(all_ids):
    pdb_path = os.path.join(afdb_path, pdb_id)
    pdb_id = '.'.join(pdb_id.split('.')[:-1])

    if os.path.exists(os.path.join(save_angles_path, f'{pdb_id}.tsv')):
        continue

    df_name = os.path.join(save_angles_path, f'{pdb_id}.tsv')

    angles_init = AnglesFromStructure(
        getStructureObject(pdb_path, chain='A'))
    angles_init.to_csv(df_name, sep='\t', index=False)
    