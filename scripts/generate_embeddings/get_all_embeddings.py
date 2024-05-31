import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import ankh_embeddings
# import esm2_embeddings
import hf_esm2_embeddings
import prostt5_embeddings
import saprot_embeddings


if __name__ == "__main__":
    # task_name = 'thermostability'
    # task_name = 'go'
    task_name = 'metal'
    # task_name = 'humanppi'
    # task_name = 'fluorescence'
    # task_name = 'secondary_structure_pdb'

    emb_type = 'protein'
    # emb_type = 'residue'

    dataset_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_real.json'
    # foldseek_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_foldseek.json'
    # foldseek_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_foldseek_fixed.json'
    foldseek_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_foldseek_masked.json'

    save_emb_path_base = f'/workspace/data/docking/downstream_tasks/downstream_datasets/{task_name}/'
    # dataset_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_real_nopairs.json'
    # foldseek_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_foldseek_nopairs.json'
    # foldseek_sequence_path = f'/workspace/data/docking/downstream_tasks/{task_name}/id2seq_foldseek_masked_nopair.json'

    # hf_esm2_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'hf_esm_6'), emb_type=emb_type)
    # hf_esm2_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'hf_esm_12'), emb_type=emb_type)
    # hf_esm2_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'hf_esm_33'), emb_type=emb_type)
    # ankh_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'ankh_base'), emb_type=emb_type)
    # ankh_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'ankh_large'), emb_type=emb_type)
    # esm2_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'esm_6'), emb_type=emb_type)
    # prostt5_embeddings.main(dataset_sequence_path, os.path.join(save_emb_path_base, 'prostt5'), emb_type=emb_type)
    saprot_embeddings.main(dataset_sequence_path, foldseek_sequence_path, 
                           os.path.join(save_emb_path_base, 'saprot_pdb'), emb_type=emb_type)
    saprot_embeddings.main(dataset_sequence_path, foldseek_sequence_path, 
                           os.path.join(save_emb_path_base, 'saprot_af'), emb_type=emb_type)
    saprot_embeddings.main(dataset_sequence_path, foldseek_sequence_path, 
                           os.path.join(save_emb_path_base, 'saprot_12_af'), emb_type=emb_type)
    saprot_embeddings.main(dataset_sequence_path, foldseek_sequence_path, 
                           os.path.join(save_emb_path_base, 'saprot_12_af_seq'), emb_type=emb_type)
