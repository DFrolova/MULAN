import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import gc
import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# import esm
from tqdm.auto import tqdm
from deli import load, save_json, save
from transformers import AutoTokenizer, AutoModelForMaskedLM


def get_tokens(seqs, tokenizer):
    tokens = tokenizer.batch_encode_plus(seqs)['input_ids']
    return tokens


def get_embeddings(tokens, esm_model, device):    
    embeddings = []
    # protein_embeddings = []
    # max_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(tokens)):
            if not i%1000 and i!=0:
                # print(f"{i+1} embeddings were obtained")
                torch.cuda.empty_cache()
                gc.collect()
            batch = torch.tensor(batch).to(device)
            batch = batch[None, :]
            res = esm_model(batch, output_hidden_states=True)['hidden_states'][-1]
            embeddings.extend(res[:, 1:-1].mean(dim=1).cpu())
            # max_embeddings.extend(res[:, 1:-1].max(dim=1).values.cpu())
            # protein_embeddings.append(res[0, 0].cpu())
                
    embeddings = torch.stack(embeddings).numpy()
    # max_embeddings = torch.stack(max_embeddings).numpy()
    # protein_embeddings = torch.stack(protein_embeddings).numpy()
    return embeddings#, protein_embeddings, max_embeddings


def get_embeddings_residue(tokens, esm_model, device):    
    embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(tokens)):
            if not i%1000 and i!=0:
                torch.cuda.empty_cache()
                gc.collect()
            batch = torch.tensor(batch).to(device)
            batch = batch[None, :]
            res = esm_model(batch, output_hidden_states=True)['hidden_states'][-1]
            embeddings.extend(res[:, 1:-1].cpu())
                
    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings


def main(dataset_sequence_path, save_emb_path, emb_type):
    os.makedirs(save_emb_path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    if os.path.basename(save_emb_path) == 'hf_esm_6':
        model_checkpoint = 'facebook/esm2_t6_8M_UR50D'
    elif os.path.basename(save_emb_path) == 'hf_esm_12':
        model_checkpoint = 'facebook/esm2_t12_35M_UR50D'
    elif os.path.basename(save_emb_path) == 'hf_esm_33':
        model_checkpoint = 'facebook/esm2_t33_650M_UR50D'
    else:
        print(f'Model {os.path.basename(save_emb_path)} not found')

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) 
    model.eval()
    model.to(device=device);

    print('Model loaded')

    num_params_trainable = 0
    num_params_all = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params_trainable += int(torch.prod(torch.tensor(param.data.shape)))
        num_params_all += int(torch.prod(torch.tensor(param.data.shape)))
    print('Trainable parameters:', num_params_trainable)
    print('All parameters:', num_params_all)

    all_data = load(dataset_sequence_path)

    all_names = {}
    all_sequences = {}
    for split in all_data.keys():
        all_names[split] = list(all_data[split].keys())
        all_sequences[split] = [list(seq) for seq in list(all_data[split].values())]

    print('Sequences loaded')

    for split in all_data.keys():
        prepared_sequences = [''.join(seq) for seq in all_sequences[split]]
        tokens = get_tokens(prepared_sequences, tokenizer)
        if emb_type == 'protein':
            embeddings = get_embeddings(tokens=tokens,
                                        esm_model=model, 
                                        device=device)
            names = all_names[split]
        else:
            embeddings = get_embeddings_residue(tokens=tokens,
                                        esm_model=model, 
                                        device=device)
            names = [f'{name}_{i}' for name, seq in zip(all_names[split], prepared_sequences) 
                     for i in range(len(seq))]
        
        print(type(embeddings), embeddings.shape)
        print('names', len(names))
        save_json(names, os.path.join(save_emb_path, f'{split}_names.json'))
        save(embeddings, os.path.join(save_emb_path, f'{split}_avg_embeddings.npy.gz'), compression=1)


if __name__ == "__main__":
    dataset_sequence_path = '/workspace/data/docking/downstream_tasks/localization_deeploc/id2seq_real.json'
    save_emb_path = '/workspace/data/docking/downstream_tasks/downstream_datasets/localization_deeploc/hf_esm_12'
    main(dataset_sequence_path, save_emb_path)
