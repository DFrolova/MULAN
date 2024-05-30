import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import gc
import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import esm
from tqdm.auto import tqdm
from deli import load, save_json, save


def get_tokens(seqs, alphabet):
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter(seqs)
    return tokens


def get_embeddings(tokens, esm_model, layer, device):    
    embeddings = []
    # protein_embeddings = []
    # max_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(tokens)):
            if not i%1000 and i!=0:
                # print(f"{i+1} embeddings were obtained")
                torch.cuda.empty_cache()
                gc.collect()
            batch = batch.to(device)
            batch = batch[batch != 1][None, :]
            res = esm_model(batch, repr_layers=[layer])["representations"][layer]
                        
            embeddings.extend(res[:, 1:-1].mean(dim=1).cpu())
            # max_embeddings.extend(res[:, 1:-1].max(dim=1).values.cpu())
            # protein_embeddings.append(res[0, 0].cpu())
    
    embeddings = torch.stack(embeddings).numpy()
    # max_embeddings = torch.stack(max_embeddings).numpy()
    # protein_embeddings = torch.stack(protein_embeddings).numpy()
    return embeddings#, protein_embeddings, max_embeddings


def main(dataset_sequence_path, save_emb_path):
    os.makedirs(save_emb_path, exist_ok=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    if os.path.basename(save_emb_path) == 'esm_33':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        layer = 33
    elif os.path.basename(save_emb_path) == 'esm_12':
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        layer = 12
    elif os.path.basename(save_emb_path) == 'esm_6':
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        layer = 6
    else:
        print(f'Model {os.path.basename(save_emb_path)} not found')
        
    model.eval()
    model.to(device=device);

    print('Model loaded')

    all_data = load(dataset_sequence_path)

    all_names = {}
    all_sequences = {}
    for split in all_data.keys():
        all_names[split] = list(all_data[split].keys())
        all_sequences[split] = [list(seq) for seq in list(all_data[split].values())]

    print('Sequences loaded')

    for split in all_data.keys():
        prepared_sequences = [('_', ''.join(seq)) for seq in all_sequences[split]]
        tokens = get_tokens(prepared_sequences, alphabet)
        embeddings = get_embeddings(tokens=tokens,
                                    esm_model=model, 
                                    layer=layer,
                                    device=device)

        print(type(embeddings), embeddings.shape)
        # print(type(max_embeddings), max_embeddings.shape)
        # print(type(protein_embeddings), protein_embeddings.shape)
        save_json(all_names[split], os.path.join(save_emb_path, f'{split}_names.json'))
        save(embeddings, os.path.join(save_emb_path, f'{split}_avg_embeddings.npy.gz'), compression=1)
        # save(max_embeddings, os.path.join(save_emb_path, f'{split}_max_embeddings.npy.gz'), compression=1)
        # save(protein_embeddings, os.path.join(save_emb_path, f'{split}_protein_embeddings.npy.gz'), compression=1)


if __name__ == "__main__":
    dataset_sequence_path = '/workspace/data/docking/downstream_tasks/thermostability/id2seq_real.json'
    save_emb_path = '/workspace/data/docking/downstream_tasks/downstream_datasets/thermostability/esm_12'
    main(dataset_sequence_path, save_emb_path)
