import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from tqdm import tqdm
from deli import load, save_json, save
from transformers import EsmTokenizer, EsmForMaskedLM


def embed_dataset(model, sequences, tokenizer, device):
    mean_embeddings = []
    # max_embeddings = []
    # protein_embeddings = []
    with torch.no_grad():
        for seq in tqdm(sequences):
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embedding = model(**inputs, output_hidden_states=True)['hidden_states'][-1]
            embedding = embedding[0].detach().cpu().numpy()
                    
            mean_embeddings.append(embedding[1:-1].mean(axis=0))
            # max_embeddings.append(embedding[1:-1].max(axis=0))
            # protein_embeddings.append(embedding[0])
    return np.vstack(mean_embeddings)#, np.vstack(max_embeddings), np.vstack(protein_embeddings)

def embed_dataset_residue(model, sequences, tokenizer, device):
    mean_embeddings = []
    with torch.no_grad():
        for seq in tqdm(sequences):
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embedding = model(**inputs, output_hidden_states=True)['hidden_states'][-1]
            embedding = embedding[0].detach().cpu().numpy()
                    
            mean_embeddings.append(embedding[1:-1])
    return np.vstack(mean_embeddings)


def main(dataset_sequence_path, foldseek_sequence_path, save_emb_path, emb_type):
    os.makedirs(save_emb_path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    if os.path.basename(save_emb_path) == 'saprot_pdb':
        model_path = "westlake-repl/SaProt_650M_PDB"
    elif os.path.basename(save_emb_path) == 'saprot_af':
        model_path = "westlake-repl/SaProt_650M_AF2"
    elif os.path.basename(save_emb_path) == 'saprot_12_af':
        model_path = "westlake-repl/SaProt_35M_AF2"
    elif os.path.basename(save_emb_path) == 'saprot_12_af_seq':
        model_path = "westlake-repl/SaProt_35M_AF2_seqOnly"
        foldseek_sequence_path = dataset_sequence_path
    else:
        print(f'Model {os.path.basename(save_emb_path)} not found')

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    model.eval()
    model.to(device)
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
    foldseek_data = load(foldseek_sequence_path)

    all_names = {}
    all_sequences = {}
    for split in all_data.keys():
        all_names[split] = list(all_data[split].keys())

        if model_path == "westlake-repl/SaProt_35M_AF2_seqOnly":
            all_sequences[split] = [list('#'.join(list(foldseek_data[split][name])) + '#') for name in all_names[split]]
        else:
            all_sequences[split] = [list(foldseek_data[split][name]) for name in all_names[split]]


    print('Sequences loaded')

    for split in all_data.keys():
        sequences = [''.join(seq) for seq in all_sequences[split]]

        if emb_type == 'protein':
            mean_embeddings = embed_dataset(model, sequences, 
                                            tokenizer=tokenizer, 
                                            device=device)
            names = all_names[split]
        else:
            mean_embeddings = embed_dataset_residue(model, sequences, 
                                        tokenizer=tokenizer, 
                                        device=device)
            names = [f'{name}_{i}' for name, seq in zip(all_names[split], all_sequences[split]) 
                     for i in range(int(len(seq) / 2))]
            
        print(mean_embeddings.shape)
        print('names', len(names))
        save_json(names, os.path.join(save_emb_path, f'{split}_names.json'))
        save(mean_embeddings, os.path.join(save_emb_path, f'{split}_avg_embeddings.npy.gz'), compression=1)


if __name__ == "__main__":
    dataset_sequence_path = '/workspace/data/docking/downstream_tasks/thermostability/id2seq_real.json'
    foldseek_sequence_path = '/workspace/data/docking/downstream_tasks/thermostability/id2seq_foldseek.json'
    save_emb_path = '/workspace/data/docking/downstream_tasks/downstream_datasets/thermostability/saprot_af'
    main(dataset_sequence_path, foldseek_sequence_path, save_emb_path)
