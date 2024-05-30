import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import ankh
from tqdm import tqdm
from deli import load, save_json, save



def embed_dataset(model, sequences, tokenizer, device, max_len=7000):
    mean_embeddings = []
    # max_embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            cur_sample = sample
            cur_mean_embeddings = []
            # cur_max_embeddings = []

            while True:
                ids = tokenizer.batch_encode_plus([cur_sample[:max_len]], add_special_tokens=True, 
                                                  padding=True, is_split_into_words=True, 
                                                  return_tensors="pt")
                embedding = model(input_ids=ids['input_ids'].to(device))[0]
                
                embedding = embedding[0].detach().cpu().numpy()[:-1]
                    
                cur_sample = cur_sample[max_len:]
                cur_mean_embeddings.append(embedding.mean(axis=0))
                # cur_max_embeddings.append(embedding.max(axis=0))
                if len(cur_sample) == 0:
                    break
                    
            mean_embeddings.append(np.vstack(cur_mean_embeddings))
            # max_embeddings.append(np.vstack(cur_max_embeddings))
    return np.vstack(mean_embeddings)#, np.vstack(max_embeddings)


def embed_dataset_residue(model, sequences, tokenizer, device, max_len=7000):
    mean_embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            cur_sample = sample
            cur_mean_embeddings = []

            while True:
                ids = tokenizer.batch_encode_plus([cur_sample[:max_len]], add_special_tokens=True, 
                                                  padding=True, is_split_into_words=True, 
                                                  return_tensors="pt")
                embedding = model(input_ids=ids['input_ids'].to(device))[0]
                
                embedding = embedding[0].detach().cpu().numpy()[:-1]
                    
                cur_sample = cur_sample[max_len:]
                cur_mean_embeddings.append(embedding)
                if len(cur_sample) == 0:
                    break
                    
            mean_embeddings.append(np.vstack(cur_mean_embeddings))
    return np.vstack(mean_embeddings)

def main(dataset_sequence_path, save_emb_path, emb_type):
    os.makedirs(save_emb_path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    if os.path.basename(save_emb_path) == 'ankh_base':
        model, tokenizer = ankh.load_base_model()
    elif os.path.basename(save_emb_path) == 'ankh_large':
        model, tokenizer = ankh.load_large_model()
    else:
        print(f'Model {os.path.basename(save_emb_path)} not found')

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

        if emb_type == 'protein':
            mean_embeddings = embed_dataset(model, all_sequences[split], 
                                    tokenizer=tokenizer, device=device, max_len=7000)
            names = all_names[split]
        else:
            mean_embeddings = embed_dataset_residue(model, all_sequences[split], 
                                tokenizer=tokenizer, device=device, max_len=7000)
            names = [f'{name}_{i}' for name, seq in zip(all_names[split], all_sequences[split]) 
                     for i in range(len(seq))]
        
        print(mean_embeddings.shape)
        print('names', len(names))
        save_json(names, os.path.join(save_emb_path, f'{split}_names.json'))
        save(mean_embeddings, os.path.join(save_emb_path, f'{split}_avg_embeddings.npy.gz'), compression=1)


if __name__ == "__main__":
    dataset_sequence_path = '/workspace/data/docking/downstream_tasks/thermostability/id2seq_real.json'
    save_emb_path = '/workspace/data/docking/downstream_tasks/downstream_datasets/thermostability/ankh_base'
    main(dataset_sequence_path, save_emb_path)
