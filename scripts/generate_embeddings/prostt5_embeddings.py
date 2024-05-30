import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["HF_HOME"] = '/workspace/data/transformers_cache/'

import torch
import numpy as np
import random
import re

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from tqdm import tqdm
from deli import load, save_json, save
from transformers import T5Tokenizer, T5EncoderModel


def embed_dataset(model, sequences, tokenizer, device):
    mean_embeddings = []
    # max_embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, 
                                              padding=True, is_split_into_words=True, 
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[1:]

            mean_embeddings.append(embedding.mean(axis=0))
            # max_embeddings.append(embedding.max(axis=0))
    return np.vstack(mean_embeddings)#, np.vstack(max_embeddings)

def embed_dataset_residue(model, sequences, tokenizer, device):
    mean_embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, 
                                              padding=True, is_split_into_words=True, 
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[1:-1]

            mean_embeddings.append(embedding)
    return np.vstack(mean_embeddings)

def main(dataset_sequence_path, save_emb_path, emb_type):
    os.makedirs(save_emb_path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)#.to(device)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device.type != 'cpu':
        model.half()
    model.eval()
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
        # replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", ''.join(seq)))) for seq in all_sequences[split]]
        
        # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
        # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        sequences = ["<AA2fold>" + " " + s for s in sequences]

        if emb_type == 'protein':
            mean_embeddings = embed_dataset(model, sequences, 
                                    tokenizer=tokenizer, device=device)
            names = all_names[split]
        else:
            mean_embeddings = embed_dataset_residue(model, sequences, 
                                    tokenizer=tokenizer, device=device)
            names = [f'{name}_{i}' for name, seq in zip(all_names[split], all_sequences[split]) 
                     for i in range(len(seq))]

        print(mean_embeddings.shape)
        print('names', len(names))
        save_json(names, os.path.join(save_emb_path, f'{split}_names.json'))
        save(mean_embeddings, os.path.join(save_emb_path, f'{split}_avg_embeddings.npy.gz'), compression=1)


if __name__ == "__main__":
    dataset_sequence_path = '/workspace/data/docking/downstream_tasks/thermostability/id2seq_real.json'
    save_emb_path = '/workspace/data/docking/downstream_tasks/downstream_datasets/thermostability/prostt5'
    main(dataset_sequence_path, save_emb_path)
