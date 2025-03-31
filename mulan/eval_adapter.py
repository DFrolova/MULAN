#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HF_HOME"] = '/workspace/data/docking/transformers_cache'

import numpy as np
from transformers import AutoTokenizer
import torch
import random
from argparse import ArgumentParser
import logging

from mulan.downstream_task_embeddings import evaluate_downstream_task
from mulan.model import StructEsmForMaskedLM
from mulan.utils import load_config, get_foldseek_tokenizer


LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file")
    args = parser.parse_args()

    # folder to load config file
    config = load_config(args.config_filename)

    results_folder = config["results_folder"]
    use_foldseek_sequences = config["use_foldseek_sequences"]
    add_foldseek_embeddings = config["add_foldseek_embeddings"]
    print('add_foldseek_embeddings', add_foldseek_embeddings)

    struct_data_dim = 7
    use_struct_embeddings = config["use_struct_embeddings"]
    num_struct_embeddings_layers = config["num_struct_embeddings_layers"]
    mask_angle_inputs_with_plddt = config["mask_angle_inputs_with_plddt"]
    predict_contacts = config['predict_contacts']
    predict_angles = False
    batch_limit = config["batch_limit"]

    adapter_path = config["trained_adapter_name"]

    LOGGER.info(f'Initializing model...')
    if results_folder == 'none':
        results_dir = config['trained_adapter_name']
    else:
        results_dir = os.path.join(results_folder, config['trained_adapter_name'])
    print('results_dir', results_dir)
    checkpoint_folder = sorted(os.listdir(results_dir))[-1]
    checkpoint_path = os.path.join(results_dir, checkpoint_folder)
    # checkpoint_path = config["esm_checkpoint"]

    esm_tokenizer = AutoTokenizer.from_pretrained(config["esm_checkpoint"])

    fs_tokenizer = None
    if add_foldseek_embeddings:
        use_foldseek_sequences = False
        print('Warning: set use_foldseek_sequences = False because add_foldseek_embeddings is True') 
        fs_tokenizer = get_foldseek_tokenizer()

    model = StructEsmForMaskedLM.from_pretrained(
        checkpoint_path,
        # device_map="auto",
        num_struct_embeddings_layers=num_struct_embeddings_layers,
        struct_data_dim=struct_data_dim,
        use_struct_embeddings=use_struct_embeddings,
        predict_contacts=predict_contacts,
        predict_angles=predict_angles,
        mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
        add_foldseek_embeddings=add_foldseek_embeddings,
        fs_tokenizer=fs_tokenizer,
    )
    print(model)
    print(type(model))
    print('use_struct_embeddings', use_struct_embeddings)
    print('checkpoint_path', checkpoint_path)

    num_params_struct = 0
    num_params_all = 0
    for name, param in model.named_parameters():
        if 'struct_embedding' in name:
            num_params_struct += int(torch.prod(torch.tensor(param.data.shape)))
        num_params_all += int(torch.prod(torch.tensor(param.data.shape)))
    print('Adapter parameters:', num_params_struct)
    print('All parameters:', num_params_all)
    print('Fraction:', num_params_struct / (num_params_all - num_params_struct))

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_name = config['trained_adapter_name']

    print(type(model))

    all_experiments = {
        'localization_deeploc': config["calc_localization"],
        'localization_deeploc_binary': config["calc_localization_binary"],
        'thermostability': config["calc_thermostability"],
        'fluorescence': config["calc_fluorescence"],
        'go': config["calc_go"],
        'metal': config["calc_metal"],
        'humanppi': config["calc_humanppi"],
    }

    for task_name, do_experiment in all_experiments.items():
        if do_experiment:
            LOGGER.info(f'Computing embeddings for {task_name}')
            embeddings_path = os.path.join(config["downstream_datasets_path"], task_name, 
                                           'embeddings', exp_name)
            os.makedirs(embeddings_path, exist_ok=True)
            evaluate_downstream_task(
                model=model,
                esm_tokenizer=esm_tokenizer,
                saved_dataset_path=os.path.join(config["downstream_datasets_path"], task_name, 'dataset'),
                downstream_dataset_path=embeddings_path,
                batch_limit=batch_limit,
                mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
                protein_level=True,
                use_foldseek_sequences=use_foldseek_sequences,
                add_foldseek_embeddings=add_foldseek_embeddings,
                fs_tokenizer=fs_tokenizer,
            )

    task_name = 'secondary_structure_pdb'
    if config["calc_ss_pdb"]:
        LOGGER.info('Computing embeddings for secondary structure (PDB)')
        embeddings_path = os.path.join(config["downstream_datasets_path"], task_name,
                                       'embeddings', exp_name)
        os.makedirs(embeddings_path, exist_ok=True)
        evaluate_downstream_task(
            model=model,
            esm_tokenizer=esm_tokenizer,
            saved_dataset_path=os.path.join(config["downstream_datasets_path"], task_name, 'dataset'),
            downstream_dataset_path=embeddings_path,
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=False,
            use_foldseek_sequences=use_foldseek_sequences,
            add_foldseek_embeddings=add_foldseek_embeddings,
            fs_tokenizer=fs_tokenizer,
            dataset_names=['casp12', 'ts115', 'cb513', 'valid', 'train'],
            is_experimental_structure=True,
        )
