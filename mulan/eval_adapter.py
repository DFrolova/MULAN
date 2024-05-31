#!/usr/bin/env python
# coding: utf-8

import numpy as np
from transformers import AutoTokenizer
import torch
import random
from argparse import ArgumentParser
import logging
import os
os.environ["HF_HOME"] = '/workspace/data/docking/transformers_cache'

from mulan.downstream_task_embeddings import evaluate_downstream_task
from mulan.model import StructEsmForMaskedLM
from mulan.utils import load_config


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

    struct_data_dim = 7
    use_struct_embeddings = config["use_struct_embeddings"]
    num_struct_embeddings_layers = config["num_struct_embeddings_layers"]
    mask_angle_inputs_with_plddt = config["mask_angle_inputs_with_plddt"]
    predict_contacts = config['predict_contacts']
    predict_angles = False
    batch_limit = config["batch_limit"]

    adapter_path = config["trained_adapter_name"]

    LOGGER.info(f'Initializing model...')
    results_dir = os.path.join(results_folder, config['trained_adapter_name'])
    print('results_dir', results_dir)
    # checkpoint_folder = sorted(os.listdir(results_dir))[-1]
    # checkpoint_path = os.path.join(results_dir, checkpoint_folder)
    checkpoint_path = results_dir

    model = StructEsmForMaskedLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        num_struct_embeddings_layers=num_struct_embeddings_layers,
        struct_data_dim=struct_data_dim,
        use_struct_embeddings=use_struct_embeddings,
        predict_contacts=predict_contacts,
        predict_angles=predict_angles,
        mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    )
    print(type(model))
    print('use_struct_embeddings', use_struct_embeddings)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    esm_tokenizer = AutoTokenizer.from_pretrained(config["esm_checkpoint"])
    exp_name = config['trained_adapter_name']

    print(type(model))

    all_experiments = {
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
            dataset_names=['casp12', 'ts115', 'cb513', 'valid', 'train'],
            is_experimental_structure=True,
        )

    # if config["calc_thermostability"]:
    #     LOGGER.info('Computing embeddings for Thermostability')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_thermostability_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_thermostability_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=True,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #     )

    # if config["calc_go"]:
    #     LOGGER.info('Computing embeddings for GO')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_go_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_go_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=True,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #     )

    # if config["calc_metal"]:
    #     LOGGER.info('Computing embeddings for Metal ion binding')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_metal_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_metal_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=True,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #     )

    # if config["calc_fluorescence"]:
    #     LOGGER.info('Computing embeddings for Fluorescence')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_fluorescence_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_fluorescence_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=True,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #     )

    # if config["calc_ss_pdb"]:
    #     LOGGER.info('Computing embeddings for secondary structure (PDB)')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_ss_pdb_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_ss_pdb_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=False,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #         dataset_names=['casp12', 'ts115', 'cb513', 'valid', 'train'],
    #         is_experimental_structure=True,
    #     )

    # if config["calc_humanppi"]:
    #     LOGGER.info(
    #         'Computing embeddings for Human protein-protein interaction')
    #     evaluate_downstream_task(
    #         model=model,
    #         esm_tokenizer=esm_tokenizer,
    #         saved_dataset_path=config["saved_humanppi_dataset_path"],
    #         downstream_dataset_path=os.path.join(
    #             config["downstream_humanppi_dataset_path"], exp_name),
    #         batch_limit=batch_limit,
    #         mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
    #         protein_level=True,
    #         use_foldseek_sequences=use_foldseek_sequences,
    #     )
