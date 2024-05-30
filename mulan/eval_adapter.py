#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HF_HOME"] = '/workspace/data/docking/transformers_cache'
import logging
from argparse import ArgumentParser
import random

import torch
from transformers import AutoTokenizer
import yaml
import numpy as np


from mulan.model import StructEsmForMaskedLM
from mulan.downstream_task_embeddings import evaluate_downstream_task

LOGGER = logging.getLogger(__name__)


# Function to load yaml configuration file
def load_config(config_fname):
    with open(config_fname) as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file")
    parser.add_argument("--model", dest="model_path",
                        required=False, default=None, help="device no")
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

    dataset_type = config["dataset_type"]
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
        
    if config["calc_thermostability"]:
        LOGGER.info('Computing embeddings for Thermostability')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_thermostability_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_thermostability_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=True,
            use_foldseek_sequences=use_foldseek_sequences,
        )

    if config["calc_go"]:
        LOGGER.info('Computing embeddings for GO')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_go_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_go_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=True,
            use_foldseek_sequences=use_foldseek_sequences,
        )

    if config["calc_metal"]:
        LOGGER.info('Computing embeddings for Metal ion binding')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_metal_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_metal_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=True,
            use_foldseek_sequences=use_foldseek_sequences,
        )

    if config["calc_fluorescence"]:
        LOGGER.info('Computing embeddings for Fluorescence')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_fluorescence_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_fluorescence_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=True,
            use_foldseek_sequences=use_foldseek_sequences,
        )

    if config["calc_ss_pdb"]:
        LOGGER.info('Computing embeddings for secondary structure (PDB)')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_ss_pdb_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_ss_pdb_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=False,
            use_foldseek_sequences=use_foldseek_sequences,
            dataset_names=['casp12', 'ts115', 'cb513', 'valid', 'train'],
            is_experimental_structure=True,
        )

    if config["calc_humanppi"]:
        LOGGER.info('Computing embeddings for Human protein-protein interaction')
        evaluate_downstream_task(
            model=model, 
            esm_tokenizer=esm_tokenizer, 
            saved_dataset_path=os.path.join(config["saved_humanppi_dataset_path"], dataset_type),
            downstream_dataset_path=os.path.join(config["downstream_humanppi_dataset_path"], exp_name),
            batch_limit=batch_limit,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            protein_level=True,
            use_foldseek_sequences=use_foldseek_sequences,
        )
