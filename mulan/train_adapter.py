#!/usr/bin/env python
# coding: utf-8

import os
import logging
from datetime import datetime
from argparse import ArgumentParser
import warnings

import torch
from transformers import (
    TrainingArguments, 
    AutoTokenizer,
)
import wandb
import numpy as np

from mulan.dataset import ProteinDataset, data_collate_fn_dynamic
from mulan.trainer import MulanTrainer
from mulan.metrics import (
    compute_eval_metrics, 
    preprocess_logits_for_metrics,
)
from mulan.model import StructEsmForMaskedLM
from mulan.utils import load_config, get_foldseek_tokenizer

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file")
    parser.add_argument("-n", "--exp-name", dest="exp_name", required=True,
                        help="exp name for results path")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # folder to load config file
    config = load_config(args.config_filename)
    print(config)

    results_folder = config["results_folder"]
    protein_data_path = config["protein_data_path"]
    saved_dataset_path_AFDB = config["saved_dataset_path_AFDB"]
    split_ids_file = None #'/workspace/data/dfrolova/AFDB/dataset_tmp_split_ids.json' #None
    use_foldseek_sequences = config["use_foldseek_sequences"]
    add_foldseek_embeddings = config["add_foldseek_embeddings"]
    print('add_foldseek_embeddings', add_foldseek_embeddings)

    min_protein_length = config["min_protein_length"]
    max_protein_length = config["max_protein_length"]
    struct_data_dim = 7
    use_struct_embeddings = config["use_struct_embeddings"]
    num_struct_embeddings_layers = config["num_struct_embeddings_layers"]
    mask_angle_inputs_with_plddt = config["mask_angle_inputs_with_plddt"]
    predict_contacts = config["predict_contacts"]
    predict_angles = config["predict_angles"]
    use_sorted_batching = config["use_sorted_batching"]
    batch_limit = config["batch_limit"]


    adapter_path = config["trained_adapter_name"]
    base_checkpoint = config["esm_checkpoint"]

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H:%M:%S")
    exp_name = args.exp_name
    exp_name_full = f'{exp_name}_{date_time}'
    results_dir = os.path.join(results_folder, exp_name_full)

    if config["trained_adapter_name"] == 'None':
        results_dir2 = os.path.join(results_folder, exp_name_full)
    else:
        results_dir2 = os.path.join(results_folder, config["trained_adapter_name"])

    LOGGER.info(f'Initializing datasets...')

    real_saved_dataset_path = saved_dataset_path_AFDB
    is_experimental_structure = False

    train_dataset = ProteinDataset(
        protein_data_path=protein_data_path, 
        saved_dataset_path=real_saved_dataset_path,
        split_ids_file=split_ids_file,
        split=config["train_split"], 
        min_protein_length=min_protein_length, 
        max_protein_length=max_protein_length,
        use_sorted_batching=use_sorted_batching,
        batch_limit=batch_limit,
        predict_contacts=predict_contacts,
        use_foldseek_sequences=use_foldseek_sequences,
        is_experimental_structure=is_experimental_structure,
        add_foldseek_embeddings=add_foldseek_embeddings,
    )
    eval_dataset = ProteinDataset(
        protein_data_path=protein_data_path, 
        saved_dataset_path=real_saved_dataset_path,
        split_ids_file=split_ids_file,
        split='val', 
        min_protein_length=min_protein_length, 
        max_protein_length=max_protein_length,
        use_sorted_batching=use_sorted_batching,
        batch_limit=batch_limit,
        predict_contacts=predict_contacts,
        use_foldseek_sequences=use_foldseek_sequences,
        is_experimental_structure=is_experimental_structure,
        add_foldseek_embeddings=add_foldseek_embeddings,
    )

    if config["esm_checkpoint"] == 'None':
        esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    else:
        esm_tokenizer = AutoTokenizer.from_pretrained(config["esm_checkpoint"])

    fs_tokenizer = None
    if add_foldseek_embeddings:
        use_foldseek_sequences = False
        print('Warning: set use_foldseek_sequences = False because add_foldseek_embeddings is True') 
        fs_tokenizer = get_foldseek_tokenizer()

    if use_foldseek_sequences:
        all_amino_acids = esm_tokenizer.all_tokens[5:]
    else:
        all_amino_acids = train_dataset.tokenizer.one_letter_aas

    def data_collator(x): 
        return data_collate_fn_dynamic(
            x, esm_tokenizer=esm_tokenizer,
            fs_tokenizer=fs_tokenizer,
            nan_value=np.deg2rad(train_dataset.tokenizer.nan_fill_value),
            predict_contacts=predict_contacts,
            max_prot_len=1022,
            all_amino_acids=all_amino_acids,
            mlm_probability=0.15, 
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
            use_foldseek_sequences=use_foldseek_sequences,
            mask_inputs=True,
        )

    LOGGER.info(f'Initializing model...')
    print('Loading from', base_checkpoint)
    model = StructEsmForMaskedLM.from_pretrained(
        base_checkpoint,
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

    num_params_trainable = 0
    num_params_all = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params_trainable += int(torch.prod(torch.tensor(param.data.shape)))
        num_params_all += int(torch.prod(torch.tensor(param.data.shape)))
    print('Trainable parameters:', num_params_trainable)
    print('All parameters:', num_params_all)

    training_args = TrainingArguments(
        output_dir=results_dir,
        overwrite_output_dir=False,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],

        save_steps=7500,
        save_strategy="steps",
        save_total_limit=11,

        prediction_loss_only=False,
        report_to='wandb',
        run_name=args.exp_name,
        logging_steps=config["logging_steps"],
        evaluation_strategy="steps",
        do_eval=True,
        eval_steps=config["eval_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        learning_rate=config["learning_rate"],
        label_names=["labels"],
        dataloader_num_workers=30,
    )

    all_parameters = (param for name, param in model.named_parameters() if 'struct_embeddings' not in name)

    scheduler = None
    optimizer = None
    compute_metrics = compute_eval_metrics
    preprocess_logits_for_metrics = preprocess_logits_for_metrics

    trainer = MulanTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, scheduler),
        lr_decrease_ratio=config["lr_decrease_ratio"],
    )

    LOGGER.info(f'Start training...')
    trainer.train()
