import os
import torch
import numpy as np
from scipy import stats
import random
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore')
from pathlib import Path

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput

import sklearn
import seqeval.metrics as seq_metrics
import scipy
from functools import partial

from deli import load, save_json, save
from sklearn.preprocessing import StandardScaler

from mulan.utils import load_config



def load_prot_level_data(emb_path1, split, agg_type1):

    embeddings1 = load(os.path.join(emb_path1, f'{split}{agg_type1}_embeddings.npy.gz'))
    if embeddings1.dtype != np.float32:
        embeddings1 = embeddings1.astype(np.float32)

    names1 = load(os.path.join(emb_path1, f'{split}_names.json'))
    if type(names1[0]) == list:
        names1 = [prot_names[0] for prot_names in names1]

    print('Embeddings shape', embeddings1.shape)

    return names1, embeddings1


def load_humanppi_data(emb_path1, split, agg_type1, concat_names):

    embeddings1 = load(os.path.join(emb_path1, f'{split}{agg_type1}_embeddings.npy.gz'))
    if embeddings1.dtype != np.float32:
        embeddings1 = embeddings1.astype(np.float32)

    names1 = load(os.path.join(emb_path1, f'{split}_names.json'))
    if type(names1[0]) == list:
        names1 = [prot_names[0] for prot_names in names1]

    to_keep_names1 = [uid.split('_')[0] for uid in concat_names]
    to_keep_names2 = [uid.split('_')[1] for uid in concat_names]
    name2ind = {name: i for i, name in enumerate(names1)}
    embeddings1 = embeddings1[[name2ind[name] for name in to_keep_names1]]
    embeddings2 = embeddings1[[name2ind[name] for name in to_keep_names2]]

    embeddings1 = np.hstack([embeddings1, embeddings2])
    print('Embeddings shape', embeddings1.shape)

    return concat_names, embeddings1


def get_protein_labels(labels, names, split):
    labels = [labels[split][name] for name in names]
    return labels


def encode_tags(labels, tag2id):
    labels = list(map(str, labels))
    labels = [tag2id[doc] for doc in labels]
    return labels


class DownstreamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.encodings[idx]
        labels = self.labels[idx]
        return {'embed': torch.tensor(embedding), 'labels': torch.tensor(labels)}

    def __len__(self):
        return len(self.labels)


class AttentionClassifier(nn.Module):
    def __init__(self, is_classifier, is_multilabel, input_dim=1280, num_classes=1, kernel_size=9,
                 conv_dropout: float = 0.1, internal_embedding: int = 1024, last_embedding: int = 864):
        super().__init__()

        self.num_labels = num_classes
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.is_multilabel = is_multilabel
            if self.num_labels == 2:
                self.is_binary = True
            else:
                self.is_binary = False

        self.feature_convolution = torch.nn.Conv1d(in_channels=input_dim, out_channels=input_dim, 
                                             kernel_size=kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = torch.nn.Conv1d(in_channels=input_dim, out_channels=input_dim, 
                                               kernel_size=kernel_size, stride=1,
                                               padding=kernel_size // 2)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(conv_dropout)

        self.first_layer = torch.nn.Linear(input_dim, internal_embedding)
        self.bn_first = torch.nn.BatchNorm1d(internal_embedding)
        
        self.second_layer = torch.nn.Linear(internal_embedding, last_embedding)
        self.bn_second = torch.torch.nn.BatchNorm1d(last_embedding)

        self.output = torch.nn.Linear(last_embedding, self.num_labels) 

    def forward(self, embed, labels=None):
        b_repr = self.reweighting_inputs(embed)

        layer1 = self.dropout(self.relu(self.bn_first(self.first_layer(b_repr))))
        layer2 = self.dropout(self.relu(self.bn_second(self.second_layer(layer1))))
        logits = self.output(layer2)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def reweighting_inputs(self, input_vector):
        input_vector = torch.unsqueeze(input_vector, dim=2)
        o = self.feature_convolution(input_vector)  
        o = self.dropout(o)
        attention = self.attention_convolution(input_vector)
        o = torch.sum(o * self.softmax(attention), dim=-1)  
        return o

    def _compute_loss(self, logits, labels):
        if labels is not None:
            if self.is_classifier:
                if self.is_multilabel:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float()
                    )
                else:
                    loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = F.mse_loss(logits, labels[:, None])
        else:
            loss = None
        return loss
    
    
def align_predictions_protein_level(predictions: np.ndarray, label_ids: np.ndarray, id2tag: dict):

    if len(predictions.shape) == 3 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    preds = np.argmax(predictions, axis=1)

    if len(id2tag) > 2:
        out_label_list = []
        preds_list = []
        for i in range(len(preds)):
            if label_ids[i] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list.append(id2tag[label_ids[i]])
                preds_list.append(id2tag[preds[i]])

        return [preds_list], [out_label_list], None
    
    probas = scipy.special.softmax(predictions, axis=1)
    return preds, label_ids, probas[:, 1]
    

def compute_metrics_protein_level_classification(p: EvalPrediction, id2tag: dict):
    preds_list, out_label_list, probas = align_predictions_protein_level(p.predictions, 
                                                                         p.label_ids, 
                                                                         id2tag=id2tag)
    
    if len(id2tag) == 2:
        return {
            "roc_auc": sklearn.metrics.roc_auc_score(out_label_list, probas),
            "accuracy": sklearn.metrics.accuracy_score(out_label_list, preds_list),
            "precision": sklearn.metrics.precision_score(out_label_list, preds_list),
            "recall": sklearn.metrics.recall_score(out_label_list, preds_list),
            "f1": sklearn.metrics.f1_score(out_label_list, preds_list),
        }
    else:
        return {
            "accuracy": seq_metrics.accuracy_score(out_label_list, preds_list),
        }


def count_f1_max(pred, target):
	"""
	    F1 score with the optimal threshold, Copied from TorchDrug.

	    This function first enumerates all possible thresholds for deciding positive and negative
	    samples, and then pick the threshold with the maximal F1 score.

	    Parameters:
	        pred (Tensor): predictions of shape :math:`(B, N)`
	        target (Tensor): binary targets of shape :math:`(B, N)`
    """
	order = pred.argsort(descending=True, axis=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)
	
	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
	                torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
	             torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1[~torch.isnan(all_f1)].max()


def compute_metrics_protein_level_multilabel_classification(p: EvalPrediction):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    p.predictions = sigmoid(p.predictions)
    fmax = count_f1_max(torch.tensor(p.predictions), torch.tensor(p.label_ids))
        
    return {
        "fmax": fmax,
    }


def compute_metrics_regression(p: EvalPrediction):
    return {
        "spearmanr": stats.spearmanr(p.label_ids, p.predictions).correlation,
    }


def model_init_attn(num_tokens, embed_dim, device, conv_dropout, internal_embedding, 
                    last_embedding, is_classifier, is_multilabel):

    downstream_model = AttentionClassifier(is_classifier=is_classifier, 
                                           is_multilabel=is_multilabel,
                                           input_dim=embed_dim, num_classes=num_tokens,
                                           kernel_size=7, conv_dropout=conv_dropout, 
                                           internal_embedding=internal_embedding, 
                                           last_embedding=last_embedding)
    return downstream_model.to(device)


def run_cv_experiment(label_file, emb_path, agg_type, results_path, seed, param_grid,
                      is_classifier, is_multilabel, is_fast):
    
    def dict2str(dict_):
        return '  '.join([f'{key}={val}' for key, val in dict_.items()])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    postfix = os.path.basename(label_file).split('id2label.json')[0]
    if postfix != '':
        postfix = f'_{postfix[:-1]}'
    print('POSTFIX', postfix)

    task_name = os.path.basename(Path(label_file).parent.absolute())

    agg_type1 = agg_type
    model_type = os.path.basename(emb_path)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    all_labels = load(label_file)

    if task_name == 'secondary_structure_pdb':
        test_splits = ['casp12', 'ts115', 'cb513']
        log_steps = 250
    else:
        test_splits = ['test']
        log_steps = 10

    if task_name == 'humanppi':
        train_names, train_embeddings = load_humanppi_data(emb_path1=emb_path, agg_type1=agg_type1, 
                                                            split='train', 
                                                            concat_names=sorted(all_labels['train'].keys()))
        val_names, val_embeddings = load_humanppi_data(emb_path1=emb_path, agg_type1=agg_type1, 
                                                        split='valid',
                                                        concat_names=sorted(all_labels['valid'].keys()))
        test_names = []
        test_embeddings = []
        for test_split in test_splits:
            test_names_cur, test_embeddings_cur = load_humanppi_data(emb_path1=emb_path, agg_type1=agg_type1, 
                                                            split=test_split,
                                                            concat_names=sorted(all_labels['test'].keys()))
            test_names.append(test_names_cur)
            test_embeddings.append(test_embeddings_cur)

    else:
        train_names, train_embeddings = load_prot_level_data(emb_path1=emb_path, agg_type1=agg_type1, 
                                                            split='train')
        val_names, val_embeddings = load_prot_level_data(emb_path1=emb_path, agg_type1=agg_type1, 
                                                        split='valid')
        test_names = []
        test_embeddings = []
        print(test_splits)
        print(task_name)
        for test_split in test_splits:
            test_names_cur, test_embeddings_cur = load_prot_level_data(emb_path1=emb_path, 
                                                                       agg_type1=agg_type1, 
                                                                       split=test_split)
            test_names.append(test_names_cur)
            test_embeddings.append(test_embeddings_cur)
    
    train_labels = get_protein_labels(all_labels, train_names, split='train')
    val_labels = get_protein_labels(all_labels, val_names, split='valid')
    test_labels = [get_protein_labels(all_labels, test_names[i], split=test_splits[i]) 
                   for i in range(len(test_splits))]
    print('Train embeddings', type(train_embeddings), train_embeddings.shape)

    model_embed_dim = train_embeddings.shape[-1]

    if is_classifier:
        if is_multilabel:            
            train_labels_encodings = train_labels
            val_labels_encodings = val_labels
            test_labels_encodings = test_labels

            target_metric = 'fmax'
            compute_metrics_fn = compute_metrics_protein_level_multilabel_classification
            num_tokens = len(train_labels[0])
        else:
            # Create unique tag for each class
            unique_tags = set(doc for doc in train_labels)
            tag2id = {str(tag): int(float(tag)) for tag in unique_tags}
            id2tag = {int(float(tag)): str(tag) for tag, _ in tag2id.items()}

            # Encode the tags in the dataset
            train_labels_encodings = encode_tags(train_labels, tag2id=tag2id)
            val_labels_encodings = encode_tags(val_labels, tag2id=tag2id)
            test_labels_encodings = [encode_tags(test_labels[i], tag2id=tag2id) for i in range(len(test_splits))]

            print('Class labels', set(train_labels_encodings))
            num_tokens = len(unique_tags)
            
            compute_metrics_fn = partial(compute_metrics_protein_level_classification, id2tag=id2tag)
            if len(id2tag) == 2:
                target_metric = 'roc_auc'
            else:
                target_metric = 'accuracy'
    else:
        scaler = StandardScaler()
        train_labels_encodings = scaler.fit_transform(np.array(train_labels)[:, None])[:, 0].astype(np.float32)
        val_labels_encodings = scaler.transform(np.array(val_labels)[:, None])[:, 0].astype(np.float32)
        test_labels_encodings = [scaler.transform(np.array(test_labels[i])[:, None])[:, 0].astype(np.float32) 
                                 for i in range(len(test_splits))]

        target_metric = 'spearmanr'
        compute_metrics_fn = compute_metrics_regression
        num_tokens = 1

    training_dataset = DownstreamDataset(train_embeddings, train_labels_encodings)
    val_dataset = DownstreamDataset(val_embeddings, val_labels_encodings)
    test_datasets = [DownstreamDataset(test_embeddings[i], test_labels_encodings[i]) 
                     for i in range(len(test_splits))]
        
    experiment = f'{task_name}_{model_type}_{agg_type}_{postfix}'
    metric_for_best_model = f'test_{target_metric}'

    all_results = {}

    best_metric = -2
    best_test_metric = -2
    best_trainer = None
    best_params = None
    batch_size = param_grid['batch_size']
    grid_size = len(param_grid['lr']) * len(param_grid['conv_dropout']) * \
                len(param_grid['internal_embedding']) * len(param_grid['last_embedding']) * \
                len(param_grid['num_epochs'])
    exp_ind = 1
    is_done = False
    for lr in param_grid['lr']:
        for conv_dropout in param_grid['conv_dropout']:
            for internal_embedding in param_grid['internal_embedding']:
                for last_embedding in param_grid['last_embedding']:
                    for num_epochs in param_grid['num_epochs']:
                        print(f'{model_type} {agg_type} {postfix}: exp {exp_ind} / {grid_size}')
                        cur_params = {}
                        cur_params['lr'] = lr
                        cur_params['conv_dropout'] = conv_dropout
                        cur_params['internal_embedding'] = internal_embedding
                        cur_params['last_embedding'] = last_embedding
                        cur_params['num_epochs'] = num_epochs
                        
                        print(cur_params)
                        cur_metrics, cur_trainer = run_training(lr, conv_dropout, internal_embedding, last_embedding, 
                                    experiment, num_tokens, device, model_embed_dim, 
                                    training_dataset, val_dataset, is_classifier, is_multilabel, compute_metrics_fn,
                                    target_metric, batch_size, num_epochs, log_steps)
                        print('CUR METRIC:', cur_metrics[metric_for_best_model])
                        print('cur best metric:', best_metric)
                        for i, split in enumerate(test_splits):
                            predictions, labels, metrics_output = cur_trainer.predict(test_datasets[i])
                            print(split)
                            print(metrics_output)
                        if cur_metrics[metric_for_best_model] > best_metric:
                            best_metric = cur_metrics[metric_for_best_model]
                            best_trainer = cur_trainer
                            best_params = cur_params
                            print('NEW BEST METRIC', best_metric)
                        if metrics_output[metric_for_best_model] > best_test_metric:
                            best_test_metric = metrics_output[metric_for_best_model]
                            print('NEW BEST TEST METRIC', best_test_metric)
                        print('best params:', best_params)
                        print()
                        exp_ind += 1

                        all_results[dict2str(cur_params)] = (cur_metrics, metrics_output)
                        if is_fast:
                            is_done = True
                            break
                    if is_done:
                        break
                if is_done:
                    break
            if is_done:
                break
        if is_done:
            break

    all_metrics_output = []
    for i, split in enumerate(test_splits):
        predictions, labels, metrics_output = best_trainer.predict(test_datasets[i])
        print(split)
        print(experiment)
        print(best_test_metric, metrics_output)
        print('BEST PARAMS:', best_params)
        all_metrics_output.append(metrics_output)

    if not is_classifier:
        labels = scaler.inverse_transform(labels.reshape((1, -1)))
        predictions = scaler.inverse_transform(predictions)

    save(labels, os.path.join(results_path, f'labels{postfix}.npy'))
    save_json(metrics_output, os.path.join(results_path, 
                                           f'{experiment}{agg_type}{postfix}_seed{seed}.json'))
    save(predictions, os.path.join(results_path, 
                                   f'{experiment}{agg_type}_seed{seed}_predictions{postfix}.npy'))
    save_json(all_results, os.path.join(results_path, 
                                        f'{experiment}{agg_type}{postfix}_seed{seed}{postfix}_all_metrics.json'))
    
    print()
    print()
    return experiment, all_metrics_output, best_test_metric

def run_training(lr, conv_dropout, internal_embedding, last_embedding, experiment, num_tokens, 
                 device, model_embed_dim, training_dataset, val_dataset, is_classifier, is_multilabel, 
                 compute_metrics_fn, target_metric, batch_size, num_epochs, log_steps):

    training_args = TrainingArguments(
        output_dir=os.path.join(results_path, 'checkpoints', f'checkpoints_{experiment}_{seed}'),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.,
        learning_rate=lr,
        weight_decay=0.0,
        logging_dir=os.path.join(results_path, 'logs', f'logs_{experiment}'),
        do_train=True,
        do_eval=True,
        report_to='none',
        logging_strategy='steps',
        logging_steps=log_steps,
        evaluation_strategy="steps",
        eval_steps=log_steps,
        save_strategy='steps',
        save_steps=log_steps,
        save_total_limit=1,
        gradient_accumulation_steps=1, 
        fp16=False,
        fp16_opt_level="02",
        run_name=experiment,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model=f'eval_{target_metric}',
        greater_is_better=True,
        lr_scheduler_type='constant',
    )

    trainer = Trainer(
        model_init=partial(model_init_attn, 
                           is_classifier=is_classifier,
                           is_multilabel=is_multilabel,
                           num_tokens=num_tokens, 
                           embed_dim=model_embed_dim, 
                           device=device, 
                           conv_dropout=conv_dropout, 
                           internal_embedding=internal_embedding, 
                           last_embedding=last_embedding),
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    predictions, labels, metrics_output = trainer.predict(val_dataset)
    return metrics_output, trainer
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file")
    parser.add_argument("-f", "--fast", dest='is_fast',
                        required=False, action='store_true', default=False, 
                        help="if use no grid search for evaluation")
    args = parser.parse_args()

    config = load_config(args.config_filename)
    print(config)

    downstream_datasets_path = config["downstream_datasets_path"]
    downstream_experiments = config["downstream_tasks"]
    all_models = config["all_models"]

    all_results = {}
    for downstream_experiment in downstream_experiments:
        exp_path = os.path.join(downstream_datasets_path, downstream_experiment)
        results_path = os.path.join(exp_path, 'downstream_results')
        emb_path_common = os.path.join(exp_path, 'embeddings')
        label_files = [os.path.join(exp_path, 'id2label.json')]
        param_grid = { # base grid
                'batch_size': 4096 * 2,
                'lr': [5e-5],
                'conv_dropout': [0.1, 0.2], 
                'internal_embedding': [1536, 1024, 512],
                'last_embedding': [512, 256],
                'num_epochs': [200],
            }

        if downstream_experiment == 'localization_deeploc' or \
           downstream_experiment == 'localization_deeploc_binary' or \
           downstream_experiment == 'metal':
            is_classifier = True
            is_multilabel = False
        
        if downstream_experiment == 'thermostability' or downstream_experiment == 'fluorescence':
            is_classifier = False
            is_multilabel = False

        if downstream_experiment == 'binding':
            is_classifier = False
            is_multilabel = False

        elif downstream_experiment == 'go':
            label_files = [os.path.join(exp_path, f'{ont}_id2label.json') for ont in ['cc', 'mf', 'bp']]
            is_classifier = True
            is_multilabel = True
            param_grid = { # go
                'batch_size': 4096 * 2,
                'lr': [5e-4],
                'conv_dropout': [0.1], 
                'internal_embedding': [1536],
                'last_embedding': [768],
                'num_epochs': [200],
            }

        elif downstream_experiment == 'humanppi':
            is_classifier = True
            is_multilabel = False
            param_grid['conv_dropout'] = [0.2]

        elif downstream_experiment.startswith('secondary_structure_pdb'):
            exp_path = os.path.join(downstream_datasets_path, 'secondary_structure_pdb')
            emb_path_common = os.path.join(exp_path, 'embeddings')

            is_classifier = True
            is_multilabel = False
            param_grid = { # ss
                'batch_size': 4096 * 2,
                'lr': [1e-4],
                'conv_dropout': [0.1], 
                'internal_embedding': [1024],
                'last_embedding': [512],
                'num_epochs': [20],
            }
            
            if downstream_experiment == 'secondary_structure_pdb_3':
                label_files = [os.path.join(exp_path, 'id2label_pdb_ssp3.json')]
                results_path = os.path.join(exp_path, 'downstream_results_3state')

            elif downstream_experiment == 'secondary_structure_pdb_8':
                label_files = [os.path.join(exp_path, 'id2label_pdb_ssp8.json')]
                results_path = os.path.join(exp_path, 'downstream_results_8state')

        os.makedirs(results_path, exist_ok=True)
        print('Embeddings:', emb_path_common)

        seed = 14
        agg_type = '_avg'
        for model_i, model_name in enumerate(all_models):
            emb_path = os.path.join(emb_path_common, model_name)
            print(f'START: {model_i + 1}/{len(all_models)} {model_name} seed={seed}')
            for label_file in label_files:
                print(label_file)
                exp_name, exp_metrics, best_test_metric = run_cv_experiment(label_file, emb_path, 
                                    agg_type, results_path, seed, 
                                    param_grid=param_grid, 
                                    is_classifier=is_classifier,
                                    is_multilabel=is_multilabel,
                                    is_fast=args.is_fast)
                all_results[exp_name] = (best_test_metric, exp_metrics)
                
                print()
            print('------------------')
            print()

        print('EXPERIMENT RESULTS:')
        for key in all_results.keys():
            print(key)
            print(all_results[key])
            print()

        
    print('FINAL RESULTS:')
    for key in all_results.keys():
        print(key)
        print(all_results[key])
        print()
