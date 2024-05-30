import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import EvalPrediction



def compute_eval_metrics(p: EvalPrediction):

    labels = p.label_ids
    predictions, losses = p.predictions
    ce_loss, contact_loss, mse_loss, angle_err = losses.mean(axis=0)

    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    result_metrics = {}
    result_metrics['top1_acc'] = np.mean(predictions == labels)
    result_metrics['ce_loss'] = ce_loss
    result_metrics['contact_loss'] = contact_loss
    result_metrics['mse_loss'] = mse_loss
    result_metrics['angle_err'] = angle_err

    return result_metrics


def preprocess_logits_for_metrics(predictions, labels):
    pred_ids = torch.argmax(predictions['scores'], dim=-1)

    ce_loss_fn = CrossEntropyLoss()
    masked_lm_loss = ce_loss_fn(predictions['scores'].view(-1, predictions['scores'].shape[-1]), 
                                labels[0].view(-1)).item()

    contact_loss = 0.
    if 'contact' in predictions.keys() or 'distance' in predictions.keys() or \
                                          'bin_distance' in predictions.keys():
        contact_mask = labels[1] != -1.
        if 'contact' in predictions.keys():
            loss_fn = BCEWithLogitsLoss()
            preds_name = 'contact'
        elif 'distance' in predictions.keys():
            loss_fn = MSELoss()
            preds_name = 'distance'
        else:
            loss_fn = CrossEntropyLoss()
            preds_name = 'bin_distance'
        contact_loss = loss_fn(predictions[preds_name][contact_mask], labels[1][contact_mask])

    mse_loss = 0.
    angle_err = 0.
    if 'angles' in predictions.keys():
        angle_mask = labels[-1] > -99.
        mse_loss = torch.abs(labels[2][angle_mask] - predictions['angles'][angle_mask])
        mse_loss[mse_loss > 0.5] = 1 - mse_loss[mse_loss > 0.5]
        angle_err = mse_loss.mean().item() * 360
        mse_loss = (mse_loss ** 2).mean().item()
    
    return pred_ids.cpu(), torch.tensor([[masked_lm_loss, contact_loss, mse_loss, angle_err]])
