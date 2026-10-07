import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loss_function, dataloader, device, optimizer=None, train=False):

    losses, preds, labels = [], [], []

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, emo_labels = [
            d.to(device) for d in data[:-1]
        ]
        dia_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1
                       for j in range(len(umask))]
        label = torch.cat([emo_labels[j][:dia_lengths[j]] for j in range(len(emo_labels))])
        emo_labels = emo_labels.transpose(0, 1)

        all_log_prob, contrastive_loss, _ = model(textf, qmask, dia_lengths,
                                                  emo_labels, acouf, visuf)
        task_loss = loss_function(all_log_prob, label)
        loss = model.uncertainty_weighted_loss(task_loss, contrastive_loss)

        preds.append(torch.argmax(all_log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted', zero_division=0) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore
