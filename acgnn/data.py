import pickle
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class IEMOCAPRoBERTaDataset(Dataset):

    def __init__(self, path, train=True):
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.roberta1,
         self.videoText, self.roberta3, self.roberta4,
         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,
         self.testVid) = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText[vid])), \
            torch.FloatTensor(np.array(self.videoVisual[vid])), \
            torch.FloatTensor(np.array(self.videoAudio[vid])), \
            torch.FloatTensor([[1, 0] if x == 'M' else [0, 1]
                               for x in self.videoSpeakers[vid]]), \
            torch.FloatTensor([1] * len(self.videoLabels[vid])), \
            torch.LongTensor(self.videoLabels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4
                else pad_sequence(dat[i], True) if i < 6
                else dat[i].tolist() for i in dat]


class MELDRoBERTaDataset(Dataset):

    def __init__(self, path, train=True):
        (self.videoIDs, self.videoSpeakers, self.videoLabels, self.roberta1,
         self.videoText, self.roberta3, self.roberta4,
         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,
         self.testVid, _) = pickle.load(open(path, 'rb'))
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText[vid])), \
            torch.FloatTensor(np.array(self.videoVisual[vid])), \
            torch.FloatTensor(np.array(self.videoAudio[vid])), \
            torch.FloatTensor(np.array(self.videoSpeakers[vid])), \
            torch.FloatTensor([1] * len(self.videoLabels[vid])), \
            torch.LongTensor(self.videoLabels[vid]), \
            vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4
                else pad_sequence(dat[i], True) if i < 6
                else dat[i].tolist() for i in dat]


DATASETS = {
    'IEMOCAP_RoBERTa': IEMOCAPRoBERTaDataset,
    'MELD_RoBERTa': MELDRoBERTaDataset,
}

DEFAULT_FEATURE_PATHS = {
    'IEMOCAP_RoBERTa': './dataset_features/iemocap_multimodal_features.pkl',
    'MELD_RoBERTa': './dataset_features/meld_multimodal_features.pkl',
}


def get_train_valid_sampler(train_set, valid=0.1):
    size = len(train_set)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_data(dataset_class, path, batch_size=32, valid=0.1):

    train_set = dataset_class(path, train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(train_set, valid)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_sampler, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=valid_sampler, collate_fn=train_set.collate_fn)

    test_set = dataset_class(path, train=False)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             collate_fn=test_set.collate_fn)

    return train_loader, valid_loader, test_loader
