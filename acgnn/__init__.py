"""ACGNN: audio-visual gated cross-modal completion over a heterogeneous
conversation graph for emotion recognition in conversation."""

from .data import (DATASETS, DEFAULT_FEATURE_PATHS, IEMOCAPRoBERTaDataset,
                   MELDRoBERTaDataset, get_data)
from .engine import run_epoch, seed_everything
from .graph import build_heterogeneous_graph, extract_main_features
from .graphnet import HeteroGraphAttention, MultiSourceAttentionFusion
from .losses import FocalLoss, MultimodalContrastiveLearning
from .model import ACGNN, DualModalMaskReconstruction

__version__ = '1.0.0'

__all__ = [
    'ACGNN',
    'DualModalMaskReconstruction',
    'HeteroGraphAttention',
    'MultiSourceAttentionFusion',
    'MultimodalContrastiveLearning',
    'FocalLoss',
    'build_heterogeneous_graph',
    'extract_main_features',
    'run_epoch',
    'seed_everything',
    'get_data',
    'DATASETS',
    'DEFAULT_FEATURE_PATHS',
    'IEMOCAPRoBERTaDataset',
    'MELDRoBERTaDataset',
]
