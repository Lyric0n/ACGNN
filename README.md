## IEMOCAP

```bash
python main.py --dataset IEMOCAP_RoBERTa --epochs 120 \
    --lr 2e-4 --l2 1e-4 --batch_size 48 --dropout 0.5 --alpha 0.7 \
    --focal_loss True --class_weight False
```

## MELD

```bash
python main.py --dataset MELD_RoBERTa --epochs 60 \
    --lr 5e-5 --l2 1e-4 --batch_size 32 --dropout 0.5 --alpha 0.7 \
    --focal_loss False --class_weight False
```

## Acknowledgements

* The code structure of this repository is adapted from **MMGCN** (Multimodal Fusion via
  Deep Graph Convolution Network for Emotion Recognition in Conversation).
* The **dataset features** are provided by **RL-EMO**.
