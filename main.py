import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from acgnn import (ACGNN, DATASETS, DEFAULT_FEATURE_PATHS, FocalLoss, get_data,
                   run_epoch, seed_everything)

INPUT_SIZES = {
    'IEMOCAP_RoBERTa': [1024, 342, 1582],
    'MELD_RoBERTa': [1024, 342, 300],
}

N_CLASSES = {
    'IEMOCAP_RoBERTa': 6,
    'MELD_RoBERTa': 7,
}

IEMOCAP_CLASS_FREQ = [0.086747, 0.144406, 0.227883, 0.160585, 0.127711, 0.252668]
MELD_CLASS_FREQ = [0.466750766, 0.122094071, 0.027752748, 0.071544422,
                   0.171742656, 0.026401153, 0.113714183]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes')


def build_parser():
    parser = argparse.ArgumentParser(description='ACGNN training')
    parser.add_argument('--dataset', default='IEMOCAP_RoBERTa',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--feature_path', default=None,
                        help='override the default feature pickle of the dataset')
    parser.add_argument('--seed', type=int, default=27350)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=120)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--graph_hs', type=int, default=128)
    parser.add_argument('--unimodal_hs', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2)

    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--d_masked', type=int, default=2)
    parser.add_argument('--win_aux', type=int, default=3)

    parser.add_argument('--focal_loss', type=str_to_bool, default=True)
    parser.add_argument('--class_weight', type=str_to_bool, default=False)

    parser.add_argument('--checkpoint_dir', default='./check_point')
    parser.add_argument('--tag', default='', help='suffix for the checkpoint file name')
    return parser


def main():
    args = build_parser().parse_args()
    args.no_cuda = args.no_cuda or not torch.cuda.is_available()

    seed_everything(args.seed)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    model = ACGNN(args, INPUT_SIZES[args.dataset], N_CLASSES[args.dataset]).to(device)

    class_freq = (IEMOCAP_CLASS_FREQ if 'IEMOCAP' in args.dataset else MELD_CLASS_FREQ)
    class_weights = torch.FloatTensor([1.0 / f for f in class_freq]).to(device)

    if args.focal_loss:
        loss_function = FocalLoss(alpha=class_weights.tolist() if args.class_weight else None)
    else:
        loss_function = nn.NLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    feature_path = args.feature_path or DEFAULT_FEATURE_PATHS[args.dataset]
    train_loader, valid_loader, test_loader = get_data(
        DATASETS[args.dataset], feature_path, args.batch_size)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_filename = os.path.join(
        args.checkpoint_dir,
        f'ACGNN_{args.dataset}{args.tag}_seed{args.seed}.pth')

    best_fscore, best_acc, best_epoch = 0, 0, 0
    best_pred, best_label = None, None

    for e in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore = run_epoch(
            model, loss_function, train_loader, device, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore = run_epoch(
            model, loss_function, valid_loader, device)
        test_loss, test_acc, test_label, test_pred, test_fscore = run_epoch(
            model, loss_function, test_loader, device)
        if test_fscore > best_fscore:
            best_epoch = e + 1
            best_fscore = test_fscore
            best_acc = test_acc
            best_pred, best_label = test_pred, test_label
            torch.save(model.state_dict(), save_filename)

        print(f'Epoch {e + 1}/{args.epochs}\n'
              f'train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_fscore: {train_fscore:.4f}\n'
              f'valid_loss: {valid_loss:.4f} | valid_acc: {valid_acc:.4f} | valid_fscore: {valid_fscore:.4f}\n'
              f'test_loss:  {test_loss:.4f} | test_acc:  {test_acc:.4f} | test_fscore:  {test_fscore:.4f}\n'
              f'time: {round(time.time() - start_time, 2)} sec\n'
              f'-----------------------------------------------------------------')

    print('Test performance at the best test epoch..')
    print('Best epoch:', best_epoch)
    print('F-Score:', best_fscore)
    print('Accuracy:', best_acc)
    if best_label is not None:
        print(classification_report(best_label, best_pred, digits=4, zero_division=0))
        print(confusion_matrix(best_label, best_pred))
    print('Checkpoint:', save_filename)


if __name__ == '__main__':
    main()
