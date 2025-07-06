import os
import random
import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from Fusion_dataset_kFold import Fusion_Dataset
from Fusion_model_temporal import FusionModel
from PCQA_Model_ssg_temporal import PCQA_Model
from projectionModel import ResNet2D
from TemporalTransformer import TemporalTransformer
from utils import collate_fn, fit_function
from Loss import RankLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Fusion_Dataset(
        root_dir=args.root,
        projections_dir=args.projections_dir,
        labels_file=args.labels_file,
        patch_dir=args.patch_dir,
        csv_file=args.train_csv
    )

    test_dataset = Fusion_Dataset(
        root_dir=args.root,
        projections_dir=args.projections_dir,
        labels_file=args.labels_file,
        patch_dir=args.patch_dir,
        csv_file=args.test_csv
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

    pc_model = PCQA_Model(pretrained=True, checkpoint_path=args.checkpoint_path).to(device)
    img_model = ResNet2D().to(device)

    for model in [pc_model, img_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    fusion_model = nn.DataParallel(FusionModel(pc_model, img_model)).to(device)
    temporal_model = nn.DataParallel(TemporalTransformer(input_dim=1024)).to(device)

    optimizer = optim.Adam(list(fusion_model.parameters()) + list(temporal_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = RankLoss()
    scaler = GradScaler()

    for epoch in range(args.epochs):
        fusion_model.train()
        temporal_model.train()

        train_preds, train_labels = [], []
        total_loss = 0

        for batch_idx, (patches, img1, img2, mos_scores, _) in enumerate(train_loader):
            patches, img1, img2, mos_scores = patches.to(device), img1.to(device), img2.to(device), mos_scores.float().to(device)

            optimizer.zero_grad()

            frame_feat = fusion_model(patches, img1, img2)
            predictions = temporal_model(frame_feat)

            loss = criterion(predictions, mos_scores)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()

            train_preds.extend(predictions.detach().cpu().view(-1).numpy())
            train_labels.extend(mos_scores.detach().cpu().view(-1).numpy())

        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels).flatten()
        plcc_train = pearsonr(train_preds, train_labels)[0]
        srocc_train = spearmanr(train_preds, train_labels)[0]
        scheduler.step()

        with open(args.log_file, 'a') as file:
            file.write(f'Epoch {epoch + 1}, Train Loss: {total_loss}, PLCC: {plcc_train}, SROCC: {srocc_train}\n')

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {total_loss}, PLCC: {plcc_train}, SROCC: {srocc_train}')

        fusion_model.eval()
        temporal_model.eval()

        test_pred_scores, test_true_scores = [], []

        for batch_idx, (patches, img1, img2, mos_scores, _) in enumerate(test_loader):
            patches, img1, img2, mos_scores = patches.to(device), img1.to(device), img2.to(device), mos_scores.float().to(device)

            with torch.no_grad():
                test_frame_feat = fusion_model(patches, img1, img2)
                test_predictions = temporal_model(test_frame_feat)

            test_pred_scores.extend(test_predictions.detach().cpu().view(-1).numpy())
            test_true_scores.extend(mos_scores.detach().cpu().view(-1).numpy())

        test_pred_scores = np.array(test_pred_scores).flatten()
        test_true_scores = np.array(test_true_scores).flatten()

        test_pred_logistic = fit_function(test_true_scores, test_pred_scores)
        plcc_test = pearsonr(test_pred_logistic, test_true_scores)[0]
        srocc_test = spearmanr(test_pred_scores, test_true_scores)[0]
        krcc_test = kendalltau(test_pred_scores, test_true_scores)[0]
        rmse_test = np.sqrt(mean_squared_error(test_pred_logistic, test_true_scores))

        with open(args.log_file, 'a') as file:
            file.write(f'Epoch {epoch + 1}, PLCC: {plcc_test}, SROCC: {srocc_test}, KROCC: {krcc_test}, RMSE: {rmse_test}\n')

        print(f'Epoch {epoch+1}/{args.epochs}, Test PLCC: {plcc_test}, SROCC: {srocc_test}, KROCC: {krcc_test}, RMSE: {rmse_test}')

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Model with Temporal Aggregation Training")
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--projections_dir', type=str, required=True)
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--patch_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
