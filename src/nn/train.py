# pylint: disable=C0111, R0914, R0915
"""
Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import time
import sys

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import wandb  # Import W&B

from datasets import UNetTrainDataset
from datasets import UNetValDataset
from checkpointer import CheckPointer
from models import get_model
from log import Logger
from loss import combined_loss
# pylint: disable=C0413
sys.path.append('.')
from metrics import get_metrics, get_metrics_str
from data_utils import get_files_split


def get_data_loaders(batch_size):
    """
    Load the train and validation photo and annotation paths.
    Return PyTorch data loaders which can be used for training and validating a neural network.
    batch size should be adjusted to the GPU memory
    """
    files = get_files_split()
    val_photos, train_photos, val_annotations, train_annotations = files
    train_set = UNetTrainDataset(train_annotations, train_photos)
    val_set = UNetValDataset(val_annotations, val_photos)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, drop_last=False,
                            pin_memory=True, shuffle=False, num_workers=8)

    return train_loader, val_loader


def evaluate(cnn, loader, device):
    """ get the
        loss (float)
        predictions (list) (int, 0 or 1)
        ground_truth (list) (int, 0 or 1)
    """
    start = time.time()
    correct = 0
    total = 0
    loss_sum = 0
    all_preds = []
    all_true = []
    cnn.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:

            # Prepare data for CNN, get loss
            tensor_y_batch = y_batch.type(torch.LongTensor).to(
                device, non_blocking=True)
            outputs = cnn(x_batch.to(device, non_blocking=True))
            loss = combined_loss(outputs, tensor_y_batch)
            loss_sum += loss.item()

            # store predictions and ground truth for metrics
            _, predicted = torch.max(outputs.data, 1)
            softmaxed = softmax(outputs, 1)
            root_probs = softmaxed[:, 1, :]  # just the root probability.

            # thresholded segmentation
            predicted = (root_probs > 0.5).view(-1).int()
            pred_np = predicted.data.cpu().numpy()
            y_batch_np = y_batch.view(-1).int().numpy()
            all_preds.append(pred_np)
            total += y_batch.size(0)

            # if this doesn't happen there is a bug
            assert len(pred_np) == len(y_batch_np)
            correct += (pred_np == y_batch_np).sum()
            all_true.append(y_batch_np)

    loss = loss_sum / len(all_true)
    all_true = np.concatenate(all_true)
    duration = time.time() - start
    print(f"Evaluate duation: {duration:.1f}")
    return loss, np.concatenate(all_preds), all_true


def train(cnn, outdir, learning_rate, epochs, batch_size):
    # Initialize W&B

    train_loader, val_loader = get_data_loaders(batch_size)

    # To use multiple GPUs
    # cnn = torch.nn.DataParallel(cnn, device_ids=[0, 1])

    optimizer = torch.optim.AdamW(cnn.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.3)
    checkpointer = CheckPointer(outdir, 'f1_score', 'max',
                                train_loader.batch_size,
                                len(train_loader.dataset))
    logger = Logger(outdir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cnn.to(device)
    global_step = 0

    for epoch in range(1, epochs + 1):
        print("Starting epoch", epoch)
        print("Assigning new tiles")
        train_loader.dataset.assign_new_tiles()
        cnn.train()
        epoch_start = time.time()

        all_preds = []
        all_true = []
        loss_sum = 0

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            # -- forward + backward + optimize --
            optimizer.zero_grad()
            outputs = cnn(x_batch) # each output in outputs is 388x388
            softmaxed = softmax(outputs, 1)
            root_probs = softmaxed[:, 1, :]  # just the root probability.
            predicted = root_probs > 0.5
            loss = combined_loss(outputs, y_batch)
            loss.backward()
            optimizer.step()
            preds = predicted.view(-1).cpu().data.numpy()
            y_batch = y_batch.view(-1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y_batch)
            loss_sum += loss.cpu().item()
            sys.stdout.write(f"Epoch progress: "
                             f"{step * train_loader.batch_size}"
                             f"/{len(train_loader.dataset)}  \r")
            sys.stdout.flush()
            global_step += 1

        duration = time.time() - epoch_start
        if scheduler:
            scheduler.step()

        print(f"\nTraining: epoch duration: {duration:.1f}")
        loss = loss_sum / len(all_true)
        all_true = np.concatenate(all_true)
        all_preds = np.concatenate(all_preds)
        metrics = get_metrics(all_preds, all_true, loss)
        logger.log_metrics('Train', metrics, global_step)
        print('Train', get_metrics_str(metrics))
        train_metrics_log = {f"train_{key}": value for key, value in metrics.items()}
        train_metrics_log["epoch"] = epoch
        train_metrics_log["train_loss"] = loss
        train_metrics_log["epoch_duration_s"] = duration
        wandb.log(train_metrics_log)

        val_loss, val_preds, val_true = evaluate(cnn, val_loader, device)
        val_metrics = get_metrics(val_preds, val_true, val_loss)
        print('Val', get_metrics_str(val_metrics))
        logger.log_metrics('Val', val_metrics, global_step)
        checkpointer.maybe_save(cnn, val_metrics, epoch)

        # Log validation metrics to W&B
        val_metrics_log = {f"val_{key}": value for key, value in val_metrics.items()}
        val_metrics_log["epoch"] = epoch
        val_metrics_log["val_loss"] = val_loss
        wandb.log(val_metrics_log)
        # Log metrics to W&B
        wandb.log({"epoch": epoch, "train_loss": loss, "val_loss": val_loss})

    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    wandb.init(project="segmentation_of_roots_in_soil_with_unet", entity="abe404-university-of-copenhagen")

    if wandb.run is not None:
        wandb.run.name = 'batch_sweep_' + str(wandb.config.model) + '_bs_' + str(wandb.config.batch_size) + '_lr_' + str(wandb.config.learning_rate)
        # Wandb is running, load parameters from Wandb config
        model = wandb.config.model
        learning_rate = wandb.config.learning_rate
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size
        pretrained_backbone = wandb.config.get("pretrained_backbone", False)
        pretrained_model = wandb.config.get("pretrained_model", False)
        outdir = wandb.config.get("outdir", f"../output/{model}/train_output")
    else:
        # Standalone mode, use command line arguments
        parser = argparse.ArgumentParser(
            description="Train a deep model for image segmentation"
        )
        parser.add_argument(
            "-m", "--model", choices=(
                "unet", "deeplabv3_mobilenet_v3_large", "deeplabv3_resnet101",
                "deeplabv3_resnet50", "fcn_resnet101", "fcn_resnet50",
                "lraspp_mobilenet_v3_large"))
        parser.add_argument("-a", "--learning-rate", type=float, default=1e-2)
        parser.add_argument("-o", "--outdir")
        parser.add_argument("-e", "--epochs", type=int, default=80)
        parser.add_argument("-B", "--pretrained-backbone", action="store_true")
        parser.add_argument("-M", "--pretrained-model", action="store_true")
        args = parser.parse_args()

        # Set output directory if not provided
        if args.outdir is None:
            args.outdir = f"../output/{args.model}/train_output"

        # Map arguments to variables for consistency
        model = args.model
        learning_rate = args.learning_rate
        epochs = args.epochs
        pretrained_backbone = args.pretrained_backbone
        pretrained_model = args.pretrained_model
        outdir = args.outdir

    # Now use the model and arguments
    train(
        get_model(model, pretrained_model, pretrained_backbone),
        outdir, learning_rate, epochs, batch_size
    )
