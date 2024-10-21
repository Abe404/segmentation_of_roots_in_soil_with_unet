# pylint: disable=C0111, R0913
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

import datetime
import time
import os
import sys
import glob
import torch
import wandb

# pylint: disable=C0413
sys.path.append('.')
from metrics import VALID_METRICS


def delete_all_except(directory, fnames):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path not in fnames:
            os.remove(file_path)


def latest_pkl_file(dirpath):
    # only looks at pkl (pickle files)
    list_of_files = glob.glob(dirpath + '/*.pkl')
    latest_file = None
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def delete_old_checkpoints(checkpoint_dir):
    best_checkpoint = latest_pkl_file(checkpoint_dir)
    if best_checkpoint:
        pkl_file = best_checkpoint.split('.')[0] + '.pkl'
        txt_file = best_checkpoint.split('.')[0] + '.txt'
        delete_all_except(checkpoint_dir, [pkl_file, txt_file])
    else:
        delete_all_except(checkpoint_dir, [])




class CheckPointer:

    def __init__(self, outdir, evaluation_metric, mode, batch_size,
                 train_data_size) -> None:
        self.checkpoint_dir = os.path.join(outdir, 'checkpoints')
        self.train_start = time.time()
        if not os.path.exists(self.checkpoint_dir):
            print('Creating', self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir)
        assert mode in ('max', 'min')
        self.mode = mode  # min or max (goal for eval metric and when to save)
        if evaluation_metric not in VALID_METRICS:
            raise Exception(f"{evaluation_metric} not in valid metrics"
                            f"please choose one of {VALID_METRICS}")
        self.evaluation_metric = evaluation_metric
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        if mode == "min":
            self.best_val = float('inf')
        else:
            self.best_val = 0

        # Track best model path and epoch to overwrite
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.best_epoch = 0

        # Create a W&B artifact once and update it during the run
        self.artifact = wandb.Artifact('best_model', type='model')

    def maybe_save(self, cnn, metrics, epoch) -> None:
        val = metrics[self.evaluation_metric]

        improvement = False
        if self.mode == 'min':
            improvement = (val < self.best_val)
        if self.mode == 'max':
            improvement = (val > self.best_val)

        if improvement:
            print(f"\nValidation {self.evaluation_metric} improved from {self.best_val:.5f} to {val:.5f}")
            self.best_epoch = epoch  # Track the best epoch
            self.best_val = val
            self.save_checkpoint(cnn, metrics, epoch)

        else:
            print(f"Not saving checkpoint as validation {self.evaluation_metric} did not improve")

    def save_checkpoint(self, cnn, val_metrics, epoch: int) -> None:
        """ Overwrite the best model checkpoint and update W&B artifact.
        """
        print(f"Overwriting best model checkpoint at {self.best_model_path} (Epoch: {epoch})")
        torch.save(cnn.state_dict(), self.best_model_path)

        # Create a text file with the details of the best model
        text_file_path = self.best_model_path.replace('pth', "txt")
        with open(text_file_path, "w") as text_file:
            now = datetime.datetime.now()
            train_duration = time.time() - self.train_start
            print(f"Date: {now}", file=text_file)
            print(f"Epoch: {epoch}", file=text_file)
            print(f"Batch set size: {self.batch_size}", file=text_file)
            print(f"Training data size: {self.train_data_size}", file=text_file)
            print(f"Train Duration: {train_duration:.1f} seconds", file=text_file)
            for name, val in val_metrics.items():
                print(f"Validation {name}: {val}", file=text_file)

        # Overwrite the existing W&B artifact
        self.artifact.add_file(self.best_model_path)  # Log model
        self.artifact.add_file(text_file_path)        # Log text file with epoch info
        wandb.log_artifact(self.artifact, aliases=['best_model'])
