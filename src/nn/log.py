# pylint: disable=C0111
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


import os
from typing import Dict
from collections import defaultdict
import pickle
from tensorboard_logger import Logger as TFLogger

# pylint: disable=too-few-public-methods
class Logger:
    """
    Deals with writing tensorboard summaries.
    And logging metric history to a pickle file
    """
    def __init__(self, outdir):
        self.outdir = outdir
        self.tf_logger = TFLogger(os.path.join(outdir, 'run'), flush_secs=2)
        self.metric_history: Dict = defaultdict(list)

    def log_metrics(self, phase, metrics, global_step):
        """ Logs scalar values as tf summaries.
            Don't bother with true_mean, it stays the same
            and doesn't really work as a graph. """
        for name, value in metrics.items():
            if name != "true_mean":
                self.tf_logger.log_value(f"{phase} {name}", value, global_step)

        # save standard pickle object for easy matloblib plot or perf over epochs
        self.metric_history[phase].append(metrics)
        with open(os.path.join(self.outdir, "metric_history.pkl"), "wb") as metric_file:
            pickle.dump(self.metric_history, metric_file)
