# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Source: https://github.com/google/uncertainty-baselines/blob/main/baselines/mnist/utils.py
#

import numpy as np
import torch
import logging


def calibration(y, class_pred, conf, num_bins=10):
    """Compute the calibration.

    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263

    Args:
      y: one-hot encoding of the true classes, size (?, num_classes)
      p_mean: numpy array, size (?, num_classes)
             containing the mean output predicted probabilities
      num_bins: number of bins

    Returns:
      ece: Expected Calibration Error
      mce: Maximum Calibration Error
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        class_pred = class_pred.cpu().numpy()
        conf = conf.cpu().numpy()
    # Compute for every test sample x, the predicted class.
    # class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    # conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    # y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins + 1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    if len(nb_items_bin) == 0:
        logging.warning("ECE computation failed.")
        return float("nan"), float("nan")

    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(float) / np.sum(nb_items_bin),
    )
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))
    return ece, mce
