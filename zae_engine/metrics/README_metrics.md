# Metrics Subpackage

The `metrics` subpackage of `zae-engine` provides various quantitative metrics for evaluating the performance of deep learning models. This subpackage allows for comparing model outputs with true labels, calculating different performance metrics, and objectively evaluating model performance to identify areas for improvement.

## Overview

The `metrics` subpackage includes the following metrics:

- **BijectiveMetrix**: Computes a bijective confusion matrix between predicted and actual sequences. This can be used with other functions in the subpackage to determine F1 score and accuracy.
  ```python
  from zae_engine.metrics import BijectiveMetrix
  metric = BijectiveMetrix(prediction, label, num_classes=3)
  print(metric.bijective_f1)
  ```
  - **Parameters**: `prediction`, `label`, `num_classes`, `th_run_length` (default: 2).
- **confusion_matrix**: Calculates a confusion matrix by comparing predicted labels (`y_hat`) with true labels (`y_true`). This allows for analyzing error distribution across classes.
  ```python
  from zae_engine.metrics import confusion_matrix
  conf_mat = confusion_matrix(y_hat, y_true, num_classes=3)
  ```
- **accuracy**: Calculates the accuracy by computing the ratio of correct predictions to the total number of predictions.
  ```python
  from zae_engine.metrics import accuracy
  acc = accuracy(true, predict)
  ```
- **f_beta**: Calculates the F-beta score, which balances recall and precision. The `beta` value can be adjusted to emphasize either recall or precision.
  ```python
  from zae_engine.metrics import f_beta
  f_beta_score = f_beta(pred, true, beta=1.0, num_classes=3, average='micro')
  ```
  - **Parameters**: The `beta` value is used to set the weight between recall and precision, and the `average` option allows for `micro` or `macro` averaging.
- **miou**: Calculates the mean Intersection over Union (mIoU) to evaluate object segmentation performance.
  ```python
  from zae_engine.metrics import miou
  score = miou(img1, img2)
  ```
- **giou**: Computes the Generalized Intersection over Union (GIoU), which extends IoU to better handle non-overlapping bounding boxes in object detection.
  ```python
  from zae_engine.metrics import giou
  score = giou(img1, img2)
  ```
- **rms**: Calculates the Root Mean Square (RMS) value of a signal, which measures the magnitude of the signal.
  ```python
  from zae_engine.metrics import rms
  rms_value = rms(signal)
  ```
- **signal_to_noise**: Computes the Signal-to-Noise Ratio (SNR) to evaluate the ratio of signal power to background noise power.
  ```python
  from zae_engine.metrics import signal_to_noise
  snr_value = signal_to_noise(signal, noise)
  ```
- **peak_signal_to_noise**: Calculates the Peak Signal-to-Noise Ratio (PSNR), which measures the ratio of peak signal power to background noise power.
  ```python
  from zae_engine.metrics import peak_signal_to_noise
  psnr_value = peak_signal_to_noise(signal, noise)
  ```

## Summary
The `metrics` subpackage of `zae-engine` provides various metrics to evaluate the quantitative performance of deep learning models. These metrics include:

- **Confusion Matrix-Based Metrics**: Analyze class prediction performance using `confusion_matrix` and `BijectiveMetrix`.
- **Basic Accuracy Metrics**: Evaluate classification accuracy using `accuracy`.
- **F-beta Score**: Balance recall and precision using `f_beta`.
- **IoU-Based Metrics**: Evaluate object segmentation and detection performance using `miou` and `giou`.
- **Signal Analysis Metrics**: Evaluate signal magnitude and noise ratio using `rms`, `signal_to_noise`, and `peak_signal_to_noise`.

These metrics can help users comprehensively evaluate and improve the performance of their deep learning models.

