# Loss Subpackage

The `loss` subpackage of `zae-engine` provides a variety of loss functions for different tasks in deep learning. It includes basic loss functions used in classification and regression tasks, as well as advanced and specialized loss functions for angular margin and IoU-based tasks.

## Overview

The `loss` subpackage consists of the following loss functions:

- **cross\_entropy**: Computes the binary cross-entropy loss with logits, commonly used for classification tasks.
  ```python
  from zae_engine.loss import cross_entropy
  loss = cross_entropy(logit, y_hot, class_weights)
  ```
- **compute\_gram\_matrix**: Computes the Gram matrix for a batch of vectors. This is useful for style transfer or regularization purposes.
  ```python
  from zae_engine.loss import compute_gram_matrix
  gram_matrix = compute_gram_matrix(batch)
  ```
- **ArcFaceLoss**: Implements ArcFace loss, which introduces an angular margin to improve the discriminative power of classification models. It is particularly useful in face recognition tasks.
  ```python
  from zae_engine.loss import ArcFaceLoss
  loss_fn = ArcFaceLoss(in_features=512, out_features=10)
  loss = loss_fn(features, labels)
  ```
  - **Parameters**: `in_features`, `out_features`, `s` (norm of input), `m` (margin).
  - This loss function is based on the paper "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" by Jiankang Deng et al.
- **mIoU**: Computes the mean Intersection over Union, a metric used to evaluate the accuracy of object segmentation.
  ```python
  from zae_engine.loss import mIoU
  score = mIoU(pred, true)
  ```
- **IoU**: Computes the Intersection over Union for segmentation tasks.
  ```python
  from zae_engine.loss import IoU
  score = IoU(pred, true)
  ```
- **GIoU**: Computes the Generalized Intersection over Union, which is useful for object detection tasks. GIoU extends IoU to handle non-overlapping bounding boxes better.
  ```python
  from zae_engine.loss import GIoU
  score = GIoU(true_onoff, pred_onoff)
  ```
- **mse**: Computes the mean squared error, a basic regression loss function that measures the average squared difference between predicted and true values.
  ```python
  from zae_engine.loss import mse
  loss = mse(true, predict)
  ```
  - MSE is commonly used in regression tasks to measure prediction accuracy.

## Summary

The `loss` subpackage of `zae-engine` provides a comprehensive collection of loss functions to support a wide range of deep learning tasks, including:

- **Basic loss functions** for classification (`cross_entropy`) and regression (`mse`).
- **Advanced angular margin loss** for facial recognition and other tasks requiring strong discriminative features (`ArcFaceLoss`).
- **IoU-based losses** for segmentation and object detection tasks (`mIoU`, `IoU`, `GIoU`).

These loss functions help users easily implement effective training strategies for various deep learning applications.
