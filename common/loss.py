# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import traceback

import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    norms = (torch.norm(predicted - target, dim=len(target.shape)-1))
    return torch.nansum(norms) / torch.nansum(torch.isnan(norms) == False)
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    predicted = predicted[:,~np.isnan(target).all(axis=0).all(axis=1),:]
    target = target[:, ~np.isnan(target).all(axis=0).all(axis=1), :]
    target = target[:, ~np.isnan(predicted).all(axis=0).all(axis=1), :]
    predicted = predicted[:, ~np.isnan(predicted).all(axis=0).all(axis=1), :]

    muX = np.nanmean(target, axis=1, keepdims=True)
    muY = np.nanmean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.nansum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.nansum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    X0[np.isnan(X0)] = 0
    Y0[np.isnan(Y0)] = 0
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.nansum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t

    # Return MPJPE
    ret = np.nanmean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    if ret == np.nan:
        print("IS NAN")
        print(traceback.format_exc())
    return ret
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.nansum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.nansum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.nanmean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))