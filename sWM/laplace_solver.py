#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves Laplace equation over the domain of white matter.

Using grey matter as the source and ventricles as the sink.
Inputs are expected to be Free/FastSurfer aparc+aseg.mgz in .nii.gz format

Parameters
----------
NIFTI  :    str
            Parcellation file generated by Freesurfer/fastsurfre in nii.gz format (from mri/aparc+aseg.mgz).
NIFTI  :    str
            Output laplacian file path (nii.gz)

Returns
-------
NIFTI
    Laplacian image (nii.gz)

Usage
-----
laplace_solver.py aparc+aseg.nii.gz laplace-wm.nii.gz

Created on October 2023

@author: Jordan DeKraker
code from https://github.com/khanlab/hippunfold/blob/master/hippunfold/workflow/scripts/laplace_coords.py

"""

import nibabel as nib
import numpy as np
import skfmm
from scipy.ndimage import binary_dilation
from astropy.convolution import convolve as nan_convolve
import sys


def _get_bounding_box(mask, padding=1):
    # Get the bounding box coordinates
    non_zero_indices = np.transpose(np.nonzero(mask))

    # Get the minimum and maximum values along each axis
    min_coords = np.maximum(0, np.min(non_zero_indices, axis=0) - padding)
    max_coords = np.minimum(
        np.array(mask.shape), np.max(non_zero_indices, axis=0) + padding
    )

    return min_coords, max_coords


def _crop(x, min_bbox, max_bbox):
    return x[
        min_bbox[0] : max_bbox[0], min_bbox[1] : max_bbox[1], min_bbox[2] : max_bbox[2]
    ]


def laplace(
    init_coords,
    fg,
    source,
    sink,
    kernelSize=3,
    convergence_threshold=1e-4,
    max_iters=1000,
):
    hl = np.ones([kernelSize, kernelSize, kernelSize])
    hl = hl / np.sum(hl)

    # initialize coords
    coords = np.zeros(init_coords.shape)
    coords[fg] = init_coords[fg]
    coords[source] = 0
    coords[sink] = 1

    print("initialized solution")

    # iterate until the solution doesn't change anymore (or reach max iters)
    for i in range(max_iters):

        upd_coords = nan_convolve(coords, hl, fill_value=np.nan, preserve_nan=True)

        upd_coords[source] = 0
        upd_coords[sink] = 1

        # check difference between last
        diff_coords = coords[fg] - upd_coords[fg]
        diff_coords[np.isnan(diff_coords)] = 0
        ssd = (diff_coords * diff_coords).sum(axis=None)
        print(f"iteration {i}, convergence: {ssd}")
        if ssd < convergence_threshold:
            break
        coords = upd_coords

    return coords


def solve_laplace(
    in_seg,
    out_laplace,
    convergence_threshold=1e-4,
    max_iters=1000,
    kernelSize=3,
    alpha=0.1,
    fg_labels=[41, 2],
    src_labels=np.concatenate((np.arange(1000, 2999), [0])),
):
    print("starting laplace solver")

    # load data
    lbl_nib = nib.load(in_seg)
    lbl = lbl_nib.get_fdata()
    print("loaded data and parameters")

    # initialize foreground , source, and sink
    fg = np.isin(lbl, fg_labels)
    fg = binary_dilation(
        fg
    )  # dilate to make sure we always "catch" neighbouring surfaces in our gradient
    source = np.isin(lbl, src_labels)
    source[fg] = 0
    sink = ~(fg | source)

    # initialize solution with fast marching
    # fast march forward
    phi = np.ones_like(lbl)
    phi[source] = 0
    mask = np.ones_like(lbl)
    mask[fg] = 0
    mask[source] = 0
    phi = np.ma.MaskedArray(phi, mask)
    forward = skfmm.travel_time(phi, np.ones_like(lbl))
    init_coords = forward.data
    init_coords = init_coords - np.min(init_coords)
    init_coords = init_coords / np.max(init_coords)
    init_coords[fg] = 0

    # Work on cropped labelmap
    min_bbox, max_bbox = _get_bounding_box(fg, padding=kernelSize)

    cropped_coords = laplace(
        _crop(init_coords, min_bbox, max_bbox),
        _crop(fg, min_bbox, max_bbox),
        _crop(source, min_bbox, max_bbox),
        _crop(sink, min_bbox, max_bbox),
        kernelSize=kernelSize,
        convergence_threshold=convergence_threshold,
        max_iters=max_iters,
    )

    # Back to original size
    coords = np.zeros_like(init_coords)
    coords[
        min_bbox[0] : max_bbox[0], min_bbox[1] : max_bbox[1], min_bbox[2] : max_bbox[2]
    ] = cropped_coords
    coords[source] = 0
    coords[sink] = 1

    coords = coords * (1 - alpha) + (init_coords * alpha)

    # save file
    print("saving")
    coords_nib = nib.Nifti1Image(coords, lbl_nib.affine, lbl_nib.header)
    nib.save(coords_nib, out_laplace)


if __name__ == "__main__":
    in_seg = sys.argv[1]
    out_laplace = sys.argv[2]
    solve_laplace(in_seg, out_laplace)
