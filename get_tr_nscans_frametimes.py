#!/usr/bin/env python3

import collections
from collections.abc import Iterable
import nibabel as nib
import numpy as np
from typing import Union
import os
from nilearn.image import concat_imgs

def get_tr_nscans_frametimes(
    fmri_img:Union[nib.nifti1.Nifti1Image, str, os.PathLike],
    **kwargs
) -> tuple:
    if isinstance(fmri_img, nib.nifti1.Nifti1Image):
        fmri_img = fmri_img
    if isinstance(fmri_img, Iterable):
        fmri_img = concat_imgs(fmri_img, auto_resample=True)
    else:
        if isinstance(fmri_img, str) or isinstance(fmri_img, os.PathLike):
            fmri_img = nib.load(fmri_img)
    img_header = dict(fmri_img.header)
    tr, n_scans = img_header['pixdim'][4], img_header['dim'][4]
    frame_times = np.arange(n_scans) * tr
    return tr, n_scans, frame_times

def main():
    get_tr_nscans_frametimes(fmri_img, **kwargs)
 
if __name__ == "__main__":
    main()

