#!/usr/bin/env python3

from nilearn.image import clean_img, iter_img, load_img, resample_to_img
from nilearn.masking import compute_multi_epi_mask
import collections
from collections.abc import Iterable, Sequence
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from typing import Union
import pandas as pd
from pandas import DataFrame as df

import cimaqprep
import loadutils as lu

def clean_resampled_fmri(fmri_img:Union[nib.nifti1.Nifti1Image,
                                         str, os.PathLike,
                                         collections.abc.Iterable],
                         mask_img:Union[nib.nifti1.Nifti1Image,
                                         str, os.PathLike]=None,
                         t_r:float=None,
                         confounds:pd.DataFrame=None,
                         detrend:bool=True,
                         low_pass:float=None,
                         high_pass:float=None,
                         ensure_finite:bool=True,
                         standardize:bool=True,
                         **kwargs
                         ) -> list:
    '''
    Description
    -----------
    Clean the fmri volumes to increase SN ratio
    Parameters
    ----------
    fmri_img: fMRI (4D) nifti image,
    mask_img: *optional* EPI (3D) nifti image,
    confounds: pd.DataFrame, *optional, default = None,
        - pd.DataFrame containing the confounds values
        - DataFrame X axis should be the same lenght as the number of fMRI volumes in the 4D image
            - i.e: len(nilearn.image.iter_img(fmri_img)) == confounds.shape[0]
        - Nilearn says:
        According to Lindquist et al. (2018), removal of confounds will be done
        orthogonally to temporal filters (low- and/or high-pass filters),
        if both are specified.
    low_pass: float, *optional, default = None,
        - Nilearn says:
        Low-pass filtering improves specificity.
        Filtering is only meaningful on evenly-sampled signals.
    high_pass: float, *optional, default = None,
        - Nilearn says:
        High-pass filtering should be kept small, to keep some sensitivity.
        Filtering is only meaningful on evenly-sampled signals.
    ensure_finite: bool, *optional, default=True.
        - from help(nilearn.image.clean_img):
            - If True, the non-finite values (NaNs and infs)
              found in the images will be replaced by zeros.
    t_r: float, *optional* default = None,
        - Can be manually specified, or else,
          it'll be found in the fmri_img's header
    resample_to_mask:bool, default=True,
        - If resample_to_mask is set to False, then all 3D images in 4D volume
          should all be of the same affine and shape as mask_image
    Returns
    -------
    clean_img (4D fMRI image)
    
    From http://www.humanbrainmapping.org/files/2015/Ed%20Materials/Temporal%20preprocessing%20REALLY%20FINAL_Tong.pdf
    Slide 13
    Lowpass filtering is typically not performed, because with standard TR values,
    cardiac and respiratory waveforms are severely aliased – modeling is usually a
    better approach to removing them – filtering the aliased signal will interfere.
    From https://www.sciencedirect.com/science/article/pii/S0165027021000157
    Detrending could be the solution: van Driel, Oliver & Fahrenfort (2018)
    "Trial-masked robust detrending."
    '''
    img_shapes_as_mask_shape = pd.Series(img.shape == mask_img.shape for img in
                                         list(iter_img(fmri_img))).unique()[0] == True
    if img_shapes_as_mask_shape:
        fmri_img = fmri_img
    else:
        fmri_img = cimaqprep.resample_fmri_to_mask(fmri_img, mask_img)
    if t_r is not None:
        t_r = t_r
    else:
        t_r = cimaqprep.get_tr_nscans_frametimes(fmri_img)[0]
    return clean_img(imgs=fmri_img,
                     detrend=detrend, 
                     standardize=standardize,
                     confounds=confounds,
                     low_pass=low_pass, high_pass=high_pass,
                     t_r=t_r,
                     ensure_finite=ensure_finite,
                     mask_img=mask_img)

def main():
    clean_resampled_fmri(fmri_img, **kwargs)
 
if __name__ == "__main__":
    main()                              
        
