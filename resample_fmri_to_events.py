#!/usr/bin/env python3

from nilearn.image import clean_img, iter_img, load_img, mean_img, resample_to_img
from nilearn.masking import compute_multi_epi_mask
import collections
from collections.abc import Iterable, Sequence
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import Union
import pandas as pd
from pandas import DataFrame as df

import loadutils as lu
import cimaqprep

def resample_fmri_to_events(fmri_img:nib.Nifti1Image,
                            mask_img:nib.Nifti1Image=None,
                            n_events:Union[int,pd.DataFrame,np.array]=None,
                            t_r:float=None,
                            frame_times:collections.abc.Iterable=None,
                            resample_to_mask:bool=True,
                            clean_resampled_imgs:bool=True,
                            **kwargs):
    from nilearn.image import clean_img, iter_img, resample_to_img
    # Resample fmri volumes' shapes and affines to those of the epi mask
    '''
    From nilearn.image.clean_img documentation:
    "If mask is provided, it should have same shape and affine as imgs.""
    '''
    img_shapes_as_mask_shape = pd.Series([img if isinstance(img, nib.Nifti1Image)
                                          else nib.load(img)][0].shape == mask_img.shape
                                         for img in fmri_img).unique()[0] == True
    if img_shapes_as_mask_shape:
        fmri_img = fmri_img
    else:
        fmri_img = cimaqprep.resample_fmri_to_mask(fmri_img, mask_img)
#     if resample_to_mask:
#         fmri_img = cimaqprep.resample_fmri_to_mask(fmri_img, mask_img)
#     else:
#         fmri_img = fmri_img
    if clean_resampled_imgs:
        fmri_img = cimaqprep.clean_resampled_fmri(fmri_img=fmri_img,
                                         mask_img=mask_img,
                                         resample_to_mask=[False if img_shapes_as_mask_shape
                                                           else True][0],
                                         **kwargs)
    else:
        fmri_img = fmri_img
    if frame_times is not None:
        frame_times=frame_times
    if t_r:
        t_r = t_r
    if not t_r and not frame_times:
        t_r, n_scans, frame_times = get_tr_nscans_frametimes(fmri_img)
    if isinstance(n_events, int):
        n_events = n_events
    if isinstance(n_events, (pd.DataFrame, np.ndarray)):
        n_events = n_events.shape[0]
    decomp_func = df((img for img in iter_img(fmri_img)),
                     columns=['imgs'])
    decomp_func['frame_times'] = frame_times
    test=df(pd.cut(decomp_func['frame_times'], n_events))
    test['imgs'] = decomp_func['imgs']
    return df(((grp, mean_img(
                test.groupby('frame_times').get_group(grp)['imgs']))
               for grp in tqdm(list(test.groupby('frame_times').groups),
                               desc='resampling fMRI volumnes to events')))

def main():
    resample_fmri_to_events(fmri_img)
 
if __name__ == "__main__":
    main()

