#!/usr/bin/env python3
import os
from os.path import expanduser as xpu
from nilearn.image import clean_img, concat_imgs, iter_img, load_img, mean_img, resample_to_img
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

# def resample_fmri_to_events(fmri_img:nib.Nifti1Image,
#                             mask_img:nib.Nifti1Image=None,
#                             n_events:Union[int,pd.DataFrame,np.array]=None,
#                             t_r:float=None,
#                             frame_times:collections.abc.Iterable=None,
# #                             resample_to_mask:bool=True,
# #                             clean_resampled_imgs:bool=True,
#                             **kwargs):
def resample_fmri_to_events(func_img,
                            frame_times,
                            n_events):
    '''
    Description
    -----------
    Resample fmri volumes' shapes and affines to those of the epi mask
    
    Parameters
    ----------
    fmri_img: 4D fMRI image
    mask_img: 3D EPI mask image
              From nilearn.image.clean_img documentation:
              "If mask is provided, it should have same shape and affine as imgs.""
    n_events: Number of events (or trials) in an event-related experimental design
    frame_times: Onset of each slice in fMRI image
    clean_resampled_imgs: Indicate if fMRI slices should be cleaned to increase SNR or not
    kwargs: Keyword arguments suitable for nilearn.image.clean_image function
    
    Returns
    -------
    List of fMRI slices of lenght equal to the number of trials (n_events)
    '''
#     img_shapes_as_mask_shape = pd.Series(img.shape == mask_img.shape
#                                          for img in list(iter_img(fmri_img))).unique()[0] == True
#     if img_shapes_as_mask_shape:
#         fmri_img = fmri_img
#     else:
#         fmri_img = cimaqprep.resample_fmri_to_mask(fmri_img, mask_img)
#     if clean_resampled_imgs:
#         fmri_img = cimaqprep.clean_resampled_fmri(fmri_img=fmri_img,
#                                          mask_img=mask_img, **kwargs)
#     else:
#         fmri_img = fmri_img
#     if frame_times is not None:
#         frame_times=frame_times
#     if t_r:
#         t_r = t_r
#     if not t_r and not frame_times:
#         t_r, n_scans, frame_times = get_tr_nscans_frametimes(fmri_img)
#     if isinstance(n_events, int):
#         n_events = n_events
#     if isinstance(n_events, (pd.DataFrame, np.ndarray)):
#         n_events = n_events.shape[0]
#     decomp_func = df((img for img in iter_img(fmri_img)),
#                      columns=['imgs'])
#     decomp_func['frame_times'] = frame_times
#     test=df(pd.cut(decomp_func['frame_times'], n_events))
#     test['imgs'] = decomp_func['imgs']
#     return df(((grp, mean_img(
#                 test.groupby('frame_times').get_group(grp)['imgs']))
#                for grp in tqdm(list(test.groupby('frame_times').groups),
#                                desc='resampling fMRI volumnes to events')))

    imgdf=df(zip(frame_times,
                 list(iter_img(func_img))),
             columns=['frame_times','images'])
    test=df(pd.cut(imgdf['frame_times'], n_events))
    newtimes=df(test.frame_times.unique(), columns=['frame_times'])
    newtimes['images']=[mean_img(concat_imgs([subrow[1].images
                                              for subrow in imgdf.iterrows()
                        if subrow[1].frame_times in row[1].frame_times]))
                        for row in tqdm(list(newtimes.iterrows()),
                                        desc='resampling fMRI image to events lenght')]
    return concat_imgs(newtimes.images).shape

def main():
    resample_fmri_to_events(fmri_img)
 
if __name__ == "__main__":
    main()
