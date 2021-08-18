#!/usr/bin/env python3

from nilearn.image import clean_img, iter_img, load_img, resample_to_img
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

def get_epi_mask_fromdata(imgs):
    ''' Compute a nilearn.masking.compute_multi_epi_mask from all available epi data
        Gets nilearn.image.mean_img for 3D images
        If an epi image is 4D, then gets mean_img for each 3D vol outputed by
        nilearn.image.iter_img(<4D_epi_img.nii.gz>), concatenates and auto resample
        all obtained 3D volumes and iterates over each concatenated, resampled and averaged
        3D volume to make the epi_imgs:list parameter to be passed to
        nilearn.image.compute_multi_epi_mask

        Parameters
        ----------
        imgs: list
        List of nifti images paths
        - valid epi images should contain the string "_epi" in their path
    '''
    all_resampled_epi_imgs = lu.flatten(list(image.iter_img(concat_imgs(
                                  lu.flatten(list([mean_img(load_img(img)) if 
                                                   len(load_img(img).shape)==3
                                                   else[mean_img(vol) for vol in
                                                        iter_img(load_img(img))]]
                                                  for img in tqdm(imgs)
                                                  if '_epi' in img)), auto_resample=True))))
    return compute_multi_epi_mask(epi_imgs=all_resampled_epi_imgs)
    
def main():
    get_epi_mask_fromdata(imgs)
 
if __name__ == "__main__":
    main()

