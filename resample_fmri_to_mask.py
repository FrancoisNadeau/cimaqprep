#!/usr/bin/env python3

from nilearn.image import clean_img, concat_imgs, iter_img, load_img, resample_to_img
from nilearn.masking import compute_multi_epi_mask
import collections
from collections.abc import Iterable, Sequence
import nibabel as nib
import os
import numpy as np
from tqdm import tqdm
from typing import Union
import pandas as pd
from pandas import DataFrame as df

import loadutils as lu

def resample_fmri_to_mask(fmri_img:Union[nib.Nifti1Image, str, os.PathLike],
						  mask_img:nib.Nifti1Image
                         ) -> nib.Nifti1Image:
	'''
	Description
	-----------
	Resample each fMRI slices in fMRI nib.Nifti1Image 4D file to
	the same shape and affine as the participant's EPI mask
	to avoid errors while using nilearn.image.clean_img function
	
	Parameters
	----------
	fmri_img: Participant's fMRI 4D image or path to fMRI 4D image
	mask_img: Participant's EPI 3D mask or path to this mask
	
	Returns
	-------
	Participant's fMRI 4D image, but with each slice now with
	the same shape and affine as the participant's EPI mask.
	'''
	fmri_img = [fmri_img if isinstance(fmri_img, nib.Nifti1Image)
				else nib.load(fmri_img)][0]
	mask_img = [mask_img if isinstance(mask_img, nib.Nifti1Image)
				else nib.load(mask_img)][0]
    return concat_imgs([resample_to_img(source_img=img,
                                                      target_img=mask_img,
                                                      force_resample=True)
                                      for img in tqdm(list(iter_img(fmri_img)),
						      desc='Resampling fMRI volumes to epi mask')])
def main():
    resample_fmri_to_mask(fmri_img, mask_img)
 
if __name__ == "__main__":
    main()                              
        
