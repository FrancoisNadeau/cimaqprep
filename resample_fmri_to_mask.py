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

def resample_fmri_to_mask(fmri_img:nib.Nifti1Image,
						  mask_img:nib.Nifti1Image
                         ) -> nib.Nifti1Image:
    return concat_imgs([resample_to_img(source_img=img,
                                                      target_img=mask_img,
                                                      force_resample=True)
                                      for img in tqdm(list(iter_img(fmri_img)),
						      desc='Resampling fMRI volumes to epi mask)])
def main():
    resample_fmri_to_mask(fmri_img, mask_img)
 
if __name__ == "__main__":
    main()                              
        
