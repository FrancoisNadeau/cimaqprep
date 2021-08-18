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

import loadutils as lu

def resample_to_smallest(imgs:Union[Iterable, Sequence],
                 **kwargs
                )->list:
    target_img = next(img for img in imgs if
                      [img if isinstance(img, nib.nifti1.Nifti1Image)
                       else nib.load(img)][0].shape == \
                      min(list([img if isinstance(img, nib.nifti1.Nifti1Image)
                                else nib.load(img)][0].shape for img in imgs)))
    source_imgs = list(img for img in imgs if img != target_img)
    return [nilearn.image.resample_to_img(source_img=img,
                                          target_img=target_img,
                                          interpolation='continuous',
                                          copy=True,
                                          order='F',
                                          clip=False,
                                          fill_value=0,
                                          force_resample=True)
                           for img in source_imgs + [target_img]]
def main():
    resample_to_smallest(imgs, **kwargs)
 
if __name__ == "__main__":
    main()
                          
