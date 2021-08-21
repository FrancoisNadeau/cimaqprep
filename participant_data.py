#!/usr/bin/env python3

from nilearn.image import clean_img, iter_img, load_img, resample_to_img
from nilearn.masking import compute_multi_epi_mask
import collections
from collections.abc import Iterable, Sequence
import nibabel as nib
import numpy as np
import os
import sklearn
from tqdm import tqdm
from typing import Union
import pandas as pd

from os import listdir as ls
from os.path import basename as bname
from os.path import dirname as dname
from os.path import expanduser as xpu
from os.path import join as pjoin
from pandas import DataFrame as df
from pandas import DataFrame as df

import loadutils as lu

from .clean_resampled_fmri import clean_resampled_fmri
from .fetch_events_behav import fetch_events_behav
from .fetch_infos import fetch_infos
from .fetch_scans import fetch_scans
from .fetch_participant import fetch_participant
from .get_tr_nscans_frametimes import get_tr_nscans_frametimes 
from .resample_to_smallest import resample_to_smallest 
from .get_epi_mask_fromdata import get_epi_mask_fromdata 
from .resample_fmri_to_events import resample_fmri_to_events 
from .get_outcomes import get_outcomes 
from .resample_fmri_to_mask import resample_fmri_to_mask

class participant_data:
    '''
    Description
    -----------
    Fetch and load a single participant's data (scans, infos, events and behavioural) in
    an object similar to a sklearn.utils.Bunch object (the way nilearn datasets are fetched)
    
    Parameters
    ----------
    cimaq_nov_dir: Path to CIMA-Q november 2019 dataset
    cimaq_mar_dir: Path to CIMA-Q march 2019 dataset
    events_path: Path to events (in-scan trials) directory
    behav_path: Path to behavioural (out-scan trials) directory
    participants_path: Path to directory containing participant's
                       BIDS information and list of participant's for which
                       quality assessment was successful
    
    Returns
    -------
    participant_data: object similar to a sklearn.utils.Bunch
                      object containing a single participant's data
                      (scans, infos, events and behavioural)
    '''
    from nilearn import masking
    from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_epi
    from nilearn.image import concat_imgs, mean_img
    def __init__(self,cimaq_nov_dir,cimaq_mar_dir,
                 events_path,behav_path,participants_path,
                 **kwargs):
        self.cimaq_nov_dir = cimaq_nov_dir
        self.cimaq_mar_dir = cimaq_mar_dir
        self.events_path = events_path
        self.behav_path = behav_path
        self.participants_path = participants_path
        self.sub_id = fetch_participant(self.cimaq_mar_dir)
        self.mar_scans = fetch_scans(subject_dir=pjoin(self.cimaq_mar_dir, self.sub_id[0]))
        self.mar_infos = fetch_infos(subject_dir=pjoin(self.cimaq_mar_dir, self.sub_id[0]))
        self.nov_scans = fetch_scans(subject_dir=pjoin(self.cimaq_nov_dir, self.sub_id[1]))
        self.nov_infos = fetch_infos(pjoin(self.cimaq_nov_dir, self.sub_id[1]))
        self.events, self.behav = fetch_events_behav(self.cimaq_mar_dir, self.events_path,
                                                     self.behav_path, self.sub_id)
        self.confounds = [pd.read_csv(itm, sep='\t') for itm in
                          lu.loadimages(pjoin(self.cimaq_mar_dir,\
                                              'derivatives/CIMAQ_fmri_memory/data/confounds/resample'))
                          if bname(itm).split('_')[1][3:] == self.sub_id[0].split('-')[1]][0]

        self.t_r, self.n_scans, self.frame_times = \
             get_tr_nscans_frametimes(self.mar_scans.func[1][0])
        self.resampled_frame_times=np.arange(0, self.frame_times.max(),
                                          self.frame_times.max()/self.events.shape[0])
        # Compute epi masks for march and november scans, respectively
        self.mar_epi_mask = get_epi_mask_fromdata(imgs=self.mar_scans.fmap[1])
        self.nov_epi_mask = get_epi_mask_fromdata(imgs=self.nov_scans.fmap[1])
        self.resampled_fmri_to_events=resample_fmri_to_events(fmri_img=self.mar_scans.func[1][0],
                                                              mask_img=self.mar_epi_mask,
                                                                resample_to_mask=True,
                                                                clean_resampled_imgs=True,
                                                                **{'confounds':self.confounds,
                                                                   'low_pass':None,
                                                                   'high_pass':None,
                                                                   'detrend':True,
                                                                   'standardize':True,
                                                                   't_r':self.t_r,
                                                                   'ensure_finite':True,
                                                                   'frame_times':self.frame_times,
                                                                   'n_events':self.events})[1]
def main():
    subject = participant_data(cimaq_nov_dir = xpu('~/../../data/cisl/DATA/cimaq_20190901'),
                               cimaq_mar_dir = xpu('~/../../data/cisl/DATA/cimaq_03-19'),
                               events_path = xpu('~/../../data/cisl/DATA/cimaq_corrected_events/events'),
                               behav_path = xpu('~/../../data/cisl/DATA/cimaq_corrected_behavioural/behavioural'),
                               participants_path = pjoin(cimaq_mar_dir, 'derivatives/CIMAQ_fmri_memory/data/participants'),
                               **kwargs)
    return subject
 
if __name__ == "__main__":
    main()
