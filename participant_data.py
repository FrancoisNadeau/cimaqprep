#!/usr/bin/env python3

import numpy as np
import os
from typing import Union
import pandas as pd
from os.path import basename as bname
from os.path import expanduser as xpu
from os.path import join as pjoin
import loadutils as lu
from cimaqprep.fetch_events_behav import fetch_events_behav
from cimaqprep.fetch_scans_infos import fetch_scans_infos
from cimaqprep.fetch_participant import fetch_participant
from cimaqprep.get_tr_nscans_frametimes import get_tr_nscans_frametimes 
from cimaqprep.get_epi_mask_fromdata import get_epi_mask_fromdata 
from cimaqprep.resample_fmri_to_events import resample_fmri_to_events 

class participant_data:
    '''
    Description
    -----------
    Fetch and load a single participant's data (scans, infos,
    events and behavioural) in an object similar
    to a sklearn.utils.Bunch object (the way nilearn datasets are fetched)
    
    Parameters
    ----------
    cimaq_nov_dir: Path to CIMA-Q november 2019 dataset
    cimaq_mar_dir: Path to CIMA-Q march 2019 dataset
    events_dir: Path to events (in-scan trials) directory
    behav_dir: Path to behavioural (out-scan trials) directory
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
    def __init__(self,cimaq_nov_dir:Union[str,os.PathLike],
                 cimaq_mar_dir:Union[str,os.PathLike],
                 events_dir:Union[str,os.PathLike],
                 behav_dir:Union[str,os.PathLike],
                 **kwargs):
        self.cimaq_nov_dir = cimaq_nov_dir
        self.cimaq_mar_dir = cimaq_mar_dir
        self.events_dir = events_dir
        self.behav_dir = behav_dir
        self.data_dir = pjoin(self.cimaq_mar_dir,
                              'derivatives/CIMAQ_fmri_memory/data/')
        self.confounds_dir = (pjoin(self.data_dir,'confounds/resample'))
        self.participants_dir = pjoin(self.data_dir, 'participants')
        self.sub_id = fetch_participant(self.cimaq_mar_dir)
        self.mar_scans, self.mar_infos = fetch_scans_infos(pjoin(self.cimaq_mar_dir,self.sub_id[0]))
        self.nov_scans, self.nov_infos = fetch_scans_infos(pjoin(self.cimaq_nov_dir,self.sub_id[1]))
        self.events, self.behav = fetch_events_behav(self.cimaq_mar_dir, self.events_dir,
                                                     self.behav_dir, self.sub_id)
        self.confounds = [pd.read_csv(itm, sep='\t') for itm in
                          lu.loadimages(self.confounds_path)
                          if bname(itm).split('_')[1][3:] == \
                              self.sub_id[0].split('-')[1]][0]

        self.t_r, self.n_scans, self.frame_times = \
             get_tr_nscans_frametimes(self.mar_scans.func[1][0])
        self.resampled_frame_times = \
            np.arange(0, self.frame_times.max(),
                      self.frame_times.max()/self.events.shape[0])
        self.mar_epi_mask = get_epi_mask_fromdata(imgs=self.mar_scans.fmap[1])
        self.nov_epi_mask = get_epi_mask_fromdata(imgs=self.nov_scans.fmap[1])
        self.resampled_fmri_to_events = \
            resample_fmri_to_events(fmri_img=self.mar_scans.func[1][0],
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
    subject = participant_data(
        cimaq_nov_dir = xpu('~/../../media/francois/seagate_1tb/cimaq_20190901'),
        cimaq_mar_dir = xpu('~/../../media/francois/seagate_1tb/cimaq_03-19'),
        events_dir = xpu('~/../../media/francois/seagate_1tb/cimaq_corrected_events/events'),
        behav_dir = xpu('~/../../media/francois/seagate_1tb/cimaq_corrected_behavioural/behavioural'))
    return subject
 
if __name__ == "__main__":
    main()
