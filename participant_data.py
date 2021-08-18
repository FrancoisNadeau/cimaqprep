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
from pandas import DataFrame as df

import loadutils as lu

from .clean_resampled_fmri import clean_resampled_fmri 
from .get_tr_nscans_frametimes import get_tr_nscans_frametimes 
from .resample_to_smallest import resample_to_smallest 
from .get_epi_mask_fromdata import get_epi_mask_fromdata 
from .resample_fmri_to_events import resample_fmri_to_events 
from .get_outcomes import get_outcomes 
from .resample_fmri_to_mask import resample_fmri_to_mask

class participant_data:
    from nilearn import masking
    from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_epi
    from nilearn.image import concat_imgs, mean_img
    def __init__(self,cimaq_nov_dir,cimaq_mar_dir,
                 events_path,behav_path,participants_path):
        cimaq_nov_dir = cimaq_nov_dir
        cimaq_mar_dir = cimaq_mar_dir
        events_path = events_path
        behav_path = behav_path
        participants_path = participants_path
    
        # Load participants infos and indexing file
        participants = pd.read_csv(pjoin(participants_path, Participants_bids.tsv),
                                   sep = '\t')
        # Assing each participant to its double identifier
        subjects = df(tuple(('sub-'+str(itm[0]), 'sub-'+str(itm[1])) for itm in
                            tuple(zip(participants.participant_id, participants.pscid))),
                      columns = ['mar_subs', 'nov_subs'])

        # Remove participants who failed quality control
        task_qc = tuple('sub-'+str(itm[0]) for itm in
                        pd.read_csv(pjoin(dname(participants_dir), sub_list_TaskQC.tsv),
                              sep = '\t').values)
        subjects = subjects.iloc[[row[0] for row in subjects.iterrows()
                                        if row[1].mar_subs in task_qc]]

        # Select a random participant
        self.sub_id = subjects.sample(1).values.flatten()
        # Sort march scans and november scans in their respective DataFrames

        mar_scans = lu.loadfiles([itm for itm in
                                    lu.loadimages(pjoin(cimaq_mar_dir, self.sub_id[0]))
                                    if not itm.endswith('.json')])
        nov_scans = lu.loadfiles([itm for itm in
                                    lu.loadimages(pjoin(cimaq_nov_dir, self.sub_id[1]))
                                    if not itm.endswith('.json')])
        self.mar_scans = df(((grp, mar_scans.groupby('parent').get_group(grp).fpaths.values)
                             for grp in mar_scans.groupby('parent').groups)).set_index(0).T

        self.nov_scans = df(((grp, nov_scans.groupby('parent').get_group(grp).fpaths.values)
                             for grp in nov_scans.groupby('parent').groups)).set_index(0).T
        
        mar_infos = lu.loadfiles([itm for itm in
                                    lu.loadimages(pjoin(cimaq_mar_dir, self.sub_id[0]))
                                    if itm.endswith('.json')])
        self.mar_infos = df(((grp, mar_infos.groupby('parent').get_group(grp))
                             for grp in mar_infos.groupby('parent').groups))
        nov_infos = lu.loadfiles([itm for itm in
                                    lu.loadimages(pjoin(cimaq_nov_dir, self.sub_id[0]))
                                    if itm.endswith('.json')])
        self.nov_infos = df(((grp, nov_infos.groupby('parent').get_group(grp))
                             for grp in nov_infos.groupby('parent').groups))

        self.events = [pd.read_csv(pjoin(events_path, itm), sep = '\t')
                       for itm in lu.loadimages(events_path)
                       if bname(itm).split('_')[1] == self.sub_id[0].split('-')[1]][0]
        self.events['duration'] = [abs(row[1].stim_onset - row[1].fix_onset)
                                   for row in self.events.iterrows()]
        self.events = self.events.rename(columns = {'stim_onset': 'onset'})
        self.events['trial_type'] = self.events['category']
        self.behav = [pd.read_csv(pjoin(behav_path, itm), sep = '\t')
                      for itm in lu.loadimages(behav_path)
                      if bname(itm).split('_')[1] == \
                      self.sub_id[1].split('-')[1]][0].iloc[:, :-1]
        correctsources = self.events[['oldnumber', 'correctsource']]
        self.behav['correctsource'] = correctsources.correctsource
        self.behav['correctsource'] = [row[1].correctsource if row[1].oldnumber
                                               in lst_intersection(self.events.oldnumber,
                                                                   self.behav.oldnumber)
                                               else np.nan for row in self.behav.iterrows()]
        self.behav['spatial_acc'] = [row[1].spatial_resp == row[1].correctsource
                                             for row in self.behav.iterrows()]
        self.behav['recognition_acc'] = \
             self.behav['recognition_acc'].replace({0: False, 1: True})
        self.behav.recognition_resp = \
             self.behav.recognition_resp.replace({1: 'old', 2:'new'})
        recognition_accuracy = [row[1].category == row[1].recognition_resp
                                for row in self.behav.iterrows()]
        self.behav['recognition_acc'] = self.behav.recognition_resp.values == \
                                                  self.behav.category.values
        outcomes = get_outcomes(self.behav)
        self.behav['outcomes'] = outcomes
        outcomesdict=dict(zip(self.behav.oldnumber,
                      self.behav.outcomes))
        eventscopy=self.events
        eventscopy['outcomes']=eventscopy.oldnumber.map(outcomesdict)
        self.events=eventscopy
        self.confounds = [pd.read_csv(itm, sep='\t') for itm in
                          lu.loadimages(pjoin(cimaq_mar_dir,\
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
                                                                   'n_events':self.events})
def main():
    subject = participant_data(cimaq_nov_dir = xpu('~/../../data/cisl/DATA/cimaq_20190901'),
                               cimaq_mar_dir = xpu('~/../../data/cisl/DATA/cimaq_03-19'),
                               events_path = xpu('~/../../data/cisl/DATA/cimaq_corrected_events/events'),
                               behav_path = xpu('~/../../data/cisl/DATA/cimaq_corrected_behavioural/behavioural'),
                               participants_path = pjoin(cimaq_mar_dir, 'derivatives/CIMAQ_fmri_memory/data/participants/Participants'),
                               **kwargs)
    return subject
 
if __name__ == "__main__":
    main()
