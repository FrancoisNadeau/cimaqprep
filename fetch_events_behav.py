#!/usr/bin/env python3

import nibabel as nib
import numpy as np
import os
from os import listdir as ls
from os.path import basename as bname
from os.path import join as pjoin
import pandas as pd
from pandas import DataFrame as df
from tqdm import tqdm
from typing import Union
from .get_outcomes import get_outcomes
import loadutils as lu

def fetch_events_behav(cimaq_mar_dir:Union[str, os.PathLike],
                       events_path:Union[str, os.PathLike],
                       behav_path:Union[str, os.PathLike],
                       sub_id:tuple) -> pd.DataFrame:
    '''
    Description
    -----------
    Load in-scan (events) and out-scan (behavioural) tasks performances
    and make them suitable for passing as parameters to
    nilearn.glm.first_level.FirstLevelModel and nilearn.glm.first_level.make_first_level_design_matrix

    Parameters
    ----------
    cimaq_mar_dir: Path to CIMA-Q march 2019 dataset
    events_path: Path to CIMA-Q march 2019 events files
    behav_path: Path to CIMA-Q march 2019 behavioural files
    sub_id: Tuple (march_id, november_id) containing a subject's 6-digit and 7-digit identifiers, respectively

    Returns
    -------
    Subject's events DataFrame and subject's behavioural DataFrame
    '''
    events = [pd.read_csv(pjoin(events_path, itm), sep = '\t')
			  for itm in tqdm(lu.loadimages(events_path),
                               desc='fetching in-scan events')
			  if bname(itm).split('_')[1] == sub_id[0].split('-')[1]][0]
#     events['duration'] = [abs(row[1].stim_onset - row[1].fix_onset)
#                                for row in tqdm(events.iterrows(),
#                                                 desc='computing trial durations')]
    events = events.rename(columns = {'stim_onset': 'onset'})
    events['trial_type'] = events['category']
    behav = [pd.read_csv(pjoin(behav_path, itm), sep = '\t')
			 for itm in tqdm(lu.loadimages(behav_path),
                              desc='fetching out-scan behavioural data')
			 if bname(itm).split('_')[1] == \
			 	sub_id[1].split('-')[1]][0].iloc[:, :-1]
    behav['correctsource'] = events.correctsource
    behav['correctsource'] = [row[1].correctsource if row[1].oldnumber
							  in lu.lst_intersection(events.oldnumber, behav.oldnumber)
							  else np.nan for row in tqdm(behav.iterrows(),
                                                          desc='finding correct spatial answers')]
    behav['spatial_acc'] = [row[1].spatial_resp == row[1].correctsource
							for row in tqdm(behav.iterrows(),
                                            desc='computing spatial accuracy')]
    behav['recognition_acc'] = \
         behav['recognition_acc'].replace({0: False, 1: True})
    behav.recognition_resp = behav.recognition_resp.replace({1: 'old', 2:'new'})
    behav['recognition_acc'] = behav.recognition_resp.values == behav.category.values
    behav['outcomes'] = get_outcomes(behav)
    events['outcomes']=events.oldnumber.map(dict(zip(behav.oldnumber, behav.outcomes)))
    func_dir = pjoin(cimaq_mar_dir,sub_id[0],'ses-4/func')
	func_hdr = dict(nib.load(pjoin(func_dir, os.path.listdir(func_dir)[1])).header)
    scan_shape, scan_tr = func_hdr['dim'][4],func_hdr['pixdim'][4]
    scan_dur=scan_shape*scan_tr
    events=events.drop('Unnamed: 0',axis=1).where(events.onset+events.duration < scan_dur)
    todrop=[row[0] for row in events.iterrows()
            if row[1].isnull().all()]
    events=events.drop(todrop,axis=0)
    return events, behav

def main():
    fetch_events_behav(cimaq_mar_dir,events_path,behav_path,sub_id)
 
if __name__ == "__main__":
    main()
							 
