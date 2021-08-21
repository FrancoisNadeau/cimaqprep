#!/usr/bin/env python3

import numpy as np
import os
from os import listdir as ls
from os.path import basename as bname
from os.path import join as pjoin
import pandas as pd
from pandas import DataFrame as df
from typing import Union
from .get_outcomes import get_outcomes
import loadutils as lu

def fetch_events_behav(cimaq_mar_dir:Union[str, os.PathLike],
                       events_path:Union[str, os.PathLike],
                       behav_path:Union[str, os.PathLike],
                       sub_id:tuple) -> pd.DataFrame:

    events = [pd.read_csv(pjoin(events_path, itm), sep = '\t')
                   for itm in lu.loadimages(events_path)
                   if bname(itm).split('_')[1] == sub_id[0].split('-')[1]][0]
    events['duration'] = [abs(row[1].stim_onset - row[1].fix_onset)
                               for row in events.iterrows()]
    events = events.rename(columns = {'stim_onset': 'onset'})
    events['trial_type'] = events['category']
    behav = [pd.read_csv(pjoin(behav_path, itm), sep = '\t')
                  for itm in lu.loadimages(behav_path)
                  if bname(itm).split('_')[1] == \
                  sub_id[1].split('-')[1]][0].iloc[:, :-1]
    behav['correctsource'] = events.correctsource
    behav['correctsource'] = [row[1].correctsource if row[1].oldnumber
                                           in lu.lst_intersection(events.oldnumber,
                                                               behav.oldnumber)
                                           else np.nan for row in behav.iterrows()]
    behav['spatial_acc'] = [row[1].spatial_resp == row[1].correctsource
                                         for row in behav.iterrows()]
    behav['recognition_acc'] = \
         behav['recognition_acc'].replace({0: False, 1: True})
    behav.recognition_resp = \
         behav.recognition_resp.replace({1: 'old', 2:'new'})
    behav['recognition_acc'] = behav.recognition_resp.values == \
                                              behav.category.values
    behav['outcomes'] = get_outcomes(behav)
    events['outcomes']=events.oldnumber.map(dict(zip(behav.oldnumber, behav.outcomes)))
    return events, behav

def main():
    fetch_events_behav(cimaq_mar_dir,events_path,behav_path,sub_id)
 
if __name__ == "__main__":
    main()
							 
