#!/usr/bin/env python3

import os
from os import listdir as ls
from os.path import dirname as dname
from os.path import join as pjoin
import pandas as pd
from pandas import DataFrame as df
from typing import Union
from .get_outcomes import get_outcomes
import loadutils as lu

def fetch_participant(cimaq_mar_dir:Union[str,os.PathLike]) -> tuple:
	# Load participants infos and indexing file
	
	participants_path = pjoin(cimaq_mar_dir, 'derivatives/CIMAQ_fmri_memory/data/participants')
	task_qc_path = pjoin(dname(participants_path), 'sub_list_TaskQC.tsv')
	participants = pd.read_csv(pjoin(participants_path, 'Participants_bids.tsv'), sep = '\t')
	
	# Assing each participant to its double identifier
	subjects = df(tuple(('sub-'+str(itm[0]), 'sub-'+str(itm[1])) for itm in
		                tuple(zip(participants.participant_id, participants.pscid))),
		          columns = ['mar_subs', 'nov_subs'])

	# Remove participants who failed quality control
	task_qc = tuple('sub-'+str(itm[0]) for itm in
		            pd.read_csv(task_qc_path, sep = '\t').values)
	subjects = subjects.iloc[[row[0] for row in subjects.iterrows()
		                            if row[1].mar_subs in task_qc]]

	# Select a random participant
	return subjects.sample(1).values.flatten()

def main():
    fetch_participant(cimaq_mar_dir)
 
if __name__ == "__main__":
    main()

