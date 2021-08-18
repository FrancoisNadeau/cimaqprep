#!/usr/bin/env python3

from nilearn.image import clean_img, iter_img, resample_to_img
import collections
from collections.abc import Iterable, Sequence
import numpy as np
import os
from typing import Union
import pandas as pd
from pandas import DataFrame as df

def get_outcomes(behav:pd.DataFrame):
	'''
	Compute behavioural (outside scanner) trial outcomes.
	"hit" = successful object and position recognition
	"recog_ok_spatial_wrong" = successful object recognition and
	 wrong position recognition
	"false_alarm" = new object misrecognized as old
	"corr_rejection" = new object recognized as new
	"miss" = old object misrecognized as new'''
	responses = []
    for row in behav.iterrows():
        if row[1].recognition_acc and row[1].spatial_acc:
            responses.append('hit')
        elif row[1].recognition_acc and not row[1].spatial_acc:
            responses.append('recog_ok_spatial_wrong')
        elif row[1].category == 'new' and row[1].recognition_resp == 'old':
            responses.append('false_alarm')
        elif row[1].category == 'new' and row[1].recognition_resp == 'new':
            responses.append('corr_rejection')
        elif row[1].category == 'old' and row[1].recognition_resp == 'new':
            responses.append('miss')
    return responses
    
def main():
    get_outcomes(behav)
 
if __name__ == "__main__":
    main()
    
