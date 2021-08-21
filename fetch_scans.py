#!/usr/bin/env python3

import os
import pandas as pd
from pandas import DataFrame as df
from typing import Union
import loadutils as lu

def fetch_scans(subject_dir:Union[str,os.PathLike]) -> pd.DataFrame:
	scans = lu.loadfiles([itm for itm in lu.loadimages(subject_dir)
						  if not itm.endswith('.json')])
	return df(((grp, scans.groupby('parent').get_group(grp).fpaths.values)
                             for grp in scans.groupby(
                                 'parent').groups)).T                              
def main():
    fetch_scans(subject_dir)
 
if __name__ == "__main__":
    main()

