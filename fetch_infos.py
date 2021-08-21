#!/usr/bin/env python3

import os
import pandas as pd
from pandas import DataFrame as df
from typing import Union
import loadutils as lu

def fetch_infos(subject_dir:Union[str,os.PathLike]) -> pd.DataFrame:
	infos = lu.loadfiles([itm for itm in lu.loadimages(subject_dir)
						  if itm.endswith('.json')])
	return df(((grp, infos.groupby('parent').get_group(grp).fpaths.values)
                             for grp in infos.groupby(
                                 'parent').groups)).T        
def main():
    fetch_infos(subject_dir)
 
if __name__ == "__main__":
    main()
