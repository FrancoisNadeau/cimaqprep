#!/usr/bin/env python3import os

import pandas as pd
from pandas import DataFrame as df
from typing import Union
import loadutils as lu

def fetch_scans_infos(subject_dir:Union[str,os.PathLike]) -> pd.DataFrame:
    '''
    Description
    -----------
    Load a participant's scans (nibabel.nifti1.Nifti1Image)
    and corresponding information (.json) files about each of
    his or her respective scans (nibabel.nifti1.Nifti1Image) files.

    Parameters
    ----------
    subject_dir: Path to participant's data directory

    Returns
    -------
    DataFrame containg subject's scans, in the same order as
    this subject's infos DataFrame (outputed by cimaqprep.fetch_infos)
    '''
    subject_files = lu.loadimages(subject_dir)
    scans = lu.filterlist_exc(exclude=['.json'], str_lst=subject_files)
    infos = lu.filterlist_exc(exclude=scans,str_lst=subject_files)
    def regroup_data(col:str,data:pd.DataFrame):
        return df(((grp, data.groupby(col).get_group(grp).fpaths.values)
                             for grp in data.groupby(
                                 col).groups)).set_index(0).T
    scan_df = regroup_data(col='parent',data=lu.loadfiles(scans))
    info_df = regroup_data(col='parent',data=lu.loadfiles(infos))
    return scan_df,info_df

def main():
    fetch_scans_infos(subject_dir)

if __name__ == "__main__":
    main()

