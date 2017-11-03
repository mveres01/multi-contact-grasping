import os
import csv
import sys
sys.path.append('..')
import glob
import h5py
import numpy as np

from lib.utils import float32
from lib.python_config import (config_train_dir, config_test_dir,
                               config_dataset_path,
                               config_train_item_list,
                               config_processed_data_dir)

def split_dataset(n_samples, train_size=0.8):
    """Generate train and test indices to split a dataset."""

    # Split data into train/test/valid
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_samples_train = int(train_size*n_samples)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    return train_indices, test_indices


def load_grasp_data(fname=None):
    """Loads a specific hdf5 file."""

    datafile = h5py.File(fname, 'r')

    # For convenience, access all of the information we'll need
    pregrasp = datafile['pregrasp']
    work2grasp = pregrasp['work2grasp'][:]

    # Collect everything else but the grasp
    keys = [k for k in pregrasp.keys() if k != 'work2grasp']
    misc_dict = {k:pregrasp[k] for k in keys}

    return (work2grasp, misc_dict)


def merge_datasets(data_dir, save_path=config_dataset_path):
    """Given a directory, load a set of hdf5 files given as a list."""

    # We'll append all data into a list before shuffle train/test/valid
    grasps = []
    misc_props = {}

    data_list = glob.glob(os.path.join(data_dir, '*.hdf5'))
    
    # Write each of the train/test/valid splits to file
    savefile = h5py.File(save_path, 'w')
    
    # For each of the decoded objects in the data_dir
    for fname in data_list:

        data = load_grasp_data(fname)

        if data is None:
            print '%s no data returned!'%fname
            continue
        grasps, props = data

        group = savefile.create_group(props['object_name'][0])
        group.create_dataset('grasps', data=grasps, compression='gzip')
        
        for key in props.keys():
            group.create_dataset(key, data=props[key], compression='gzip')
    
    savefile.close()


if __name__ == '__main__':
    merge_datasets(config_processed_data_dir, config_dataset_path)
