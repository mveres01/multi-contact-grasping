import os
import sys
import glob
import h5py
import numpy as np

sys.path.append('..')

from lib.utils import (format_htmatrix, invert_htmatrix,
                       get_unique_idx, convert_grasp_frame)
from lib.config import (config_output_collected_dir,
                        config_output_processed_dir)


def get_outlier_mask(data_in, sigma=3):
    """Find dataset outliers by whether or not it falls within a given number
    of standard deviations from a given population

    Parameters
    ----------
    data_in : array of (n_samples, n_variables) datapoints
    m : number of standard deviations to evaluate
    """

    if data_in.ndim == 1:
        data_in = np.atleast_2d(data_in).T

    # Collect mean and std
    mean = np.mean(data_in, axis=0)
    std = np.std(data_in, axis=0)

    # Create a boolean mask of data within *m* std
    mask = abs(data_in - mean) < sigma * std
    mask = np.sum(mask, axis=1)

    # Want samples where all variables are within a 'good' region
    return np.where(mask == data_in.shape[1])


def postprocess(h5_pregrasp, h5_postgrasp):
    """Standardizes data by removing outlier grasps."""

    def remove_from_dataset(dataset, indices):
        """Convenience function for filtering bad indices from dataset."""
        for key, value in dataset.iteritems():
            if isinstance(value, dict):
                for subkey in dataset[key].keys():
                    dataset[key][subkey] = dataset[key][subkey][indices]
            else:
                dataset[key] = dataset[key][indices]
        return dataset

    pregrasp, postgrasp = {}, {}
    for key, val in h5_pregrasp.iteritems():
        pregrasp[key] = val[:]
    for key, val in h5_postgrasp.iteritems():
        postgrasp[key] = val[:]

    if len(pregrasp) == 0 or len(postgrasp) == 0:
        return None, None

    # Clean the dataset: Remove duplicate pregrasp poses via frame_work2palm
    unique = get_unique_idx(pregrasp['frame_work2palm'], -1, 1e-1)
    print('%d / %d unique' % (len(unique), len(pregrasp['frame_work2palm'])))
    
    pregrasp = remove_from_dataset(pregrasp, unique)
    postgrasp = remove_from_dataset(postgrasp, unique)

    # Remove duplicate contact positions / normals
    grasp = np.hstack([pregrasp['work2contact0'][:],
                       pregrasp['work2contact1'][:],
                       pregrasp['work2contact2'][:],
                       pregrasp['work2normal0'][:],
                       pregrasp['work2normal1'][:],
                       pregrasp['work2normal2'][:]])
    unique = get_unique_idx(grasp, -1, 1e-1)
    print('%d / %d unique' % (len(unique), len(pregrasp['frame_work2palm'])))

    pregrasp = remove_from_dataset(pregrasp, unique)
    postgrasp = remove_from_dataset(postgrasp, unique)


    # If we've collected a lot of grasps, remove extreme outliers
    if pregrasp['frame_work2palm'].shape[0] > 50:

        # -- Remove any super wild grasps
        good_indices = get_outlier_mask(pregrasp['frame_work2palm'], sigma=4)
        pregrasp = remove_from_dataset(pregrasp, good_indices)
        postgrasp = remove_from_dataset(postgrasp, good_indices)

        if pregrasp['frame_work2palm'].shape[0] == 0:
            return None, None

    # Make sure we have the same number of samples for all data elements
    keys = pregrasp.keys()
    pregrasp_size = pregrasp['frame_work2palm'].shape[0]
    postgrasp_size = postgrasp['frame_work2palm'].shape[0]

    assert all(pregrasp_size == pregrasp[k].shape[0] for k in keys)
    assert all(postgrasp_size == postgrasp[k].shape[0] for k in keys)
    return pregrasp, postgrasp
