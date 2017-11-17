import os
import sys
import glob
import h5py
import numpy as np

sys.path.append('..')

from lib.utils import (format_htmatrix, invert_htmatrix,
                       get_unique_idx, convert_grasp_frame)
from lib.config import (config_output_collected_dir,
                        config_output_dataset_path)


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


def merge_datasets(data_dir, save_path=config_output_dataset_path):
    """Given a directory, load a set of hdf5 files given as a list."""

    data_list = glob.glob(os.path.join(data_dir, '*.hdf5'))

    # Write each of the train/test/valid splits to file
    savefile = h5py.File(save_path, 'w')

    # For each of the decoded objects in the data_dir
    for fname in data_list:

        object_name = fname.split(os.path.sep)[-1].split('.')[0]

        # If trying to open a file that wasn't closed properly
        try:
            input_file = h5py.File(fname, 'r')
        except Exception:
            continue
        pregrasp, postgrasp = postprocess(input_file['pregrasp'],
                                          input_file['postgrasp'])

        if pregrasp is None or postgrasp is None:
            print '%s no data returned!' % fname
            continue
        elif pregrasp['frame_work2palm'].shape[0] < 1:
            continue

        group = savefile.create_group(object_name)
        pregrasp_group = group.create_group('pregrasp')

        pregrasp_data = np.hstack([pregrasp['work2contact0'][:],
                                   pregrasp['work2contact1'][:],
                                   pregrasp['work2contact2'][:],
                                   pregrasp['work2normal0'][:],
                                   pregrasp['work2normal1'][:],
                                   pregrasp['work2normal2'][:]])

        pregrasp_group.create_dataset('grasp', data=pregrasp_data, compression='gzip')
        for key in pregrasp.keys():
            pregrasp_group.create_dataset(key, data=pregrasp[key], compression='gzip')

        postgrasp_group = group.create_group('postgrasp')

        postgrasp_data = np.hstack([postgrasp['work2contact0'][:],
                                    postgrasp['work2contact1'][:],
                                    postgrasp['work2contact2'][:],
                                    postgrasp['work2normal0'][:],
                                    postgrasp['work2normal1'][:],
                                    postgrasp['work2normal2'][:]])

        postgrasp_group.create_dataset('grasp', data=postgrasp_data, compression='gzip')
        for key in postgrasp.keys():
            postgrasp_group.create_dataset(key, data=postgrasp[key], compression='gzip')

    savefile.close()


if __name__ == '__main__':
    merge_datasets(config_output_collected_dir, config_output_dataset_path)
