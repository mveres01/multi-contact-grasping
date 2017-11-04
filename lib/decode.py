import os
import sys
sys.path.append('..')

import csv
import h5py

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from lib.utils import (format_htmatrix, invert_htmatrix,
                        get_unique_idx, convert_grasp_frame)

# Save/data directories
from lib.python_config import (config_collected_data_dir, config_processed_data_dir,
                               config_sample_image_dir)

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
    mask = abs(data_in - mean) < sigma*std
    mask = np.sum(mask, axis=1)

    # Want samples where all variables are within a 'good' region
    return mask == data_in.shape[1]


def parse_grasp(line, header):
    """Parses a line of information following size convention in header."""

    line = np.atleast_2d(line)

    grasp = {}
    current_pos = 0

    split = header.split(',')

    # Decode all the data into a python dictionary
    for i in range(0, len(split), 2):

        name = str(split[i][1:-1])
        n_items = int(split[i+1])
        subset = line[:, current_pos:current_pos + n_items]

        try:
            subset = subset.astype(np.float32)
        except Exception as e:
            subset = subset.astype(str)

        grasp[name] = subset.ravel()
        current_pos += n_items

    return grasp

def decode_grasp(grasp_line):
    """Extracts different homogeneous transform matrices and grasp components."""

    if grasp_line['all_in_contact'] != 1:
        return None
    elif grasp_line['num_objects_colliding'] != 0:
        return None

    # Check that the force sensor was active for this trial
    fs0 = grasp_line['forceSensorStatus0']
    fs1 = grasp_line['forceSensorStatus1']
    fs2 = grasp_line['forceSensorStatus2']
    if fs0 != 1 or fs1 != 1 or fs2 != 1:
        raise Exception('Force sensor was broken. Should not have happened.')

    # Contact points, normals, and forces should be WRT world frame
    contact_points = np.hstack(
        [grasp_line['contactPoint0_wrt_world'],
         grasp_line['contactPoint1_wrt_world'],
         grasp_line['contactPoint2_wrt_world']])

    contact_normals = np.hstack(
        [grasp_line['contactNormal0_wrt_world'],
         grasp_line['contactNormal1_wrt_world'],
         grasp_line['contactNormal2_wrt_world']])


    # Encode the grasp into the workspace frame
    world2grasp = np.hstack([contact_points, contact_normals])
    world2grasp = np.atleast_2d(world2grasp)

    # These object properties are encoded WRT workspace
    work2com = grasp_line['com_workspace_wrt_world']
    work2mass = grasp_line['mass_workspace_wrt_world']
    work2inertia = grasp_line['inertia_workspace_wrt_world']

    #workspace_to_base = g['BarrettHand'][i]
    work2obj = grasp_line['object_wrt_workspace'].reshape(3, 4)
    work2obj = format_htmatrix(work2obj)
    obj2work = invert_htmatrix(work2obj)

    world2work = grasp_line['workspace_wrt_world'].reshape(3, 4)
    world2work = format_htmatrix(world2work)
    work2world = invert_htmatrix(world2work)

    work2palm = grasp_line['palm_wrt_workspace'].reshape(3, 4)
    work2palm = format_htmatrix(work2palm)

    # Some misc frames that will be helpful later
    obj2world = np.dot(obj2work, work2world)
    world2obj = invert_htmatrix(obj2world)

    # ### finally, convert the grasp frames
    work2grasp = convert_grasp_frame(work2world, world2grasp)
    work2grasp_pos = np.atleast_2d(grasp_line['palm_wrt_workspace_pos'])
    work2grasp_ori = np.atleast_2d(grasp_line['palm_wrt_workspace_orient'])

    # Make sure everything is the right dimension so we can later concatenate
    work2com = np.atleast_2d(work2com)
    work2mass = np.atleast_2d(work2mass)
    work2inertia = np.atleast_2d(work2inertia)

    world2obj = np.atleast_2d(world2obj[:3].flatten())
    world2work = np.atleast_2d(world2work[:3].flatten())
    work2obj = np.atleast_2d(work2obj[:3].flatten())
    work2palm = np.atleast_2d(work2palm[:3].flatten())

    return {'work2grasp':work2grasp,
            'work2inertia':work2inertia,
            'work2mass':work2mass,
            'work2com':work2com,
            'frame_work2palm':work2palm,
            'frame_world2obj':world2obj,
            'frame_world2work':world2work,
            'frame_work2obj':work2obj,
            }


def decode_raw_data(all_data):
    """Primary function that decodes collected simulated data."""

    # Initialize elements to be None
    # Note that "OTO" means the image and gripper share a one-to-one mapping
    # Note that "OTM" means the image and gripper share a one-to-many mapping
    keys = ['header', 'pregrasp', 'postgrasp']

    # We need to set these to be None, or else (as a list) they share memory
    decoded = {key: list() for key in keys}
    elems = dict.fromkeys(keys, None)


    # ---------------- Loop through all recorded data ------------------------

    count = 0 # which attempt
    successful = 0 # how many successful attempts
    for i, line in enumerate(all_data):

        # First item of a line is always what the line represents
        # (e.g. an image/grasp/header)
        data_type = line[0]
        data = line[1:-1]



        # No prefix inficates that image y-direction always points upwards
        if data_type == 'HEADER':
            elems['header'] = data

        elif data_type == 'PREGRASP':
            grasp = parse_grasp(data, elems['header'])
            preg = decode_grasp(grasp)
            elems['pregrasp'] = preg

        elif data_type == 'POSTGRASP':
            grasp = parse_grasp(data, elems['header'])
            postg = decode_grasp(grasp)
            elems['postgrasp'] = postg

            # Check we've retrieved an element for each component
            # This is where we'll catch whether or not the grasp was
            #   successful, as the 'postgrasp' should not be None
            count += 1
            if all(elems[k] is not None for k in keys):

                for k in elems.keys():
                    if 'header' not in k:
                        decoded[k].append(elems[k])

                successful += 1
                if successful % 50 == 0:
                    print 'Successful grasp #%4d/%4d'%(successful, count)

            # Reset the elements to be None
            elems.update(dict.fromkeys(elems.keys(), None))
        else:
            raise Exception('Data type: %s not understood'%data_type)


    # Quick check to see that we've decoded something
    if len(decoded['pregrasp']) == 0:
        return False

    # Go through the collected pregrasp/postgrasp arrays, and combine each of
    # the elements that share the same header together
    pregrasp_dict = {}
    postgrasp_dict = {}
    grasps = zip(decoded['pregrasp'], decoded['postgrasp'])

    for i, (pregrasp, postgrasp) in enumerate(grasps):

        # Allocate a matrix for pregrasp + postgrasp
        if i == 0:
            for key in pregrasp.keys():
                key_size = pregrasp[key].shape[1]
                pregrasp_dict[key] = np.empty((len(grasps), key_size))
                postgrasp_dict[key] = np.empty((len(grasps), key_size))

        # fill it
        for key in pregrasp.keys():
            pregrasp_dict[key][i] = pregrasp[key]
            postgrasp_dict[key][i] = postgrasp[key]

    return {'pregrasp':pregrasp_dict, 'postgrasp':postgrasp_dict}


def postprocess(data, object_name):
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

    # ------------------- Clean the dataset --------------------------

    # -- Remove duplicate grasps using workspace --> camera frame
    unique = get_unique_idx(data['pregrasp']['work2grasp'], -1, 1e-5)
    data = remove_from_dataset(data, unique)

    if data['pregrasp']['work2grasp'].shape[0] > 50:

        # -- Remove any super wild grasps
        good_indices = get_outlier_mask(data['pregrasp']['work2grasp'], sigma=4)
        data = remove_from_dataset(data, good_indices)

        if data['pregrasp']['work2grasp'].shape[0] == 0:
            return

    # Make sure we have the same number of samples for all data elements
    keys = data['pregrasp'].keys()
    pregrasp_size = data['pregrasp']['work2grasp'].shape[0]
    postgrasp_size = data['postgrasp']['work2grasp'].shape[0]

    assert all(pregrasp_size == data['pregrasp'][k].shape[0] for k in keys)
    assert all(postgrasp_size == data['postgrasp'][k].shape[0] for k in keys)

    # ------------------- Save the dataset --------------------------
    save_path = os.path.join(config_processed_data_dir, object_name+'.hdf5')

    datafile = h5py.File(save_path, 'w')
    datafile.create_dataset('object_name', data=[object_name]*postgrasp_size,
                            compression='gzip')

    # Add the pregrasp to the dataset
    name_array = [object_name]*postgrasp_size,
    grasp_group = datafile.create_group('pregrasp')
    for key in data['pregrasp'].keys():
        grasp_group.create_dataset(key, data=data['pregrasp'][key],
                                   compression='gzip')
    grasp_group.create_dataset('object_name', data=name_array)

    # Add the postgrasp to the dataset
    grasp_group = datafile.create_group('postgrasp')
    for key in data['postgrasp'].keys():
        grasp_group.create_dataset(key, data=data['postgrasp'][key],
                                   compression='gzip')
    grasp_group.create_dataset('object_name', data=name_array)

    datafile.close()

    print 'Number of objects: ', postgrasp_size


def merge_files(directory):
    """Merges all files within a directory.

    This is used to join all the trials for a single given object, and assumes
    that all files have the same number of variables.
    """

    data = []
    for object_file in os.listdir(directory):

        if not '.txt' in object_file or object_file == 'commands':
            continue

        # Open the datafile, and find the number of fields
        object_path = os.path.join(directory, object_file)
        content_file = open(object_path, 'r')
        reader = csv.reader(content_file, delimiter=',')
        for line in reader:
            data.append(line)
        content_file.close()
    return data


def main():
    """Performs post-processing on grasps collected during simulation

    Notes
    -----
    This is a pretty beefy file, that does a lot of things. The data should be
    saved by the simulator in a structure similar to:
    class_objectName (folder for a specific object)
       |-> Grasp attempts 1:N (file)
       |-> Grasp attempts N:M (file)
       ...
       |-> Grasp attempts M:P (file)
    """

    if not os.path.exists(config_processed_data_dir):
        os.makedirs(config_processed_data_dir)

    # If we call the file just by itself, we assume we're going to perform
    # processing on each of objects tested during simulation.
    # Else, pass in a specific object/folder name, which can be found in
    # collected_dir
    if len(sys.argv) == 1:
        object_directory = os.listdir(config_collected_data_dir)
    else:
        object_directory = [sys.argv[1].split(os.path.sep)[-1]]
    num_objects = len(object_directory)


    for i, object_name in enumerate(object_directory):

        #try:
            print 'Processing object %d/%d: %s'%(i, num_objects, object_name)
            direct = os.path.join(config_collected_data_dir, object_name)

            # Path to .txt file and hdf5 we want to save
            save_path = os.path.join(config_processed_data_dir, object_name+'.hdf5')
            if os.path.exists(save_path):
                os.remove(save_path)

            # Open up all individual files, merge them into a single file
            merged_data = merge_files(direct)

            decoded = decode_raw_data(merged_data)

            # Check if the decoding returned successfully
            if isinstance(decoded, dict):
                postprocess(decoded, object_name)

        #except Exception as e:
        #    print 'Exception occurred: ', e

if __name__ == '__main__':
    main()
