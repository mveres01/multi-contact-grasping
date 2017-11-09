import os
import sys
import csv
import time
import h5py
import signal
import itertools
import subprocess
import itertools
import Queue
import threading

import numpy as np
import cPickle as pickle

from multiprocessing import Lock
from query import SimulatorInterface
from utils import *

from lib.python_config import (config_simulation_path, config_dataset_path,
                               project_dir)

import vrep
vrep.simxFinish(-1) # just in case, close all opened connections

import simulator as SI


sim = SI.SimulatorInterface(port=19998)


misc_params = {'rgb_near_clip':0.2,
               'depth_near_clip':0.2,
               'rgb_far_clip':10.,
               'depth_far_clip':1.25,
               'camera_fov':70*np.pi/180,
               'resolution':128,
               'local_rot':(10, 10, 10),
               'global_rot':(30, 30, 30),
               'p_light_off':0.25,
               'p_light_mag':0.1,
               'reorient_up':True,
               'randomize_texture':True,
               'randomize_colour':True,
               'randomize_lighting':True,
               'texture_path':os.path.join(project_dir, 'texture.png')}


def load_subset(h5_file, object_keys, shuffle=True):
    """Loads a subset of an hdf5 file given top-level keys.
    
    The hdf5 file was created using object names as the top-level keys.
    Loading a subset of this file means loading data corresponding to 
    specific objects.
    """

    if not isinstance(object_keys, list):
        object_keys = [object_keys]

    # Load the grasp properties.
    props = {}
    props['object_name'] = []

    for obj in object_keys:

        data = h5_file[obj]['pregrasp']

        # Since there are multiple properties per grasp, we'll do something
        # ugly and store intermediate results into a list, and merge all the
        # properties after we've looped through all the objects.
        for prop in data.keys():

            if prop not in props:
                props[prop] = []
            data = np.atleast_2d(data[prop])

            props[prop].append(data.astype(np.float32))
        
        name = np.atleast_2d([obj], dtype=str)
        name = np.repeat(name, len(data['frame_work2palm']), axis=0)
        props['object_name'].append(name)

    grasps = props['grasps'][:]
    del props['grasps']


    # Merge the list-of-lists for each property into single arrays
    idx = np.arange(grasps.shape[0])
    if shuffle is True:
        np.random.shuffle(idx)

    grasps = grasps[idx]
    for key in props.keys():
        if key == 'object_name':
            props[key] = (np.hstack(props[key]).T)[idx]
        else:
            props[key] = np.vstack(props[key])[idx]

    return grasps, props


def load_dataset(fname, n_train=-5, shuffle=True):
    """Loads the dataset. For now, we split 'train' into train + validation."""

    f = h5py.File(fname, 'r')

    object_keys = f.keys()
    train_keys = object_keys[:n_train]
    valid_keys = object_keys[n_train:]

    train_grasps, train_props = load_subset(f, train_keys, shuffle)
    valid_grasps, valid_props = load_subset(f, valid_keys, shuffle)

    return train_grasps, train_props, valid_grasps, valid_props


def get_minibatch(grasps, props, indices, num_views):
    """Performs multithreading to query V-REP simulations for image + grasps."""

    base_offset = -0.4
    offset_mag = 0.15
    local_rot = (10, 10, 10)
    global_rot = (30, 30, 30)

    sim_images, sim_grasps, sim_work2cam = [], [], []
    for idx in indices:

        for _ in xrange(num_views):

            count, maxcount = 0, 5
            while True:

                frame_work2palm = lib.utils.format_htmatrix(props['frame_work2palm'][i])

                # Compute a random camera pose in vicinity of collected grasp
                frame_work2cam = lib.utils.randomize_pose(frame_work2palm, 
                    base_offset, offset_mag, local_rot, global_rot)

                # Query simulator for an image & return camera pose
                image, frame_work2cam_ht = sim.query(frame_work2cam,
                                                     props['frame_work2work'][idx])

                # Check if an image and grasp were returned
                if image is None:
                    count = count + 1
                    if count >= maxcount:
                        raise Exception('No image or grasp returned')
                    continue

                frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
                grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasps[idx])

                # If an image / grasp WAS returned, check there's a good
                # chunk of an 'object' in it
                num_obj_pixels = np.sum(image[0, 4] == 0)
                print props['object_name'][idx], ' sum: ', num_obj_pixels
                if num_obj_pixels >= 400: 
                    break

                time.sleep(0.1)

            sim_grasps.append(np.float32(grasp))
            sim_images.append(np.float32(image))
            sim_work2cam.append(np.float32(frame_work2cam_ht[:3].flatten()))

    return np.vstack(sim_images), np.vstack(sim_grasps), np.vstack(sim_work2cam)
    
   
def create_dataset(inputs, input_props, num_views, dataset_name, res=256):
    """Collects images from sim & creates dataset for doing ML."""

    f = h5py.File(dataset_name, 'w')

    # Initialize some structures for holding the dataset
    im_shape = (inputs.shape[0] * num_views, 5, res, res)
    gr_shape = (inputs.shape[0] * num_views, inputs.shape[1])
    mx_shape = (inputs.shape[0] * num_views, 12)

    dset_im = f.create_dataset('images', im_shape, dtype='float16')
    dset_gr = f.create_dataset('grasps', gr_shape)
    
    group = f.create_group('props')
    for key in input_props.keys():
        num_var = input_props[key].shape[1]

        if key == 'object_name':
            group.create_dataset(key, (inputs.shape[0] * num_views, 1), dtype='S10')
        else:
            group.create_dataset(key, (inputs.shape[0] * num_views, num_var))
    group.create_dataset('frame_work2cam', mx_shape)



    # Loop through our collected data, and query the sim for an image and 
    # grasp transformed to the camera's frame
    batch_size = 1
    indices = np.arange(len(inputs))

    
    save_indices = np.arange(len(inputs) * num_views)
    np.random.shuffle(save_indices)

    for start_idx in range(0, len(indices) - batch_size + 1, batch_size):

        print start_idx, ' / ', len(inputs)

        excerpt = indices[start_idx:start_idx + batch_size]

        # Query simulator for data
        q_images, q_grasps, q_work2cam = \
            get_minibatch(inputs, input_props, excerpt, num_views)

        # We save the queried information in a "shuffled" manner, so grasps 
        # for the same object are spread throughout the dataset
        low = start_idx * num_views
        high = (start_idx + batch_size) * num_views       
        save_idx = sorted(save_indices[low:high])

        dset_im[save_idx] = q_images
        dset_gr[save_idx] = q_grasps

        for key in input_props.keys():
            repeat = np.repeat(input_props[key][excerpt], num_views, axis=0)

            if key == 'object_name':
                repeat = repeat.astype(str)
            group[key][save_idx] = repeat
        group['frame_work2cam'][save_idx] = q_work2cam

    f.close()


def get_equalized_idx(name_array, max_samples=250, idx_mask=None):

    freq = {object_:np.sum(names == object_) for object_ in np.unique(names)}
   
    copy_idx = []
    for object_ in freq:

        choices = np.where(names == object_)[0]
       
        if idx_mask is not None:
            choices = [c for c in choices if c not in idx_mask]

        if freq[object_] < max_samples:
            choices = np.random.choice(choices, max_samples, True)
        else:
            choices = np.random.choice(choices, max_samples, False)

        copy_idx.extend(choices)
    return np.asarray(copy_idx) 


if __name__ == '__main__':

    np.random.seed(1234)
    batch_views = 10
    max_samples = 250
    max_valid_samples = 10

    print 'Loading dataset ... '
    y_train, train_props, y_test, test_props = \
        load_dataset(config_dataset_path, shuffle=False)

    names = train_props['object_name']


    valid_idx = get_equalized_idx(names, max_samples=max_valid_samples)
    y_valid = y_train[valid_idx]
    props_valid = {p:train_props[p][valid_idx] for p in train_props}

    try:
        create_dataset(y_valid, props_valid, batch_views, 'collectValid256.hdf5')
    except Exception as e:
        print e
        pass


    train_idx = get_equalized_idx(names, max_samples=max_samples, idx_mask=valid_idx)
    y_train = y_train[train_idx]
    props_train = {p:train_props[p][train_idx] for p in train_props}

    try:
        create_dataset(y_train, props_train, batch_views, 'collectTrain256.hdf5')
    except Exception as e:
        print e
        pass


    names = test_props['object_name']
    equal_idx = get_equalized_idx(names, max_samples=max_samples)

    test = y_test[equal_idx]
    props = {p:test_props[p][equal_idx] for p in test_props}

    try:
        create_dataset(test, props, batch_views, 'collectTest256.hdf5')
    except Exception as e:
        print e
        pass

