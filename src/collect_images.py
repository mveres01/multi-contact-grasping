import os
import sys
sys.path.append('..')
import glob
import time
import h5py
import signal
import itertools
import subprocess
import itertools
import numpy as np
import cPickle as pickle

import lib
from lib.python_config import (config_simulation_path,
                               config_dataset_path,
                               config_mesh_dir,
                               project_dir)

import vrep
vrep.simxFinish(-1) # just in case, close all opened connections

import simulation as SI

sim = SI.SimulatorInterface(port=19999)


query_params = {'rgb_near_clip':0.01,
                'depth_near_clip':0.01,
                'rgb_far_clip':10.,
                'depth_far_clip':1.25,
                'camera_fov':70*np.pi/180,
                'resolution':256,
                'p_light_off':0.25,
                'p_light_mag':0.1,
                'reorient_up':True,
                'randomize_texture':True,
                'randomize_colour':True,
                'randomize_lighting':True,
                'texture_path':os.path.join(project_dir, 'texture.png')}

pose_params = {'local_rot':(10, 10, 10),
               'global_rot':(50, 50, 50),
               'base_offset':-0.4,
               'offset_mag':0.4}

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
        for p in data.keys():
            if p not in props:
                props[p] = []
            props[p].append(np.vstack(data[p][:]))

        name = np.atleast_2d([obj])
        name = np.repeat(name, len(data['frame_work2palm']), axis=0)
        props['object_name'].append(name)

    grasps = np.vstack(props['grasp'])
    del props['grasp']

    # Merge the list-of-lists for each property into single arrays
    idx = np.arange(grasps.shape[0])
    if shuffle is True:
        np.random.shuffle(idx)

    grasps = grasps[idx]
    for key in props.keys():
        props[key] = np.vstack(props[key])[idx]
    return grasps, props


def is_valid_image(mask, num_pixel_thresh=400):

    where_object = np.vstack(np.where(mask == 0)).T
    if len(where_object) == 0:
        return False

    min_row, min_col = np.min(where_object, axis=0)
    max_row, max_col = np.max(where_object, axis=0)

    if min_row == 0 or min_col == 0:
        return False
    elif max_row == mask.shape[0] -1:
        return False
    elif max_col == mask.shape[1] - 1:
        return False
    return len(where_object) >= num_pixel_thresh


def get_minibatch(grasps, props, index, num_views):
    """Performs multithreading to query V-REP simulations for image + grasps."""

    q_non_random = query_params.copy()
    for key in q_non_random:
        if 'randomize' in key:
            q_non_random[key] = False
    q_non_random['reorient_up'] = False


    sim_images_reg, sim_images_grasp, sim_grasps, sim_work2cam = [], [], [], []

    # TODO: A little hacky. Try to clean this up so the full path is specified,
    # or object with full extension is given in dataset.
    base_path = os.path.join(config_mesh_dir, props['object_name'][index, 0])
    object_path = str(glob.glob(base_path + '*')[0])

    # These properties don't matter too much since we're not going to be
    # dynamically simulating the object
    frame_world2work = props['frame_world2work'][index]
    frame_work2obj = props['frame_work2obj'][index]
    frame_work2palm = props['frame_work2palm'][index]
    frame_work2palm = lib.utils.format_htmatrix(frame_work2palm)

    mass = props['mass_workspace_wrt_world'][index]
    com = props['com_workspace_wrt_world'][index]
    inertia = props['inertia_workspace_wrt_world'][index]

    sim.load_object(object_path, com, mass, inertia)

    sim.set_object_pose(frame_work2obj)

    # Collect images of the object by itself, and the pre-grasp that was applied.
    # Set the gripper pose (won't change), but toggle between being visible / invisible
    sim.set_gripper_pose(frame_work2palm)
    for key in props.keys():
        if 'joint' not in key:
            continue
        pos = float(props[key][index, 0])
        name = str(key.split('_pos')[0])
        sim.set_joint_position_by_name(name, pos)
    sim.set_gripper_properties(visible=False, renderable=False, dynamic=False)

    # For each successful grasp, we'll do a few randomizations of camera / obj
    for count in xrange(num_views):

        sim.set_gripper_properties(visible=False, renderable=False, dynamic=False)

        while True:

            # We'll use the pose of where the grasp succeeded from as an
            # initial seedpoint for collecting images. For each image, we
            # slightly randomize the cameras pose.
            frame_work2cam = lib.utils.randomize_pose(frame_work2palm, **pose_params)

            if frame_work2cam[11] <= 0.2:
                continue

            # Query simulator for an image & return camera pose
            image, frame_work2cam_ht = \
                sim.query(frame_work2cam, frame_world2work, **query_params)

            if image is None:
                raise Exception('No image returned.')
            if is_valid_image(image[0, 4], num_pixel_thresh=600):

                # Query simulator for an image & return camera pose
                sim.set_gripper_properties(visible=True, renderable=True, dynamic=False)

                image2, _ = sim.query(frame_work2cam_ht[:3].flatten(),
                                      props['frame_world2work'][index],
                                      **q_non_random)


                from scipy import misc
                _, channels, num_rows, num_cols = image.shape
                fig = np.zeros((num_rows, num_cols*2, 3))
                fig[:, :num_cols] = image[0, :3].transpose(1, 2, 0)
                fig[:, num_cols:] = image2[0, :3].transpose(1, 2, 0)

                save_path = 'grasp_images'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                name = str(props['object_name'][index, 0])
                name = '%d_%d_%s.png'%(index, count, name)
                save_path = os.path.join(save_path, name)
                misc.imsave(save_path, fig)

                break

        frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
        grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasps[index])
        time.sleep(0.1)

        sim_grasps.append(np.float32(grasp))
        sim_images_reg.append(np.float16(image))
        sim_images_grasp.append(np.float16(image2))
        sim_work2cam.append(np.float32(frame_work2cam_ht[:3].flatten()))

    return (np.vstack(sim_images_reg), np.vstack(sim_images_grasp),
            np.vstack(sim_grasps), np.vstack(sim_work2cam) )


def create_dataset(inputs, input_props, num_views, dataset_name):
    """Collects images from sim & creates dataset for doing ML."""

    f = h5py.File(dataset_name, 'w')

    # Initialize some structures for holding the dataset
    res = query_params['resolution']

    im_shape = (inputs.shape[0] * num_views, 5, res, res)
    gr_shape = (inputs.shape[0] * num_views, inputs.shape[1])
    mx_shape = (inputs.shape[0] * num_views, 12)

    dset_im_reg = f.create_dataset('images', im_shape, dtype='float16')
    dset_im_grasp = f.create_dataset('images_gripper', im_shape, dtype='float16')
    dset_gr = f.create_dataset('grasps', gr_shape)

    group = f.create_group('props')
    for key in input_props.keys():
        num_var = input_props[key].shape[1]
        if key == 'object_name':
            dt = h5py.special_dtype(vlen=unicode)
            group.create_dataset(key, (inputs.shape[0] * num_views, 1), dtype=dt)
        else:
            group.create_dataset(key, (inputs.shape[0] * num_views, num_var))
    group.create_dataset('frame_work2cam', mx_shape)



    # Loop through our collected data, and query the sim for an image and
    # grasp transformed to the camera's frame
    save_indices = np.arange(len(inputs) * num_views)
    np.random.shuffle(save_indices)

    for idx in range(0, len(inputs)):

        print idx, ' / ', len(inputs)

        # Query simulator for data
        q_images_reg, q_images_grasp, q_grasps, q_work2cam = \
            get_minibatch(inputs, input_props, idx, num_views)

        # We save the queried information in a "shuffled" manner, so grasps
        # for the same object are spread throughout the dataset
        low = idx * num_views
        high = (idx + 1) * num_views

        for i, save_i in enumerate(save_indices[low:high]):
            dset_im_reg[save_i] = q_images_reg[i]
            dset_im_grasp[save_i] = q_images_grasp[i]
            dset_gr[save_i] = q_grasps[i]

            for key in input_props.keys():
                group[key][save_i] = input_props[key][idx]
            group['frame_work2cam'][save_i] = q_work2cam[i]

    f.close()


def get_equalized_idx(name_array, max_samples=250, idx_mask=None):

    freq = {object_:np.sum(names == object_) for object_ in np.unique(names)}

    copy_idx = []
    for object_ in freq:

        choices = np.where(names == object_)[0]

        if idx_mask is not None:
            choices = [c for c in choices if c not in idx_mask]

        if len(choices) < max_samples:
            choices = np.random.choice(choices, max_samples, True)
        else:
            choices = np.random.choice(choices, max_samples, False)

        copy_idx.extend(choices)
    return np.asarray(copy_idx)



if __name__ == '__main__':

    np.random.seed(1234)
    batch_views = 10
    max_samples = 50
    max_valid_samples = 10
    n_test_objects = 2
    shuffle_data = False

    f = h5py.File(config_dataset_path, 'r')

    object_keys = [k for k in f.keys() if 'box' in k]
    train_keys = object_keys[:-n_test_objects]
    test_keys = object_keys[-n_test_objects:]


    train_grasps, train_props = load_subset(f, train_keys, shuffle_data)

    names = train_props['object_name']

    valid_idx = get_equalized_idx(names, max_samples=max_valid_samples)
    y_valid = train_grasps[valid_idx]
    props_valid = {p: train_props[p][valid_idx] for p in train_props}

    #create_dataset(y_valid, props_valid, batch_views, 'collectValid256.hdf5')


    train_idx = get_equalized_idx(names, max_samples=max_samples, idx_mask=valid_idx)
    y_train = train_grasps[train_idx]
    props_train = {p: train_props[p][train_idx] for p in train_props}

    create_dataset(y_train, props_train, batch_views, 'collectTrain256.hdf5')


    test_grasps, test_props = load_subset(f, test_keys, shuffle_data)
    names = test_props['object_name']
    equal_idx = get_equalized_idx(names, max_samples=max_samples)

    test = y_test[equal_idx]
    props = {p:test_props[p][equal_idx] for p in test_props}

    create_dataset(test, props, batch_views, 'collectTest256.hdf5')
