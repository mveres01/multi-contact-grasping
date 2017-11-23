import os
import sys
import glob
import h5py
import numpy as np
from PIL import Image
from scipy import misc

sys.path.append('..')
import lib
from lib.config import (config_output_collected_dir,
                        config_output_processed_dir,
                        config_mesh_dir, project_dir)
from lib import vrep
from postprocess import postprocess
import simulator as SI

vrep.simxFinish(-1)

def save_images(images, images_gripper, postfix, save_dir):
    """Saves the queried images to disk."""

    name = os.path.join(save_dir, postfix + '_rgb.jpg')
    misc.imsave(name, np.uint8(images[0, :3].transpose(1, 2, 0) * 255))

    # To write the depth info, we'll save it as a 16bit float via numpy
    name = os.path.join(save_dir, postfix + '_depth')
    np.save(name, np.float16(images[0, 3]), False, True)

    name = os.path.join(save_dir, postfix + '_mask.jpg')
    misc.imsave(name, np.uint8(images[0, 4] * 255))

    name = os.path.join(save_dir, postfix + '_gripper.jpg')
    misc.imsave(name, np.uint8(images_gripper[0, :3].transpose(1, 2, 0) * 255))


def is_valid_image(mask, num_pixel_thresh=400):
    """Checks the amount of object in the image & its within all bounds."""

    where_object = np.vstack(np.where(mask == 0)).T
    if len(where_object) == 0:
        return False

    min_row, min_col = np.min(where_object, axis=0)
    max_row, max_col = np.max(where_object, axis=0)

    if min_row == 0 or min_col == 0:
        return False
    elif max_row == mask.shape[0] - 1:
        return False
    elif max_col == mask.shape[1] - 1:
        return False
    return len(where_object) >= num_pixel_thresh


def query_minibatch(pregrasp, index, num_views, object_name):
    """Queries simulator for an image and formats grasp to be WRT camera."""

    save_dir = os.path.join(config_output_processed_dir, object_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Used to take a duplicate image of the current scene
    q_non_random = query_params.copy()
    q_non_random['randomize_lighting'] = False
    q_non_random['randomize_texture'] = False
    q_non_random['randomize_colour'] = False
    q_non_random['reorient_up'] = False

    # These properties don't matter too much since we're not going to be
    # dynamically simulating the object, but are needed as placeholders for
    # loading the object
    mass = pregrasp['mass_wrt_world'][index]
    com = pregrasp['com_wrt_world'][index]
    inertia = pregrasp['inertia_wrt_world'][index]

    grasp = np.hstack([pregrasp['work2contact0'][index],
                       pregrasp['work2contact1'][index],
                       pregrasp['work2contact2'][index],
                       pregrasp['work2normal0'][index],
                       pregrasp['work2normal1'][index],
                       pregrasp['work2normal2'][index]])

    frame_world2work = pregrasp['frame_world2work'][index]
    frame_work2palm = pregrasp['frame_work2palm'][index]
    frame_work2palm = lib.utils.format_htmatrix(frame_work2palm)

    # TODO: A little hacky. Try to clean this up so the full path is specified,
    # or object with full extension is given in dataset.
    base_path = os.path.join(config_mesh_dir, object_name)
    object_path = str(glob.glob(base_path + '*')[0])

    sim.load_object(object_path, com, mass, inertia)

    sim.set_object_pose(pregrasp['frame_work2obj'][index])

    # Collect images of the object by itself, and the pre-grasp that was used.
    # Set gripper pose & toggle between being visible / invisible across views
    sim.set_gripper_pose(pregrasp['frame_work2palm'][index])
    for key in pregrasp.keys():
        if 'joint' not in key:
            continue
        pos = float(pregrasp[key][index, 0])
        sim.set_joint_position_by_name(str(key), pos)


    grasp_list, frame_work2cam_list, name_list = [], [], []

    # For each successful grasp, we'll do a few randomizations of camera / obj
    for count in xrange(num_views):

        # Toggle the gripper to be invisible / visible to the camera, to get
        # an image of the object with & without pregrasp pose
        sim.set_gripper_properties()

        while True:

            # We'll use the pose of where the grasp succeeded from as an
            # initial seedpoint for collecting images. For each image, we
            # slightly randomize the cameras pose, but make sure the camera is
            # always above the tabletop
            frame_work2cam = lib.utils.randomize_pose(frame_work2palm, **pose_params)

            if frame_work2cam[11] <= 0.2:
                continue

            # Take an image of the object
            image_wo_gripper, frame_work2cam_ht = \
                sim.query(frame_work2cam, frame_world2work, **query_params)

            if image_wo_gripper is None:
                raise Exception('No image returned.')
            elif is_valid_image(image_wo_gripper[0, 4], num_pixel_thresh=600):
                break

        # Take an image of the grasp that was used on the object
        sim.set_gripper_properties(visible=True, renderable=True)

        image_w_gripper, _ = sim.query(frame_work2cam_ht[:3].flatten(),
                                       frame_world2work, **q_non_random)

        postfix = '%d_%d' % (index, count)
        save_images(image_wo_gripper, image_w_gripper, postfix, save_dir)

        # Conver the grasp contacts / normals from workspace to camera frame
        frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
        grasp_wrt_cam = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasp)

        name_list.append(postfix)
        grasp_list.append(np.float32(grasp_wrt_cam))
        frame_work2cam_list.append(np.float32(frame_work2cam_ht[:3].flatten()))

    return (np.vstack(grasp_list), np.vstack(frame_work2cam_list),
            np.hstack(name_list))


def collect_images(file_name, input_dir, output_dir, num_views):
    """Collects images from sim."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = h5py.File(os.path.join(input_dir, file_name), 'r')
    pregrasp, postgrasp = postprocess(f['pregrasp'], f['postgrasp'])

    if pregrasp is None or postgrasp is None:
        print('No data in %s' % file_name)
        return
    num_samples = len(pregrasp[pregrasp.keys()[0]])


    out_file = h5py.File(os.path.join(output_dir, file_name), 'w')

    pregrasp_group = out_file.create_group('pregrasp')
    postgrasp_group = out_file.create_group('postgrasp')

    dt = h5py.special_dtype(vlen=unicode)
    for key in pregrasp.keys():
        num_var = pregrasp[key].shape[1]
        pregrasp_group.create_dataset(key, (num_samples * num_views, num_var))
        postgrasp_group.create_dataset(key, (num_samples * num_views, num_var))

    # Collecting this from the sim
    pregrasp_group.create_dataset('grasp_wrt_cam', (num_samples * num_views, 18))
    pregrasp_group.create_dataset('image_name', (num_samples * num_views, ), dtype=dt)
    pregrasp_group.create_dataset('frame_work2cam', (num_samples * num_views, 12))


    # Start collection
    object_name = file_name.split('.')[0]
    for i in xrange(num_samples):

        print('Querying for image set %d / %d ' % (i, num_samples))

        low = i * num_views
        high = (i + 1) * num_views

        for key in pregrasp.keys():
            pregrasp_group[key][low:high] = np.repeat(pregrasp[key][i:i+1],
                                                      num_views, axis=0)
            postgrasp_group[key][low:high] = np.repeat(postgrasp[key][i:i+1],
                                                       num_views, axis=0)

        # Query the simulator for some images; note that we're only collecting
        # this info for the pregrasp here.
        grasp_wrt_cam, frame_work2cam, image_names = \
            query_minibatch(pregrasp, i, num_views, object_name)

        pregrasp_group['grasp_wrt_cam'][low:high] = grasp_wrt_cam
        pregrasp_group['image_name'][low:high] = image_names
        pregrasp_group['frame_work2cam'][low:high] = frame_work2cam


    f.close()


if __name__ == '__main__':

    num_views_per_sample = 10

    spawn_params = {'port': 19997,
                    'ip': '127.0.0.1',
                    'vrep_path': None,
                    'scene_path': None,
                    'exit_on_stop': True,
                    'spawn_headless': True,
                    'spawn_new_console': True}

    query_params = {'rgb_near_clip': 0.01,
                    'depth_near_clip': 0.01,
                    'rgb_far_clip': 10.,
                    'depth_far_clip': 1.25,
                    'camera_fov': 70 * np.pi / 180,
                    'resolution': 256,
                    'p_light_off': 0.25,
                    'p_light_mag': 0.1,
                    'reorient_up': True,
                    'randomize_texture': True,
                    'randomize_colour': True,
                    'randomize_lighting': True,
                    'texture_path': os.path.join(project_dir, 'texture.png')}

    pose_params = {'local_rot': (10, 10, 10),
                   'global_rot': (50, 50, 50),
                   'base_offset': -0.4,
                   'offset_mag': 0.4}

    # spawn_params['vrep_path'] = 'C:\\Program Files\\V-REP3\\V-REP_PRO_EDU\\vrep.exe'

    if len(sys.argv) == 1:
        sim = SI.SimulatorInterface(**spawn_params)

        np.random.seed(1234)

        data_list = os.listdir(config_output_collected_dir)
        data_list = [d for d in data_list if '.hdf5' in d]

        for h5file in data_list:
            collect_images(h5file, config_output_collected_dir,
                           config_output_processed_dir, num_views_per_sample)

    else:
        spawn_params['port'] = int(sys.argv[1])

        sim = SI.SimulatorInterface(**spawn_params)

        # List of meshes we should run are stored in a file,
        mesh_list_file = sys.argv[2]
        with open(mesh_list_file, 'r') as f:
            while True:
                mesh_path = f.readline().rstrip()

                if mesh_path == '':
                    break

                mesh_name = mesh_path.split(os.path.sep)[-1]
                collect_images(mesh_name, config_output_collected_dir,
                               config_output_processed_dir, num_views_per_sample)
