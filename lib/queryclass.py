import os
import sys
sys.path.append('..')

import subprocess
import csv
import argparse
import time
import h5py
import numpy as np
import trimesh
import trimesh.transformations as tf
import vrep
vrep.simxFinish(-1)

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

import lib
import lib.utils
from lib.utils import plot_mesh, format_htmatrix, calc_mesh_centroid, plot_equal_aspect

from sys import platform
from decode import parse_grasp, decode_grasp


def find_name(to_find):
    from lib.python_config import config_mesh_dir

    meshes = os.listdir(config_mesh_dir)
    for mesh_name in meshes:
        if str(to_find) in mesh_name:
            return mesh_name.split('.')[0]
    return None


def rand_step(max_angle):
    """Returns a random point between (-max_angle, max_angle) in radians."""
    if max_angle is None or max_angle <= 0:
        return max_angle
    return np.float32(np.random.randint(-max_angle, max_angle))*np.pi/180.


def spherical_rotate(max_rot):
    """Performs a local & global spherical rotation of a HT matrix."""

    assert isinstance(max_rot, tuple) and len(max_rot) == 3

    xrot, yrot, zrot = max_rot

    # Build the global rotation matrix using quaternions
    q_xr = tf.quaternion_about_axis(rand_step(xrot), [1, 0, 0])
    q_yr = tf.quaternion_about_axis(rand_step(yrot), [0, 1, 0])
    q_zr = tf.quaternion_about_axis(rand_step(zrot), [0, 0, 1])

    # Multiply global and local rotations
    rotation = tf.quaternion_multiply(q_xr, q_yr)
    rotation = tf.quaternion_multiply(rotation, q_zr)
    return tf.quaternion_matrix(rotation)

def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    print 'mesh_path: ', mesh_path
    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh


def plot_mesh_with_normals(mesh, matrices, direction_vec, normals=None, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_path : path to a given mesh
    workspace2obj : 4x4 transform matrix from the workspace to object
    axis : (optional) a matplotlib axis for plotting a figure
    """

    if isinstance(direction_vec, list):
        dvec = np.atleast_2d(direction_vec).T
    elif direction_vec.ndim == 1:
        dvec = np.atleast_2d(direction_vec).T
    else:
        dvec = direction_vec

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5, 0.5, 1])
    axis.add_collection3d(poly)

    axis = plot_equal_aspect(mesh.vertices, axis)

    for i in xrange(0, len(matrices), 10):

        transform = lib.utils.format_htmatrix(matrices[i])

        # We'll find the direction by finding the vector between two points
        gripper_point = np.hstack([matrices[i, 3], matrices[i, 7], matrices[i, 11]])
        gripper_point = np.atleast_2d(gripper_point)

        direction = np.dot(transform[:3, :3], np.atleast_2d(direction_vec).T)
        direction = np.atleast_2d(direction).T

        # The plotted normals and grasp approach should be overlapping
        if normals is not None:
            a = np.hstack([gripper_point, -np.atleast_2d(normals[i])]).flatten()
            axis.quiver(*a, color='r', length=0.1)

        a = np.hstack([gripper_point, -direction]).flatten()
        axis.quiver(*a, color='k', length=0.1)

        axis.scatter(*gripper_point.flatten(), c='b', marker='o', s=10)

    return axis

def generate_candidates(mesh_path, num_samples=1000, noise_level=0.05,
                        gripper_offset=-0.1, augment=True):
    """Generates grasp candidates via surface normals of the object."""

    # Defines the up-vector for the workspace frame
    up_vector = np.asarray([0, 0, -1])

    mesh = load_mesh(mesh_path)

    #points = sample.sample_surface(mesh, num_samples)
    points = trimesh.sample.sample_surface_even(mesh, num_samples)
    normals, matrices = [], []

    # Find the normals corresponding to the sampled points
    triangles = mesh.triangles_center
    for p in points:

        face_idx = np.argmin(np.sum((p - triangles)**2, axis=1))
        normal = mesh.triangles_cross[face_idx]
        normal = lib.utils.normalize_vector(normal)

        # Add random noise to the surface normals, centered around 0
        if augment is True:
            normal += np.random.uniform(-noise_level, noise_level)
            normal = lib.utils.normalize_vector(normal)

        # Since we need to set a pose for the gripper, we need to calculate the
        # rotation matrix from a given surface normal
        matrix = lib.utils.get_rot_mat(up_vector, normal)
        matrix[:3, 3] = p

        # Calculate an offset for the gripper from the object.
        matrix[:3, 3] = np.dot(matrix, np.array([0, 0, gripper_offset, 1]).T)[:3]

        matrices.append(matrix[:3].flatten())

    '''
    matrices = np.vstack(matrices)
    plot_mesh_with_normals(mesh, matrices, up_vector)
    mesh_name = mesh_path.split(os.path.sep)[-1].split('.')[0]
    plt.title('Mesh: ' + mesh_name)
    plt.show()
    '''
    return np.vstack(matrices)

def randomize_pose(frame_work2cam, base_offset=0., offset_mag=0.01,
                   local_rot=None, global_rot=None, min_dist=0.4):
    """Computes a random offset for the camera using initial gripper pose."""

    if frame_work2cam is None:
        frame = np.eye(4)
    elif frame_work2cam.ndim == 1:
        frame = lib.utils.format_htmatrix(frame_work2cam)
    else:
        frame = frame_work2cam

    # Perform a local translation of the camera using a random offset
    offset_mag = abs(offset_mag)
    offset = base_offset + np.random.uniform(-offset_mag, offset_mag)

    translation_ht = np.eye(4)
    translation_ht[:, 3] = np.array([0, 0, offset, 1])
    translation_ht = np.dot(frame, translation_ht)

    # Calculate global & ocal camera rotations using quaternion math
    if local_rot is None:
        local_ht = np.eye(4)
    else:
        local_ht = spherical_rotate(local_rot)

    if global_rot is None:
        global_ht = np.eye(4)
    else:
        global_ht = spherical_rotate(global_rot)

    randomized_ht = np.dot(np.dot(global_ht, translation_ht), local_ht)

    # Limit the cameras z-position to be a minimum distance above table.
    randomized_ht[2, 3] = np.maximum(randomized_ht[2, 3], min_dist)

    return randomized_ht[:3].flatten()


def spawn_simulation(port, vrep_path, scene_path):
    """Spawns a child process using screen and starts a remote VREP server."""

    if platform not in ['linux', 'linux2']:
        print('Must be running on Linux to use this function.')
        return False

    # Command to launch VREP
    vrep_cmd = '%s -h -q -s -gREMOTEAPISERVERSERVICE_%d_FALSE_TRUE %s'% \
               (vrep_path, port, scene_path)

    # Command to launch VREP + detach from screen
    bash_cmd = 'screen -dmS port%d bash -c "export DISPLAY=:1 ;'\
               'ulimit -n 4096; %s " '%(port, vrep_cmd)

    process = subprocess.Popen(bash_cmd, shell=True)
    time.sleep(1)


class SimulatorInterface(object):

    def __init__(self, port, ip='127.0.0.1', vrep_path=None, scene_path=None):

        if not isinstance(port, int):
            raise Exception('Port <%s> must be of type <int>'%port)
        elif not isinstance(ip, str):
            raise Exception('IP address <%s> must be of type <str>'%ip)

        self.port = port
        self.ip = ip
        self.clientID = None

        # If we're running linux, we can try automatically spawning a child
        # process running the simulator scene
        if platform in ['linux', 'linux2'] and not self._islistening():
            if vrep_path is None:
                print('VREP is not currently running and/or is not listening on '\
                      'port <%s>. To spawn automatically, specify :param: ' \
                      'vrep_path, which is the path to the VREP exe.'%self.port)
                return
            if scene_path is None:
                scene_path = os.path.append('..','scenes','collect_multiview_grasps.ttt')
            if not os.path.exists(scene_path):
                raise Exception('Scene path <%s> not found. Is this right?'%scene_path)
            spawn_simulation(self.port, vrep_path, scene_path)

        self.clientID = self._connect()

        # Tell the scene to start running
        self._start()


    def _connect(self):
        r = vrep.simxStopSimulation(-1, vrep.simx_opmode_oneshot_wait)

        clientID = vrep.simxStart(self.ip, self.port, True, False, 5000, 5)
        if clientID == -1:
            raise Exception('Unable to connect to address <%s> on port <%d>. ' \
                            'Check that the simulator is currently running.'%\
                            (self.ip, self.port))
        return clientID

    def _islistening(self):
        """Checks whether VREP is listening on a port already or not.

        Currently only works with Linux.
        """
        if platform not in ['linux', 'linux2']:
            raise Exception('You must be running Linux to use this function.')

        # Get a list of all current open / connected ports
        try:
            netstat = subprocess.Popen(['netstat','-nao'], stdout=subprocess.PIPE)
        except Exception as e:
            raise e
        ports = netstat.communicate()[0]

        return self.ip + ':' + str(self.port) in ports

    def _start(self):
        """Tells a VREP scene to start execution."""

        if self.clientID is None:
            raise Exception('Remote API server must have been started and ' \
                            'communication begun before running the simulation.')

        r = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        if r != 1:
            raise Exception('Unable to start simulation.')

        vrep.simxSynchronous(self.clientID, False)
        vrep.simxClearStringSignal(self.clientID, 'grasp_candidate', vrep.simx_opmode_oneshot)

    @staticmethod
    def decode_images(float_string, near_clip, far_clip, res_x=128, res_y=128):
        """Decodes a float string containing Depth, RGB, and binary mask info."""

        images = np.asarray(float_string)

        depth = images[:res_x*res_y].reshape(res_y, res_x, 1)
        depth = near_clip + depth*(far_clip - near_clip)

        rgb = images[res_x*res_y : 4*res_x*res_y].reshape(res_y, res_x, 3)
        mask = images[4*res_x*res_y:].reshape(res_y, res_x, 3)

        # The image rows are inverted from what the camera sees in simulator
        images = np.concatenate([rgb, depth, mask], axis=2).astype(np.float32)
        return images[::-1].transpose(2, 0, 1)

    def isconnected(self):
        pass

    def isrunning(self):
        pass

    def end(self):
        vrep.simxFinish(self.clientID)

    def query(self, frame_work2cam, frame_world2work,  grasp,
              base_offset=-0.4, offset_mag=0.15, local_rot=(10, 10, 10),
              global_rot=(30, 30, 30), resolution=256, rgb_near_clip=0.2,
              rgb_far_clip=10.0, depth_far_clip=1.25, depth_near_clip=0.1,
              p_light_off=0.25, p_light_mag=0.1, camera_fov=1.2217,
              reorient_up=True, randomize=True,
              texture_path='C:/Users/Matt/Documents/grasping-multi-view/texture.png',
              ):
        """Queries the simulator for an image using a random camera pose."""

        # Randomize the camera pose by sampling an offset, local, and global rot
        if randomize:
            frame_work2cam = randomize_pose(frame_work2cam, base_offset,
                                            offset_mag, local_rot, global_rot)

        # Force the camera to always be looking "upwards"
        if reorient_up:
            frame_work2cam = lib.utils.reorient_up_direction( \
                frame_work2cam, frame_world2work, direction_up=[0, 0, 1])

        in_ints = [resolution]

        # Prepare the inputs; we give it both a camera and object pose
        in_floats = frame_work2cam.tolist()
        in_floats.extend([p_light_off])
        in_floats.extend([p_light_mag])
        in_floats.extend([rgb_near_clip])
        in_floats.extend([rgb_far_clip])
        in_floats.extend([depth_near_clip])
        in_floats.extend([depth_far_clip])
        in_floats.extend([camera_fov])

        in_strings = [texture_path]

        # Make a call to the simulator
        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'queryCamera', in_ints,
             in_floats, in_strings, emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != 0 :
            return None, None, None

        images = self.decode_images(r[2], depth_near_clip, depth_far_clip,
                                    res_x=resolution, res_y=resolution)
        images = images[np.newaxis]

        # Change the grasp pose to be in the new camera frame
        frame_work2cam_ht = lib.utils.format_htmatrix(frame_work2cam)
        frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
        grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasp)

        return grasp, images, frame_work2cam_ht

    def view_grasp(self, frame_world2work, frame_work2cam, grasp_wrt_cam, reset_container=0):

        # Prepare the inputs; we give it both a camera and object pose
        in_floats = frame_world2work.tolist()
        in_floats.extend(frame_work2cam)
        in_floats.extend(grasp_wrt_cam)

        in_ints = [reset_container]

        # Make a call to the simulator
        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'displayGrasp', in_ints,
             in_floats, [], emptyBuff, vrep.simx_opmode_blocking)
        return r

    def load_object(self, object_path, com, mass, inertia):

        in_floats = []
        in_floats.extend(com)
        in_floats.extend([mass])
        in_floats.extend(inertia)

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'loadObject', [],
             in_floats, [object_path], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != 0:
            raise Exception('Error loading object!')

    def set_object_pose(self, frame_work2obj):

        if not isinstance(frame_work2obj, list):
            frame_work2obj = frame_work2obj.tolist()

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setObjectPose', [],
             frame_work2obj, [], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != 0:
            raise Exception('Error setting object pose!')

    def get_object_pose(self):

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'getObjectPose', [],
             [], [], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != 0:
            raise Exception('Error setting object pose!')
        return r[2] # floats

    def run_candidate(self, frame_work2palm):

        if not isinstance(frame_work2palm, list):
            frame_work2palm = frame_work2palm.tolist()

        for signal in ['header', 'pregrasp', 'postgrasp']:
            vrep.simxClearStringSignal(self.clientID, signal, vrep.simx_opmode_oneshot)

        # Launch the threaded app
        vals = vrep.simxPackFloats(frame_work2palm)
        vrep.simxSetStringSignal(self.clientID, 'grasp_candidate', vals, vrep.simx_opmode_oneshot)

        r, header = vrep.simxGetStringSignal(self.clientID,'header', vrep.simx_opmode_oneshot_wait)
        while r != vrep.simx_return_ok:
            r, header = vrep.simxGetStringSignal(self.clientID,'header', vrep.simx_opmode_oneshot_wait)

        r, pregrasp = vrep.simxGetStringSignal(self.clientID,'pregrasp', vrep.simx_opmode_oneshot_wait)
        while r != vrep.simx_return_ok:
            r, pregrasp = vrep.simxGetStringSignal(self.clientID,'pregrasp', vrep.simx_opmode_oneshot_wait)

        r, postgrasp = vrep.simxGetStringSignal(self.clientID,'postgrasp', vrep.simx_opmode_oneshot_wait)
        while r != vrep.simx_return_ok:
            r, postgrasp = vrep.simxGetStringSignal(self.clientID,'postgrasp', vrep.simx_opmode_oneshot_wait)

        for signal in ['header', 'pregrasp', 'postgrasp']:
            vrep.simxClearStringSignal(self.clientID, signal, vrep.simx_opmode_oneshot)

        header = header.lstrip('{').rstrip('}')
        if header == '-1':
            return None, None
        pregrasp = parse_grasp(vrep.simxUnpackFloats(pregrasp), header)
        postgrasp = parse_grasp(vrep.simxUnpackFloats(postgrasp), header)

        return pregrasp, postgrasp

    def start(self):
        pass





if __name__ == '__main__':

    import h5py
    import utils
    import sys
    sys.path.append('..')
    from lib.python_config import config_simulation_path, config_mesh_dir

    GLOBAL_DATAFILE = 'C:/Users/Matt/Documents/grasping-multi-view/learning/valid256.hdf5'

    sim = SimulatorInterface(port=19999)

    # Load the data. Note that Grasps are encoded WRT workspace frame
    dataset = h5py.File(GLOBAL_DATAFILE, 'r')
    grasps = dataset['grasps']
    props = dataset['props']

    for i in xrange(len(grasps)):

        com = props['work2com'][i]
        mass = props['work2mass'][i]
        inertia = props['work2inertia'][i]

        object_name = find_name(props['object_name'][i, 0]) + '.stl'
        object_path = os.path.join(config_mesh_dir, object_name)

        sim.load_object(object_path, com, mass, inertia)

        sim.set_object_pose(props['frame_work2obj'][i])

        sim.query(props['frame_work2palm'][i], props['frame_world2work'][i], grasps[i])

        sim.view_grasp(props['frame_world2work'][i],
                       props['frame_work2cam'][i],
                       grasps[i], reset_container=1)
        pregrasp, postgrasp = sim.run_candidate(props['frame_work2palm'][i])
        if pregrasp is None or postgrasp is None:
            continue


        '''
        candidates = generate_candidates(object_path, num_samples=1000,
                                         noise_level=0.05,
                                         gripper_offset=-0.15)

        pose = format_htmatrix(np.asarray(sim.get_object_pose()))
        for count, row in enumerate(candidates):

            mult = np.dot(pose, format_htmatrix(row))

            direction = np.dot(mult[:3, :3], np.atleast_2d([0, 0, 1]).T)
            if direction[2] * 10 <= 0.:
                continue
            print count

            print 'Attempting grasp'
            header, pregrasp, postgrasp = sim.run_candidate(row)

            if header == '-1':
                continue

            from decode import parse_grasp, decode_grasp
            parsed = parse_grasp(pregrasp, header)
            decoded = decode_grasp(parsed)

            print 'Attempting original grasp'
            header, pregrasp, postgrasp = sim.run_candidate(decoded['frame_work2palm'][i])

            time.sleep(5)
        '''
