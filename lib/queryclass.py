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

import lib
import lib.utils
from lib.utils import plot_mesh, format_htmatrix, calc_mesh_centroid, plot_equal_aspect

from sys import platform


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

        # Since we're just query simulation for camera images (for now), there's
        # no reason to use any dynamics / have control over dynamics.
        vrep.simxSynchronous(self.clientID, False)

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
        images = images[::-1]
        return images.transpose(2, 0, 1)

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
        in_floats = []
        in_floats.extend(frame_work2cam)
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
        vrep.simxSynchronousTrigger(self.clientID)

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

    def send_grasp(self, object_name, frame_world2work, frame_work2obj,
                    frame_work2cam, grasp_wrt_cam, reset_container=0):
        """TODO: Calculate an optimized frame_work2palm matrix, and send that
        as well when we want to verify / test grasps.
        """

        # Prepare the inputs; we give it both a camera and object pose
        in_floats = []
        in_floats.extend(frame_world2work)
        in_floats.extend(frame_work2obj)
        in_floats.extend(frame_work2cam)
        in_floats.extend(grasp_wrt_cam)

        in_strings = []
        in_strings.extend([find_name(object_name)])

        in_ints = []
        in_ints.extend([reset_container])

        # Make a call to the simulator
        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'displayGrasp', in_ints,
             in_floats, in_strings, emptyBuff, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.clientID)

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

        # Make a call to the simulator
        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'runGrasp', [],
             frame_work2palm, [], emptyBuff, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.clientID)

        # TODO
        # NOTE: Watch these modes of operation
        # http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm#simxGetStringSignal
        mode = vrep.simx_opmode_streaming
        while not vrep.simxGetStringSignal(self.clientID, 'postgrasp', mode):
            time.sleep(0.1)
            mode = simx_opmode_buffer

        r, header = vrep.simxGetStringSignal(self.clientID,'header', vrep.simx_opmode_streaming)
        r, pregrasp = vrep.simxGetStringSignal(self.clientID,'pregrasp', vrep.simx_opmode_streaming)
        r, postgrasp = vrep.simxGetStringSignal(self.clientID,'postgrasp', vrep.simx_opmode_streaming)

        for signal in ['header', 'pregrasp', 'postgrasp']:
            vrep.simxClearStringSignal(self.clientID, signal, vrep.simx_opmode_oneshot)

        #header = vrep.simxUnpackFloats(header)
        pregrasp = vrep.simxUnpackFloats(pregrasp)
        postgrasp = vrep.simxUnpackFloats(postgrasp)
        return header, pregrasp, postgrasp

    def start(self):
        pass





def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    print 'mesh_path: ', mesh_path
    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh

def generate_candidates(mesh_path, num_samples=1000, noise_level=0.05, gripper_offset=-0.1):
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

        # Since we need to set a pose for the gripper, we need to calculate the
        # rotation matrix from a given surface normal
        matrix = lib.utils.get_rot_mat(up_vector, normal)
        matrix[:3, 3] = p

        # Calculate an offset for the gripper from the object.
        matrix[:3, 3] = np.dot(matrix, np.array([0, 0, gripper_offset, 1]).T)[:3]

        matrices.append(matrix[:3].flatten())

    return np.vstack(matrices)


if __name__ == '__main__':

    import h5py
    import utils
    import sys
    sys.path.append('..')
    from lib.python_config import config_simulation_path, config_mesh_dir

    GLOBAL_DATAFILE = 'C:/Users/Matt/Documents/grasping-multi-view/learning/grasping.hdf5'

    sim = SimulatorInterface(port=19999)

    # Load the data. Note that Grasps are encoded WRT workspace frame
    dataset = h5py.File(GLOBAL_DATAFILE, 'r')
    grasps, props = utils.load_subset(dataset, dataset.keys())

    for i in xrange(len(grasps)):

        com = props['work2com'][i]
        mass = props['work2mass'][i]
        inertia = props['work2inertia'][i]

        object_name = props['object_name'][i, 0] + '.stl'
        object_path = os.path.join(config_mesh_dir, object_name)

        sim.load_object(object_path, com, mass, inertia)

        sim.set_object_pose(props['frame_work2obj'][i])

        sim.query(props['frame_work2palm'][i], props['frame_world2work'][i], grasps[i])


        candidates = generate_candidates(object_path, num_samples=1000,
                                         noise_level=0.05, gripper_offset=-0.1)


        sim.set_object_pose(props['frame_work2obj'][i])
        pose = sim.get_object_pose()

        print pose
        print np.asarray(pose)
        pose = format_htmatrix(np.asarray(pose))

        for row in candidates:

            mult = np.dot(pose, format_htmatrix(row))
            if mult[3, 3] * 10 <= 0.:
                continue
            print 'Object pose: ', pose, 'set pose: ', props['frame_work2obj'][i]

            header, pregrasp, postgrasp = sim.run_candidate(row)

            print header, pregrasp, postgrasp




    misc_params = {'rgb_near_clip':0.2,
                   'depth_near_clip':0.2,
                   'rgb_far_clip':10.,
                   'depth_far_clip':1.25,
                   'camera_fov':75*np.pi/180,
                   'resolution':256,
                   'base_offset':-0.4,
                   'offset_mag':0.15,
                   'local_rot':(10, 10, 10),
                   'global_rot':(30, 30, 30),
                   'p_light_off':0.25,
                   'p_light_mag':0.1,
                   'reorient_up':True,
                   'randomize':True,
                   'texture_path':'C:/Users/Matt/Documents/grasping-multi-view/texture.png'}




    # Loop over all the grasps, sample num_samples_per views and save image
    for i in xrange(0, grasps.shape[0], 1):

        print 'Sampling images for datapoint %d/%d'%(i, grasps.shape[0])

        p = {key : props[key][i] for key in props}
        p.update(misc_params)

        '''
        p['randomize'] = False
        base_offset = -np.random.uniform(0.4, 0.7)
        offset_mag = 0.1
        local_rot = (30, 30, 30)
        global_rot = (60, 60, 60)
        min_dist = 0.6

        p['frame_work2cam'] = randomize_pose(None, base_offset, offset_mag,
                                             local_rot, global_rot, min_dist)
        '''
        p['frame_work2cam'] = p['frame_work2palm']

        _, image, _ = sim.query(grasps[i], **p)

        unique = np.unique(image[0, 4])
        for u in unique:
            print u, np.sum(image[0, 4] == u)

        if image is None:
            raise Exception('No image returned.')

        #import matplotlib.pyplot as plt
        #plt.imshow(image[0, :3].transpose(1, 2, 0))
        #plt.show()
        time.sleep(2)
