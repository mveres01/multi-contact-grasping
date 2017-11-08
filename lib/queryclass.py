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
from lib.utils import format_htmatrix
from lib.python_config import project_dir

from sys import platform

def wait_for_signal(clientID, signal, mode=vrep.simx_opmode_oneshot_wait):
    r = -1
    while r != vrep.simx_return_ok:
        r, data = vrep.simxGetStringSignal(clientID, signal, mode)
    return data

def rand_step(max_angle):
    """Returns a random point between (-max_angle, max_angle) in radians."""
    if max_angle is None or max_angle <= 0:
        return max_angle
    return np.float32(np.random.randint(-max_angle, max_angle))*np.pi/180.


def spherical_rotate(max_rot_degrees):
    """Calculate a random rotation matrix using angle magnitudes in max_rot."""

    assert isinstance(max_rot_degrees, tuple) and len(max_rot_degrees) == 3

    xrot, yrot, zrot = max_rot_degrees

    # Build the global rotation matrix using quaternions
    q_xr = tf.quaternion_about_axis(rand_step(xrot), [1, 0, 0])
    q_yr = tf.quaternion_about_axis(rand_step(yrot), [0, 1, 0])
    q_zr = tf.quaternion_about_axis(rand_step(zrot), [0, 0, 1])

    # Multiply global and local rotations
    rotation = tf.quaternion_multiply(q_xr, q_yr)
    rotation = tf.quaternion_multiply(rotation, q_zr)
    return tf.quaternion_matrix(rotation)


def randomize_pose(frame_work2pose, base_offset=0., offset_mag=0.01,
                   local_rot=None, global_rot=None, min_dist=0.4):
    """Computes a random pose for any frame by varying position + orientation.

    Given an initial frame of WRT the workspace, we choose a random offset
    along the local z-direction according to offset_mag. The frame is then
    rotated according to random local and global rotations, by sampling
    (x, y, z) values specified in local/global_rot.
    """

    if frame_work2pose is None:
        frame = np.eye(4)
    elif frame_work2pose.ndim == 1:
        frame = lib.utils.format_htmatrix(frame_work2pose)
    else:
        frame = frame_work2pose

    # Perform a local translation of the camera using a random offset
    offset = base_offset
    if offset_mag != 0:
        offset += np.random.uniform(-abs(offset_mag), abs(offset_mag))

    translation_ht = np.eye(4)
    translation_ht[:, 3] = np.array([0, 0, offset, 1])
    translation_ht = np.dot(frame, translation_ht)

    # Calculate local & global rotations using quaternion math
    local_ht = np.eye(4) if local_rot is None else spherical_rotate(local_rot)
    global_ht = np.eye(4) if global_rot is None else spherical_rotate(global_rot)

    # Compute new frame & limit the z-pos to be a minimum height above workspace
    randomized_ht = np.dot(np.dot(global_ht, translation_ht), local_ht)
    randomized_ht[2, 3] = np.maximum(randomized_ht[2, 3], min_dist)

    return randomized_ht[:3].flatten()


def decode_images(float_string, near_clip, far_clip, res_x=128, res_y=128):
    """Decodes a float string containing Depth, RGB, and binary mask info."""

    assert len(float_string) == res_x * res_y * 7, \
        'Image data has length <%d> but expected length <%d>'%\
        (len(float_string, res_x, res_y * 7))

    images = np.asarray(float_string)

    depth = images[:res_x*res_y].reshape(res_y, res_x, 1)
    depth = near_clip + depth*(far_clip - near_clip)

    rgb = images[res_x*res_y:4*res_x*res_y].reshape(res_y, res_x, 3)
    mask = images[4*res_x*res_y:].reshape(res_y, res_x, 3)

    # The image rows are inverted from what the camera sees in simulator
    images = np.float32(np.concatenate([rgb, depth, mask], axis=2))[::-1]
    return images[np.newaxis].transpose(0, 3, 1, 2)


def parse_grasp(header, line):
    """Parses a line of information following size convention in header."""

    line = np.atleast_2d(line)
    split = header.split(',')

    grasp = {}
    current_pos = 0

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

        self._clear_signals()

    def _clear_signals(self, mode=vrep.simx_opmode_oneshot):
        """Clears all signals this script can set in the simulation."""

        vrep.simxClearStringSignal(self.clientID, 'object_resting', mode)
        vrep.simxClearStringSignal(self.clientID, 'run_drop_object', mode)
        vrep.simxClearIntegerSignal(self.clientID, 'run_grasp_attempt', mode)
        vrep.simxClearStringSignal(self.clientID, 'header', mode)
        vrep.simxClearStringSignal(self.clientID, 'pregrasp', mode)
        vrep.simxClearStringSignal(self.clientID, 'postgrasp', mode)

    def _connect(self):
        r = vrep.simxStopSimulation(-1, vrep.simx_opmode_oneshot_wait)

        # Start communication thread
        clientID = vrep.simxStart(self.ip, self.port, True, False, 5000, 5)
        if clientID == -1:
            raise Exception('Unable to connect to address <%s> on port <%d>. ' \
                            'Check that the simulator is currently running.'%\
                            (self.ip, self.port))
        return clientID

    def _islistening(self):
        """Checks whether a program is listening on a port already or not"""

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

    def load_object(self, object_path, com, mass, inertia):
        """Loads an object into the simulator given it's full path.

        This function also sets the initial center of mass, mass, and
        inertia of the object.
        see: http://www.coppeliarobotics.com/helpFiles/en/regularApi/simImportMesh.htm
        """

        if '.obj' in object_path:
            file_format = 0
        elif '.dxf' in object_path:
            file_format = 1
        elif '.3ds' in object_path:
            file_format = 2
        elif '.stl' in object_path: # 3 is regular stl, 4 is binary stl & default
            file_format = 4

        in_floats = []
        in_floats.extend(com)
        in_floats.extend([mass])
        in_floats.extend(inertia)

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'loadObject', [file_format],
             in_floats, [object_path], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error loading object!')

    def set_object_pose(self, frame_work2obj):
        """Sets the pose for the current object to be WRT the workspace frame."""

        if not isinstance(frame_work2obj, list):
            frame_work2obj = frame_work2obj.tolist()

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setObjectPose', [],
             frame_work2obj, [], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting object pose!')

    def get_object_pose(self):
        """Queries the simulator for current object pose WRT the workspace."""

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'getObjectPose', [],
             [], [], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting object pose!')
        return format_htmatrix(np.asarray(r[2]))

    def set_gripper_pose(self, frame_work2palm):
        """Sets the pose for the current object to be WRT the workspace frame."""

        if not isinstance(frame_work2palm, list):
            frame_work2palm = frame_work2palm.tolist()

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setGripperPose', [],
             frame_work2palm, [], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper pose!')

    def set_pose_by_name(self, name, frame_work2pose):
        """Given a name of an object in the scene, set pose WRT to workspace."""

        if not isinstance(frame_work2pose, list):
            frame_work2pose = frame_work2pose.tolist()

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setPoseByName', [],
             frame_work2pose, [name], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting pose for name <%s>!'%name)

    def get_pose_by_name(self, name):
        """Queries the simulator for the pose of object corresponding to <name>.

        The retrieved pose is with respect to the workspace frame.
        """

        emptyBuff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'getPoseByName', [],
             [], [name], emptyBuff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error getting pose for name <%s>!'%name)
        return r[2]

    def query(self, frame_work2cam, frame_world2work,
              base_offset=-0.4, offset_mag=0.15, local_rot=(10, 10, 10),
              global_rot=(30, 30, 30), resolution=256, rgb_near_clip=0.2,
              rgb_far_clip=10.0, depth_far_clip=1.25, depth_near_clip=0.1,
              p_light_off=0.25, p_light_mag=0.1, camera_fov=70*np.pi/180.,
              reorient_up=True, randomize_frame=True, randomize_texture=True,
              randomize_colour=True, randomize_lighting=True,
              texture_path=os.path.join(project_dir, 'texture.png')):
        """Queries the simulator for an image using a camera post WRT workspace.

        The parameters in the signature help define those needed for performing
        domain randomization. Given a camera pose, the simulator samples:
        1. Random light positions
        2. Random number of lights
        3. Object texture / colour
        4. Workspace texture
        """

        # Randomize the camera pose by sampling an offset, local, and global rot
        if randomize_frame:
            frame_work2cam = randomize_pose(frame_work2cam, base_offset,
                                            offset_mag, local_rot, global_rot)

        # Force the camera to always be looking "upwards"
        if reorient_up:
            frame_work2cam = lib.utils.reorient_up_direction( \
                frame_work2cam, frame_world2work, direction_up=[0, 0, 1])

        in_ints = [resolution]
        in_ints.extend([int(randomize_texture)])
        in_ints.extend([int(randomize_colour)])
        in_ints.extend([int(randomize_lighting)])

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

        if r[0] != vrep.simx_return_ok:
            return None, None, None

        images = decode_images(r[2], depth_near_clip, depth_far_clip,
                               res_x=resolution, res_y=resolution)

        # Change the grasp pose to be in the new camera frame
        return images, lib.utils.format_htmatrix(frame_work2cam)

    def run_threaded_drop(self, frame_work2obj):
        """Gets an initial object pose by 'dropping' it WRT frame_work2obj.

        This function launches a threaded script that sets an object position,
        enables the object to be dynamically simulated, and allows it to fall
        to the ground and come to a resting pose. This is usually the first
        step when collecting grasps.
        """

        if not isinstance(frame_work2obj, list):
            frame_work2obj = frame_work2obj.tolist()

        self._clear_signals()

        # Launch the threaded script
        vrep.simxSetStringSignal(self.clientID, 'run_drop_object',
                                 vrep.simxPackFloats(frame_work2obj),
                                 vrep.simx_opmode_oneshot)

        r = -1
        while r != vrep.simx_return_ok:
            r, success = vrep.simxGetIntegerSignal(
                self.clientID, 'object_resting', vrep.simx_opmode_oneshot_wait)
        if success == 0:
            raise Exception('Error dropping object!')

    def run_threaded_candidate(self, finger_angle=0):
        """Launches a threaded scrip in simulator that tests a grasp candidate.

        If the initial grasp has all three fingers touching the objects, then
        values for the pre- and post-grasp will be returned. Otherwise, a {-1}
        will be returned (grasp was attempted, but initial fingers weren't in
        contact with the object).
        """

        # The simulator is going to send these signals back to us, so clear them
        # to make sure we're not accidentally reading old values
        self._clear_signals()

        # Launch the grasp process and wait for a return value
        vrep.simxSetIntegerSignal(self.clientID, 'run_grasp_attempt',
                                  finger_angle, vrep.simx_opmode_oneshot)

        header = wait_for_signal(self.clientID, 'header')
        pregrasp = wait_for_signal(self.clientID, 'pregrasp')
        postgrasp = wait_for_signal(self.clientID, 'postgrasp')

        # Decode the results into a dictionary
        header = header.lstrip('{').rstrip('}')
        if header == '-1':
            return None, None

        pregrasp = parse_grasp(header, vrep.simxUnpackFloats(pregrasp), )
        postgrasp = parse_grasp(header, vrep.simxUnpackFloats(postgrasp))
        return pregrasp, postgrasp

    def view_grasp(self, frame_world2work, frame_work2cam, grasp_wrt_cam, reset_container=0):
        """Plots the contact positions and normals of a grasp WRT camera frame.

        This function first converts the grasp from the camera frame to workspace
        frame, then plots the contact positions and normals. This function
        signature is like this to quickly accomodate grasp predictions made
        by machine learning / neural nets predicting WRT a visual image.
        """

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


    def start(self):
        pass

    def isconnected(self):
        pass

    def isrunning(self):
        pass

    def end(self):
        # Stop simulation
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)

        # End communication thread
        vrep.simxFinish(self.clientID)


if __name__ == '__main__':

    def find_name(to_find):
        from lib.python_config import config_mesh_dir

        meshes = os.listdir(config_mesh_dir)
        for mesh_name in meshes:
            if str(to_find) in mesh_name:
                return mesh_name.split('.')[0]
        return None

    import h5py
    import utils
    import sys
    sys.path.append('..')
    from lib.python_config import config_simulation_path, config_mesh_dir

    GLOBAL_DATAFILE = 'C:/Users/Matt/Documents/grasping-multi-view/learning/valid256.hdf5'

    sim = SimulatorInterface(port=19999)
    #sim.end()


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

        sim.load_object(object_path, com, mass, inertia * 10)

        sim.set_object_pose(props['frame_work2obj'][i])

        images, frame_work2cam_ht = sim.query(props['frame_work2palm'][i],
                                              props['frame_world2work'][i])

        frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
        grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasps[i])


        sim.view_grasp(props['frame_world2work'][i],
                       props['frame_work2cam'][i],
                       grasps[i], reset_container=1)

        sim.set_gripper_pose(props['frame_work2palm'][i])

        pregrasp, postgrasp = sim.run_threaded_candidate()
        if pregrasp is None or postgrasp is None:
            continue
