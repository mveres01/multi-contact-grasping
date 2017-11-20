import os
import sys
from sys import platform
from distutils.spawn import find_executable
import subprocess
import time
import numpy as np

sys.path.append('..')

import lib
import lib.utils
from lib.config import project_dir, config_simulation_path
from lib import vrep
vrep.simxFinish(-1)


def wait_for_signal(clientID, signal, mode=vrep.simx_opmode_oneshot_wait):
    """Waits for a signal from V-REP before continuing main client code."""
    r = -1
    while r != vrep.simx_return_ok:
        r, data = vrep.simxGetStringSignal(clientID, signal, mode)
        time.sleep(0.1)
    return data


def decode_images(float_string, near_clip, far_clip, res_x=128, res_y=128):
    """Decodes a float string containing Depth, RGB, and binary mask info.

    Images are encoded as (rows, cols, channels), and when retrieved from the
    simulator have their rows inverted. We undo these transformations and
    return a tensor of shape (1, channels, rows, cols).
    """

    assert len(float_string) == res_x * res_y * 7, \
        'Image data has length <%d> but expected length <%d>' %\
        (len(float_string, res_x, res_y * 7))

    images = np.asarray(float_string)

    depth = images[:res_x * res_y].reshape(res_y, res_x, 1)
    depth = near_clip + depth * (far_clip - near_clip)

    rgb = images[res_x * res_y:4 * res_x * res_y].reshape(res_y, res_x, 3)
    mask = images[4 * res_x * res_y:].reshape(res_y, res_x, 3)[:, :, 0:1]

    # The image rows are inverted from what the camera sees in simulator
    images = np.float32(np.concatenate([rgb, depth, mask], axis=2))[::-1]
    return images[np.newaxis].transpose(0, 3, 1, 2)


def decode_grasp(header, line):
    """Parses a line of information following size convention in header."""

    line = np.atleast_2d(line)
    split = header.split(',')

    grasp = {}
    current_pos = 0

    # Decode all the data into a python dictionary
    for i in range(0, len(split), 2):

        name = str(split[i][1:-1])
        n_items = int(split[i + 1])
        subset = line[:, current_pos:current_pos + n_items]

        try:
            subset = subset.astype(np.float32)
        except Exception:
            subset = subset.astype(str)

        grasp[name] = subset.ravel()
        current_pos += n_items
    return grasp


def spawn_simulation(port, vrep_path, scene_path, exit_on_stop,
                     spawn_headless, spawn_new_console):
    """Spawns a child process using screen and starts a remote VREP server.

    Parameters
    ----------
    port : int
        integer denoting which port to connect to

    vrep_path : a path to the V-REP executable or None.
        If None, attempt to use either an "vrep.exe" (windows) or "vrep.sh"
        (linux) and check that it can be found in PATH.

    scene_path : string
        The path to the scene we wish to load in the simulator

    exit_on_stop : bool
        Whether we want V-REP to exit when the simulation is stopped or not

    spawn_headless : bool
        Whether we want to run vrep using the GUI (True) or without (False)

    spawn_new_console : bool
        Whether we want output from the sim to be on current command line, or
        on a new console. If spawn_headless is True, this flag has no effect.
    """

    using_linux = platform in ['linux', 'linux2']

    # E.g. full path to V-REP executable is specified
    if vrep_path is not None:
        if not os.path.exists(vrep_path):
            raise Exception('Cannot find file <%s>' % vrep_path)
        elif not using_linux:
            vrep_path = '"%s"' % vrep_path
    else:
        vrep_path = 'vrep.sh' if using_linux else 'vrep.exe'

        if find_executable(vrep_path) is None:
            raise Exception('Cannot find %s in PATH to spawn a sim. '
                            'Try specifying full path to executable. ' % vrep_path)

    # Command to launch VREP
    headless_flag = '-h' if spawn_headless else ''
    exit_flag = '-q' if exit_on_stop else ''

    vrep_cmd = '%s %s %s -s -gREMOTEAPISERVERSERVICE_%d_FALSE_FALSE %s' % \
        (vrep_path, headless_flag, exit_flag, port, scene_path)

    if platform in ['linux', 'linux2']:
        vrep_cmd = 'bash -c "export DISPLAY=:1 ; %s " ' % (vrep_cmd)

    print('Using command: \n%s\nto spawn simulation' % vrep_cmd)

    if spawn_new_console and not using_linux:
        cflags = subprocess.CREATE_NEW_CONSOLE
        process = subprocess.Popen(vrep_cmd, shell=True, creationflags=cflags)
    else:    
        process = subprocess.Popen(vrep_cmd, shell=True)
    time.sleep(1)
    return process


class SimulatorInterface(object):
    """Defines an interface for using Python to interact with a VREP simulation.

    Connecting to V-REP can be done through 2 different modes:

    1. Continuous Remote API Service
    -----------------------------
    The continuous remote API tells the simulator to open a persistent
    communication channel on a given port. This persistance allows us to
    interact with the simulator even while a scene may be stopped or paused.

    2. 'Dynamic' Remote API Service
    ----------------------------
    A more dynamic communication scheme can be started by launching VREP,
    and telling it to open / listen on a specific port _while the scene
    is running_. Here, V-REP must be running in order to communicate with
    it, and prevents us from starting a stopped simulation.

    Here we'll mostly take advantage of mode (2).
    """

    def __init__(self, port, ip='127.0.0.1', vrep_path=None, scene_path=None,
                 exit_on_stop=True, spawn_headless=True, spawn_new_console=True):

        if not isinstance(port, int):
            raise Exception('Port <%s> must be of type <int>' % port)
        elif not isinstance(ip, str):
            raise Exception('IP address <%s> must be of type <str>' % ip)

        # See if there's a simulation already listening on this port
        clientID = self._start_communication(ip, port)

        # If there's nothing listening, we can try spawning a sim on linux
        if clientID is None:

            if scene_path is None:
                scene_path = config_simulation_path
            if not os.path.exists(scene_path):
                raise Exception('Scene <%s> not found' % scene_path)

            print('Spawning a Continuous Server on port <%d>' % port)
            spawn_simulation(port, vrep_path, scene_path, exit_on_stop,
                             spawn_headless, spawn_new_console)

            # Try starting communication again
            clientID = self._start_communication(ip, port)

            if clientID is None:
                raise Exception('Unable to connect to address <%s> on port '
                                '<%d>. Check that the simulator is currently '
                                'running, or the continuous service was '
                                'started successfully.' % (ip, port))

        self.port = port
        self.ip = ip
        self.clientID = clientID

        # Tell the scene to start running
        self._start_simulation()

        # Remove All previous signals in the scene
        self._clear_signals()

    def _clear_signals(self, mode=vrep.simx_opmode_oneshot):
        """Clears all signals this script can set in the simulation."""

        vrep.simxClearStringSignal(self.clientID, 'run_drop_object', mode)
        vrep.simxClearIntegerSignal(self.clientID, 'object_resting', mode)
        vrep.simxClearIntegerSignal(self.clientID, 'run_grasp_attempt', mode)
        vrep.simxClearStringSignal(self.clientID, 'header', mode)
        vrep.simxClearStringSignal(self.clientID, 'pregrasp', mode)
        vrep.simxClearStringSignal(self.clientID, 'postgrasp', mode)

    def _start_communication(self, ip, port,
                             wait_until_start_communicationed=True,
                             do_not_reconnect_once_disconnected=False,
                             time_out_in_ms=15000, comm_thread_cycle_in_ms=5):
        """Requests a communication pipe with the simulator."""

        clientID = vrep.simxStart(ip, port,
                                  wait_until_start_communicationed,
                                  do_not_reconnect_once_disconnected,
                                  time_out_in_ms,
                                  comm_thread_cycle_in_ms)
        return clientID if clientID != -1 else None

    def _start_simulation(self):
        """Tells a VREP scene to start execution."""

        if self.clientID is None:
            raise Exception('Client is not connected to V-REP server and '
                            'cannot start the simulation. Check the sim is '
                            'running and try connecting again.')

        r = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        if r != vrep.simx_return_ok:
            raise Exception('Unable to start simulation. Return code ', r)
        vrep.simxSynchronous(self.clientID, False)

    @staticmethod
    def _format_matrix(matrix_in):
        """Formats input matrices to V-REP as a list of 12 components."""

        matrix = matrix_in
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)

        size = matrix.size
        if size not in [12, 16]:
            raise Exception('Length of input matrix must be either [12, 16] '
                            'but provided length was <%d>' % size)
        matrix_ht = matrix.reshape(size // 4, 4)

        return matrix_ht[:3].flatten().tolist()

    def load_object(self, object_path, com, mass, inertia,
                    use_convex_as_respondable=False):
        """Loads an object into the simulator given it's full path.

        This function also sets the initial center of mass, mass, and
        inertia of the object.
        see: http://www.coppeliarobotics.com/helpFiles/en/regularApi/simImportMesh.htm

        In some cases, meshes that get loaded in to the simulator may be very
        complex, which is difficult for the dynamics engines to simulate
        properly. The flag use_convex_as_respondable indicates whether to
        perform all dynamics & collisions calculations relative to the convex
        hull over the object. This will improve stability, but depending on the
        shape of the object may make grasps look funny (e.g. not touching the
        visible mesh). Use with care.
        """

        if '.obj' in object_path:
            file_format = 0
        elif '.stl' in object_path:
            file_format = 4
        else:
            raise Exception('File format must be in {.obj, .stl}')

        in_ints = [file_format, use_convex_as_respondable]

        in_floats = []
        in_floats.extend(com)
        in_floats.extend([mass])
        in_floats.extend(inertia)

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'loadObject', in_ints, in_floats,
                                        [object_path], bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error loading object! Return code ', r)

    def get_object_pose(self):
        """Queries the simulator for current object pose WRT the workspace."""
        return self.get_pose_by_name('object')

    def set_object_pose(self, frame_work2obj):
        """Sets the pose for the mesh object to be WRT the workspace frame."""
        return self.set_pose_by_name('object', frame_work2obj)

    def set_gripper_pose(self, frame_work2palm, reset_config=True):
        """Sets the pose for the current object to be WRT the workspace frame.

        Setting gripper pose is a bit more intricate then the others, as since
        it's a dynamic object,
        """
        frame = self._format_matrix(frame_work2palm)

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'setGripperPose', [reset_config],
                                        frame, [], bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper pose! Return code ', r)

    def get_pose_by_name(self, name):
        """Queries the simulator for the pose of object corresponding to <name>.

        The retrieved pose is with respect to the workspace frame.
        """

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'getPoseByName', [], [], [name],
                                        bytearray(), vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error getting pose for <%s>!' % name,
                            'Return code ', r)
        return lib.utils.format_htmatrix(r[2])

    def set_pose_by_name(self, name, frame_work2pose):
        """Sets the pose of a scene object (by name) WRT workspace."""

        frame = self._format_matrix(frame_work2pose)

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'setPoseByName', [], frame, [name],
                                        bytearray(), vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting pose for name <%s>!' % name,
                            'Return code ', r)

    def get_joint_position_by_name(self, name):
        """Given a name of a joint, get the current position"""

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'getJointPositionByName', [], [],
                                        [name], bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting joint position for <%s>!' % name,
                            'Return code ', r)
        return r[2][0]

    def set_joint_position_by_name(self, name, position):
        """Given a name of a joint, get the current position"""

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'setJointPositionByName', [],
                                        [position], [name], bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting joint position for <%s>!' % name,
                            'Return code ', r)

    def set_gripper_kinematics_mode(self, mode='forward'):

        joint_modes = ['forward', 'inverse']
        if mode not in joint_modes:
            raise Exception('Joint mode must be in %s' % joint_modes)

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'setJointKinematicsMode', [], [],
                                        [mode], bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper kinematics mode.',
                            'Return code ', r)

    def set_gripper_properties(self, collidable=False, measureable=False,
                               renderable=False, detectable=False,
                               cuttable=False, dynamic=False,
                               respondable=False, visible=False):
        """Sets misc. parameters of the gripper model in the sim.

        This is used to accomplish things such as: moving the gripper without
        colliding with anything, setting it to be visible & being captured
        by the cameras, static so fingers don't move, etc ...
        """
        props = [collidable, measureable, renderable, detectable, cuttable,
                 dynamic, respondable, visible]

        # V-REP encodes these properties as 'not_xxxxx', so we'll just invert
        # them here to make calls in the simulator straightforward
        props = [not p for p in props]

        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'setGripperProperties', props, [], [],
                                        bytearray(), vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper properties. Return code ', r)

    def query(self, frame_work2cam, frame_world2work=None,
              resolution=128, rgb_near_clip=0.2, rgb_far_clip=10.0,
              depth_far_clip=1.25, depth_near_clip=0.2, p_light_off=0.25,
              p_light_mag=0.1, camera_fov=70 * np.pi / 180., reorient_up=True,
              randomize_texture=True, randomize_colour=True,
              randomize_lighting=True,
              texture_path=os.path.join(project_dir, 'texture.png')):
        """Queries the simulator for an image using a camera post WRT workspace.

        The parameters in the signature help define those needed for performing
        domain randomization. Given a camera pose, the simulator samples:
        1. Random light positions
        2. Random number of lights
        3. Object texture / colour
        4. Workspace texture

        Returns
        -------
        images: 4-d array of shape (1, channels, rows, cols), where channels
            [0, 1, 2] = RGB, [3] = Depth, [4] = Object mask
        frame_work2cam: 4x4 HT matrix of camera WRT workspace
        """

        if randomize_texture and not os.path.exists(texture_path):
            print('Cannot find <%s> in system, not randomizing textures.')
            randomize_texture = False

        # Force the camera to always be looking "upwards"
        if reorient_up and frame_world2work is not None:
            frame_work2cam = lib.utils.reorient_up_direction(
                frame_work2cam, frame_world2work, direction_up=[0, 0, 1])
        elif reorient_up:
            print('Must provide <frame_world2work> in order to reorient the '
                  'cameras y-direction to be upwards.')

        frame_work2cam = self._format_matrix(frame_work2cam)

        in_ints = [resolution]
        in_ints.extend([int(randomize_texture)])
        in_ints.extend([int(randomize_colour)])
        in_ints.extend([int(randomize_lighting)])

        # Prepare the inputs; we give it both a camera and object pose
        in_floats = frame_work2cam[:]
        in_floats.extend([p_light_off])
        in_floats.extend([p_light_mag])
        in_floats.extend([rgb_near_clip])
        in_floats.extend([rgb_far_clip])
        in_floats.extend([depth_near_clip])
        in_floats.extend([depth_far_clip])
        in_floats.extend([camera_fov])

        in_strings = [texture_path]

        # Make a call to the simulator
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                        vrep.sim_scripttype_childscript,
                                        'queryCamera', in_ints, in_floats,
                                        in_strings, bytearray(),
                                        vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            return None, None

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

        Note that we also set the gripper to be non-collidable / respondable,
        so it doesn't interfere with the drop process
        """

        self._clear_signals()

        self.set_gripper_properties(visible=True, dynamic=True)

        frame = self._format_matrix(frame_work2obj)

        # Launch the threaded script
        vrep.simxSetStringSignal(self.clientID, 'run_drop_object',
                                 vrep.simxPackFloats(frame),
                                 vrep.simx_opmode_oneshot)

        r = -1
        while r != vrep.simx_return_ok:
            r, success = vrep.simxGetIntegerSignal(
                self.clientID, 'object_resting', vrep.simx_opmode_oneshot_wait)

        if success == 0:
            raise Exception('Error dropping object! Return code: ', r)

    def run_threaded_candidate(self, finger_angle=0):
        """Launches a threaded scrip in simulator that tests a grasp candidate.

        If the initial grasp has all three fingers touching the objects, then
        values for the pre- and post-grasp will be returned. Otherwise, a {-1}
        will be returned (grasp was attempted, but initial fingers weren't in
        contact with the object).

        Note that we also set the gripper to be collidable & respondable so it
        is able to interact with the object & table.
        """

        # The simulator is going to send these signals back to us, so clear
        # them to make sure we're not accidentally reading old values
        self._clear_signals()

        self.set_gripper_properties(visible=True, dynamic=True,
                                    collidable=True, respondable=True)

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

        pregrasp = decode_grasp(header, vrep.simxUnpackFloats(pregrasp))
        postgrasp = decode_grasp(header, vrep.simxUnpackFloats(postgrasp))
        return pregrasp, postgrasp

    def stop(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def start(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

    def is_running(self):
        """Checks if simulator is connectedd by querying for connection ID."""
        return vrep.simxGetConnectionId(self.clientID) != -1
