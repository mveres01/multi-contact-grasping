import os
import sys
sys.path.append('..')
from sys import platform
import subprocess
import time
import numpy as np
import trimesh.transformations as tf

import lib
import lib.utils
from lib.python_config import project_dir, config_simulation_path
import vrep
vrep.simxFinish(-1)


def is_listening(ip='127.0.0.1', port=19997):
    """Checks whether a program is listening on a port already or not"""

    if platform not in ['linux', 'linux2']:
        raise Exception('You must be running Linux to use this function.')

    # Get a list of all current open / connected ports
    try:
        netstat = subprocess.Popen(['netstat', '-nao'], stdout=subprocess.PIPE)
    except Exception as e:
        raise e
    ports = netstat.communicate()[0]

    return ip + ':' + str(port) in ports


def wait_for_signal(clientID, signal, mode=vrep.simx_opmode_oneshot_wait):
    r = -1
    while r != vrep.simx_return_ok:
        r, data = vrep.simxGetStringSignal(clientID, signal, mode)
    return data


def decode_images(float_string, near_clip, far_clip, res_x=128, res_y=128):
    """Decodes a float string containing Depth, RGB, and binary mask info.

    Images are encoded as (rows, cols, channels), and when retrieved from the
    simulator have their rows inverted. We undo these transformations and
    return a tensor of shape (1, channels, rows, cols).
    """

    assert len(float_string) == res_x * res_y * 7, \
        'Image data has length <%d> but expected length <%d>'%\
        (len(float_string, res_x, res_y * 7))

    images = np.asarray(float_string)

    depth = images[:res_x*res_y].reshape(res_y, res_x, 1)
    depth = near_clip + depth*(far_clip - near_clip)

    rgb = images[res_x*res_y:4*res_x*res_y].reshape(res_y, res_x, 3)
    mask = images[4*res_x*res_y:].reshape(res_y, res_x, 3)[:, :, 0:1]

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
        n_items = int(split[i+1])
        subset = line[:, current_pos:current_pos + n_items]

        try:
            subset = subset.astype(np.float32)
        except Exception:
            subset = subset.astype(str)

        grasp[name] = subset.ravel()
        current_pos += n_items
    return grasp


def spawn_simulation(port, vrep_path, scene_path):
    """Spawns a child process using screen and starts a remote VREP server."""

    if platform not in ['linux', 'linux2']:
        raise Exception('Must be running on Linux to spawn a simulation.')

    vrep_path = 'vrep.sh' if vrep_path is None else vrep_path

    # Command to launch VREP
    vrep_cmd = '%s -h -q -s -gREMOTEAPISERVERSERVICE_%d_FALSE_TRUE %s'% \
               (vrep_path, port, scene_path)

    # Command to launch VREP + detach from screen
    bash_cmd = 'screen -dmS port%d bash -c "export DISPLAY=:1 ;'\
               'ulimit -n 4096; %s " '%(port, vrep_cmd)

    process = subprocess.Popen(bash_cmd, shell=True)
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

    When running with linux, it's easier to start a continuous service by
    launcing V-REP from the command line and attaching it to a screen session.
    """

    def __init__(self, port, ip='127.0.0.1', vrep_path=None, scene_path=None):


        if not isinstance(port, int):
            raise Exception('Port <%s> must be of type <int>'%port)
        elif not isinstance(ip, str):
            raise Exception('IP address <%s> must be of type <str>'%ip)

        self.port = port
        self.ip = ip
        self.clientID = None

        # See if there's a simulation already listening on this port
        clientID = self._start_communication()

        # If there's nothing listening, we can try spawning a sim on linux
        if clientID == -1:
            if platform in ['linux', 'linux2'] and not is_listening(ip, port):

                if scene_path is None:
                    scene_path = config_simulation_path
                if not os.path.exists(scene_path):
                    raise Exception('Scene <%s> not found'%scene_path)

                print('Spawning a Continuous Server on port <%d>'%port)
                spawn_simulation(port, vrep_path, scene_path)

                # Try starting communication
                clientID = self._start_communication()

        if clientID == -1:
            raise Exception('Unable to connect to address <%s> on port '\
                            '<%d>. Check that the simulator is currently '\
                            'running, or the continuous service was started'\
                            'successfully.'%(ip, port))

        # Have communication
        self.clientID = clientID

        # Tell the scene to start running
        self._start_simulation()

        # Remove All previous signals in the scene
        self._clear_signals()

    def _clear_signals(self, mode=vrep.simx_opmode_oneshot):
        """Clears all signals this script can set in the simulation."""

        vrep.simxClearStringSignal(self.clientID, 'object_resting', mode)
        vrep.simxClearStringSignal(self.clientID, 'run_drop_object', mode)
        vrep.simxClearIntegerSignal(self.clientID, 'run_grasp_attempt', mode)
        vrep.simxClearStringSignal(self.clientID, 'header', mode)
        vrep.simxClearStringSignal(self.clientID, 'pregrasp', mode)
        vrep.simxClearStringSignal(self.clientID, 'postgrasp', mode)

    def _start_communication(self, wait_until_start_communicationed=True,
                             do_not_reconnect_once_disconnected=False,
                             time_out_in_ms=15000, comm_thread_cycle_in_ms=5):
        """Requests a communication pipe with the simulator."""

        return vrep.simxStart(self.ip, self.port, wait_until_start_communicationed,
                              do_not_reconnect_once_disconnected, time_out_in_ms,
                              comm_thread_cycle_in_ms)

    def _start_simulation(self):
        """Tells a VREP scene to start execution."""

        if self.clientID is None or self.clientID == -1:
            raise Exception('Unable to start the simulation scene. This is '\
                            'likely a result of not being connected to the '\
                            'simulator. Try reconnecting and start again.')

        r = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        if r != vrep.simx_return_ok:
            raise Exception('Unable to start simulation.')
        vrep.simxSynchronous(self.clientID, False)

    @staticmethod
    def _format_matrix(matrix_in):
        """Formats input matrices to V-REP as a lists of 12 components."""

        matrix = matrix_in
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)

        size = matrix.size
        if size not in [12, 16]:
            raise Exception('Length of input matrix must be either [12, 16] ' \
                            'but provided length was <%d>'%size)
        matrix_ht = matrix.reshape(size // 4, 4)

        return matrix_ht[:3].flatten().tolist()

    def load_object(self, object_path, com, mass, inertia):
        """Loads an object into the simulator given it's full path.

        This function also sets the initial center of mass, mass, and
        inertia of the object.
        see: http://www.coppeliarobotics.com/helpFiles/en/regularApi/simImportMesh.htm
        """

        if '.obj' in object_path:
            file_format = 0
        elif '.stl' in object_path: # 3 is regular stl, 4 is binary stl & default
            file_format = 4
        else:
            raise Exception('File format must be in [.obj, .stl]')

        in_floats = []
        in_floats.extend(com)
        in_floats.extend([mass])
        in_floats.extend(inertia)

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'loadObject', [file_format],
             in_floats, [object_path], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error loading object!')

    def set_object_pose(self, frame_work2obj):
        """Sets the pose for the current object to be WRT the workspace frame."""
        return self.set_pose_by_name('object', frame_work2obj)

    def get_object_pose(self):
        """Queries the simulator for current object pose WRT the workspace."""
        return self.get_pose_by_name('object')

    def set_gripper_pose(self, frame_work2palm):
        """Sets the pose for the current object to be WRT the workspace frame.

        Setting gripper pose is a bit more intricate then the others, as since
        it's a dynamic object,
        """

        frame = self._format_matrix(frame_work2palm)

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setGripperPose', [],
             frame, [], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper pose!')

    def set_pose_by_name(self, name, frame_work2pose):
        """Given a name of an object in the scene, set pose WRT to workspace."""

        frame = self._format_matrix(frame_work2pose)

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setPoseByName', [],
             frame, [name], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting pose for name <%s>!'%name)

    def get_pose_by_name(self, name):
        """Queries the simulator for the pose of object corresponding to <name>.

        The retrieved pose is with respect to the workspace frame.
        """

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'getPoseByName', [],
             [], [name], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error getting pose for <%s>!'%name)
        return lib.utils.format_htmatrix(r[2])

    def set_joint_position_by_name(self, name, position):
        """Given a name of an object in the scene, set pose WRT to workspace."""

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setJointPositionByName', [],
             [position], [name], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting joint position for <%s>!'%name)

    def set_gripper_properties(self, collidable=False, measureable=False,
                               renderable=False, detectable=False,
                               cuttable=False, dynamic=False,
                               respondable=False, visible=False):
        """Sets misc. parameters of the gripper model in the sim."""
        props = [collidable, measureable, renderable, detectable, cuttable,
                 dynamic, respondable, visible]

        # V-REP encodes these properties as 'not_xxxxx', so we'll just invert
        # them here to make calls in the simulator straightforward
        props = [not p for p in props]

        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'setGripperProperties', props,
             [], [], empty_buff, vrep.simx_opmode_blocking)

        if r[0] != vrep.simx_return_ok:
            raise Exception('Error setting gripper properties.')

    def query(self, frame_work2cam, frame_world2work,
              resolution=128, rgb_near_clip=0.2, rgb_far_clip=10.0,
              depth_far_clip=1.25, depth_near_clip=0.2, p_light_off=0.25,
              p_light_mag=0.1, camera_fov=70*np.pi/180., reorient_up=True,
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
        frame_work2cam: 4x4 homogeneous transformat matrix from workspace to camera
        """

        # Force the camera to always be looking "upwards"
        if reorient_up:
            frame_work2cam = lib.utils.reorient_up_direction( \
                frame_work2cam, frame_world2work, direction_up=[0, 0, 1])
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
        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'queryCamera', in_ints,
             in_floats, in_strings, empty_buff, vrep.simx_opmode_blocking)

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
        """
        self._clear_signals()

        frame = self._format_matrix(frame_work2obj)

        # Launch the threaded script
        vrep.simxSetStringSignal(self.clientID, 'run_drop_object',
                                 vrep.simxPackFloats(frame),
                                 vrep.simx_opmode_oneshot)

        r = -1
        while r != vrep.simx_return_ok:
            r, success = vrep.simxGetIntegerSignal(
                self.clientID, 'object_resting', vrep.simx_opmode_oneshot_wait)
        self._clear_signals()

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

        self._clear_signals()

        # Decode the results into a dictionary
        header = header.lstrip('{').rstrip('}')
        if header == '-1':
            return None, None

        pregrasp = decode_grasp(header, vrep.simxUnpackFloats(pregrasp))
        postgrasp = decode_grasp(header, vrep.simxUnpackFloats(postgrasp))
        return pregrasp, postgrasp

    def view_grasp(self, frame_world2work, frame_work2cam, grasp_wrt_cam,
                   reset_container=0):
        """Plots the contact positions and normals of a grasp WRT camera frame.

        This function first converts the grasp from the camera frame to workspace
        frame, then plots the contact positions and normals. This function
        signature is like this to quickly accomodate grasp predictions made
        by machine learning / neural nets predicting WRT a visual image.
        """

        frame_world2work = self._format_matrix(frame_world2work)
        frame_work2cam = self._format_matrix(frame_work2cam)

        # Prepare the inputs; we give it both a camera and object pose
        in_floats = frame_world2work[:]
        in_floats.extend(frame_work2cam)
        in_floats.extend(grasp_wrt_cam)

        in_ints = [reset_container]

        # Make a call to the simulator
        empty_buff = bytearray()
        r = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
             vrep.sim_scripttype_childscript, 'displayGrasp', in_ints,
             in_floats, [], empty_buff, vrep.simx_opmode_blocking)
        if r[0] != vrep.simx_return_ok:
            raise Exception('Error when trying to display grasps in simulator.')

    def stop(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def start(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

    def is_running(self):
        """Checks if simulator is connectedd by querying for connection ID."""
        return vrep.simxGetConnectionId(self.clientID) != -1



if __name__ == '__main__':

    def find_name(to_find):
        from lib.python_config import config_mesh_dir

        meshes = os.listdir(config_mesh_dir)
        for mesh_name in meshes:
            if str(to_find) in mesh_name:
                return mesh_name.split('.')[0]
        return None

    import h5py
    from lib.python_config import config_mesh_dir

    GLOBAL_DATAFILE = '/scratch/mveres/grasping-cvae-multi/valid256.hdf5'


    sim = SimulatorInterface(port=19000)


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


        # Randomize the camera pose by sampling an offset, local, and global rot
        base_offset = -0.4
        offset_mag = 0.15
        local_rot = (10, 10, 10)
        global_rot = (30, 30, 30)

        frame_work2cam = lib.utils.format_htmatrix(props['frame_work2palm'][i])

        frame_work2cam = lib.utils.randomize_pose(frame_work2cam, base_offset,
                                                  offset_mag, None, None)

        images, frame_work2cam_ht = sim.query(frame_work2cam,
                                              props['frame_world2work'][i],
                                              camera_fov=60*np.pi/180.)

        if frame_work2cam_ht is not None:
            frame_cam2work_ht = lib.utils.invert_htmatrix(frame_work2cam_ht)
            grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasps[i])

        print 'Running?: ', sim.is_running()


        '''
        sim.view_grasp(props['frame_world2work'][i],
                       props['frame_work2cam'][i],
                       grasps[i], reset_container=1)
        '''

        sim.set_gripper_pose(props['frame_work2palm'][i])

        pregrasp, postgrasp = sim.run_threaded_candidate()
        if pregrasp is None or postgrasp is None:
            continue

        grasp = np.hstack([pregrasp['work2contact0'],
                           pregrasp['work2contact1'],
                           pregrasp['work2contact2'],
                           pregrasp['work2normal0'],
                           pregrasp['work2normal1'],
                           pregrasp['work2normal2']])
        grasp = lib.utils.convert_grasp_frame(frame_cam2work_ht, grasp)


        sim.view_grasp(props['frame_world2work'][i],
                       props['frame_work2cam'][i],
                       grasps[i], reset_container=1)
        sim.set_object_pose(props['frame_work2obj'][i])
