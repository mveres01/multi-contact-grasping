import os
import sys
sys.path.append('..')
import glob
import h5py
import numpy as np
import trimesh
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import lib.utils
from lib.python_config import (config_mesh_dir, config_collected_data_dir)

import siminterface as SI


def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
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
    elif isinstance(direction_vec, np.ndarray) and direction_vec.ndim == 1:
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

    axis = lib.utils.plot_equal_aspect(mesh.vertices, axis)

    for i in xrange(0, len(matrices), 10):

        transform = lib.utils.format_htmatrix(matrices[i])

        # We'll find the direction by finding the vector between two points
        gripper_point = np.hstack([matrices[i, 3], matrices[i, 7], matrices[i, 11]])
        gripper_point = np.atleast_2d(gripper_point)

        direction = np.dot(transform[:3, :3], dvec)
        direction = np.atleast_2d(direction).T

        # The plotted normals and grasp approach should be overlapping
        if normals is not None:
            to_plot = np.hstack([gripper_point, -np.atleast_2d(normals[i])])
            axis.quiver(*to_plot, color='r', length=0.1)

        a = np.hstack([gripper_point, -direction]).flatten()
        axis.quiver(*a, color='k', length=0.1)

        axis.scatter(*gripper_point.flatten(), c='b', marker='o', s=10)

    return axis


def generate_candidates(mesh, num_samples=1000, noise_level=0.05,
                        gripper_offset=-0.1, augment=True):
    """Generates grasp candidates via surface normals of the object."""

    # Defines the up-vector for the workspace frame
    up_vector = np.asarray([0, 0, -1])

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

    # Uncomment to view the generated grasp candidates
    matrices = np.vstack(matrices)
    plot_mesh_with_normals(mesh, matrices, up_vector)
    plt.show()

    return matrices


def collect_grasps(mesh_path, port, mass=1, initial_height=0.5, 
                   num_candidates=1000, candidate_noise_level=0.05, 
                   num_random_per_candidate=5,
                   candidate_offset=-0.07):

    # Get the paths & structures set up for saving results
    if not os.path.exists(config_collected_data_dir):
        os.makedirs(config_collected_data_dir)

    mesh_name = mesh_path.split(os.path.sep)[-1]

    output_file = mesh_name.split('.')[0] + '.hdf5'
    save_path = os.path.join(config_collected_data_dir, output_file)

    datafile = h5py.File(save_path, 'w')
    pregrasp_group = datafile.create_group('pregrasp')
    postgrasp_group = datafile.create_group('postgrasp')



    sim = SI.SimulatorInterface(port=port)


    # Load the mesh from file here, so we can generate grasp candidates 
    # and access object-specific properties like inertia.
    mesh = load_mesh(mesh_path)

    com = mesh.mass_properties['center_mass']
    inertia = mesh.mass_properties['inertia'].flatten()

    candidates = generate_candidates(mesh, num_samples=num_candidates,
                                     noise_level=candidate_noise_level,
                                     gripper_offset=candidate_offset)

    # Compute an initial object resting pose by dropping the object from a 
    # given position / height above the workspace table
    sim.load_object(mesh_path, com, mass, inertia*5)

    initial_pose = sim.get_object_pose()
    initial_pose[:3, 3] = [0, 0, initial_height]

    sim.run_threaded_drop(initial_pose)


    # Reset the object on each grasp attempt to its resting pose. Note this
    # doesn't have to be done, but it avoids instances where the object may
    # subsequently have fallen off the table
    object_pose = sim.get_object_pose()
 

    num_successful_grasps = 0
    for count, row in enumerate(candidates):

        work2candidate = np.dot(object_pose, lib.utils.format_htmatrix(row))

        # Don't want to try grasps where the gripper would be below the table
        direction = np.dot(work2candidate[:3, :3], np.atleast_2d([0, 0, 1]).T)
        if direction[2] * 10 > 0.:
            continue

        for _ in xrange(num_random_per_candidate):

            sim.set_object_pose(object_pose[:3].flatten())

            # We can randomize the gripper candidate by rotation or translation.
            # Here we let the pose vary +- 3cm along local z, and a random
            # rotation between [0, 360) degress around local z
            random_pose = lib.utils.randomize_pose(work2candidate,
                                                   offset_mag=0.03,
                                                   local_rot=(0, 0, 359))
            sim.set_gripper_pose(random_pose)

            # Try to grasp and lift the object. If the gripper during pre-grasp
            # didn't have all fingers in contact with the object, the returned
            # pregrasp & postgrasp structures will be None
            pregrasp, postgrasp = sim.run_threaded_candidate()

            if pregrasp is None or postgrasp is None:
                continue

            success = bool(int(postgrasp['all_in_contact']))
            print('Grasp %d/%d for object: %s \tSuccess? %s'%\
                  (count, len(candidates), mesh_name, success))

            if success is False: # Only save successful grasps
                continue


            # Create initial structures if dataset is currently empty
            if len(pregrasp_group) == 0:
                for key, val in pregrasp.iteritems():
                    initial = (1, val.shape[-1])
                    maxshape = (None, val.shape[-1])
                    pregrasp_group.create_dataset(key, initial, maxshape=maxshape)
                    postgrasp_group.create_dataset(key, initial, maxshape=maxshape)

            for key, val in pregrasp.iteritems():
                pregrasp_group[key].resize((num_successful_grasps+1, val.shape[-1]))
                pregrasp_group[key][num_successful_grasps] = val
            for key, val in postgrasp.iteritems():
                postgrasp_group[key].resize((num_successful_grasps+1, val.shape[-1]))
                postgrasp_group[key][num_successful_grasps] = val

            num_successful_grasps += 1
    datafile.close()

    sim.stop()


if __name__ == '__main__':

    meshes = glob.glob(os.path.join(config_mesh_dir, '*.stl'))

    '''
    #collect_grasps(os.path.join(config_mesh_dir, meshes[0]), interface)
    collect_grasps(os.path.join(config_mesh_dir, 'tetra_pak_poisson_019.stl'), 20010)
    '''

    import multiprocessing
    from multiprocessing import Process, Queue

    num_cores = 6
    port_list = np.arange(num_cores) + 20010

    queue = Queue(maxsize=len(meshes))
    for mesh in meshes:
        queue.put(os.path.join(config_mesh_dir, mesh))

    def consumer(port):
        while not queue.empty():
            mesh_name = queue.get()
            collect_grasps(mesh_name, port)

    processes = []
    for i, port in enumerate(port_list):
        p = Process(target=consumer, args=(port,))
        p.start()
        processes.append(p)

    #for p in processes:
    #    p.join()

    '''
    import glob

    port = 19997
    if len(sys.argv) == 1:
        meshes = glob.glob(os.path.join(config_mesh_dir, '*.stl'))
        #meshes = glob.glob(os.path.join(config_mesh_dir, 'watering*.stl'))
        for m in meshes:
            collect_grasps(os.path.join(config_mesh_dir, m), port)
    else:
        port = sys.argv[1]
        mesh_name = sys.argv[2]
        collect_grasps(os.path.join(config_mesh_dir, mesh_name), port)
    '''
