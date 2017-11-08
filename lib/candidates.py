import os
import sys
sys.path.append('..')
import h5py
import numpy as np
import trimesh
import trimesh.transformations as tf
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

import lib
from lib.python_config import (config_simulation_path,
                               config_mesh_dir, config_processed_data_dir)

import queryclass



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

    axis = lib.utils.plot_equal_aspect(mesh.vertices, axis)

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

    '''
    matrices = np.vstack(matrices)
    plot_mesh_with_normals(mesh, matrices, up_vector)
    mesh_name = mesh_path.split(os.path.sep)[-1].split('.')[0]
    plt.title('Mesh: ' + mesh_name)
    plt.show()
    '''
    return np.vstack(matrices)



def collect_grasps(mesh_path, port=19999, mass=1, initial_height=0.5):

    sim = queryclass.SimulatorInterface(port=port)

    # Where we'll save all our data to
    if not os.path.exists(config_processed_data_dir):
        os.makedirs(config_processed_data_dir)

    mesh_name = mesh_path.split(os.path.sep)[-1]

    output_file = mesh_name.split('.')[0] + '.hdf5'
    save_path = os.path.join(config_processed_data_dir, output_file)

    datafile = h5py.File(save_path, 'w')
    pregrasp_group = datafile.create_group('pregrasp')
    postgrasp_group = datafile.create_group('postgrasp')


    # Load the mesh from file, so we can compute grasp candidates, and access
    # information such as the center of mass & inertia
    mesh = load_mesh(mesh_path)
    com = mesh.mass_properties['center_mass']
    inertia = mesh.mass_properties['inertia'].flatten()

    candidates = generate_candidates(mesh, num_samples=1000,
                                     noise_level=0.05,
                                     gripper_offset=-0.07)

    # Load the object into the sim & specify initial properties.
    # With every new mesh, we give it a starting pose at some height above the
    # table, and allow it to fall into a natural resting pose.
    sim.load_object(mesh_path, com, mass, inertia*5)

    pose = sim.get_object_pose()

    pose[:3, 3] = [0, 0, initial_height]

    sim.run_threaded_drop(pose[:3].flatten())

    # Reset the object on each grasp attempt to its resting pose. Note this
    # doesn't have to be done, but it avoids instances where the object may
    # subsequently have fallen off the table
    pose = sim.get_object_pose()

    num_successful_grasps = 0
    for row in candidates:

        mult = np.dot(pose, lib.utils.format_htmatrix(row))

        direction = np.dot(mult[:3, :3], np.atleast_2d([0, 0, 1]).T)
        if direction[2] * 10 > 0.:
            continue

        #print 'Setting object pose!'
        sim.set_object_pose(pose[:3].flatten())

        #print 'Setting gripper pose!'
        sim.set_gripper_pose(mult[:3].flatten())

        #print 'Attempting grasp'
        pregrasp, postgrasp = sim.run_threaded_candidate()

        if pregrasp is None or postgrasp is None:
            continue

        if not postgrasp['all_in_contact']:
            continue

        # Pregrasp & postgrasp are filled the same way
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

if __name__ == '__main__':

    if len(sys.argv) == 1:
        meshes = os.listdir(config_mesh_dir)
        collect_grasps(os.path.join(config_mesh_dir, 'watering_can_poisson_003.stl'))
    else:
        port = sys.argv[1]
        mesh = sys.argv[2]
        collect_grasps(os.path.join(config_mesh_dir, mesh), port)
