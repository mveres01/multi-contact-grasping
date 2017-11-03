import os
import sys
sys.path.append('..')

import csv
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

import lib.utils
from lib.utils import plot_mesh, format_htmatrix, calc_mesh_centroid, plot_equal_aspect
from lib.python_config import (config_mesh_dir, config_candidate_dir,
                               config_pose_path)

import trimesh
from trimesh import sample



def rand_step(max_angle):
    """Returns a random point between (0, max_angle."""
    return np.float32(np.random.randint(0, max_angle))*np.pi/180.


def to_rad(deg_x, deg_y, deg_z):
    """Converts a tuple of (x, y, z) in degrees to radians."""
    return (deg_x*np.pi/180., deg_y*np.pi/180., deg_z*np.pi/180.)


def cvt4d(point_3d):
    """Helper function to quickly format a 3d point into a 4d vector."""
    return np.hstack([point_3d, 1])


def get_mesh_properties(data_vector):
    """Parses information on objects pose collected via sim."""

    mesh_props = {}
    mesh_props['name'] = str(data_vector[0])
    mesh_props['com'] = np.float32(data_vector[1:4])
    mesh_props['inertia'] = np.float32(data_vector[4:13])
    mesh_props['bbox'] = np.float32(data_vector[13:19])

    work2obj = np.float32(data_vector[19:31]).reshape(3, 4)
    obj2grip = np.float32(data_vector[31:43]).reshape(3, 4)

    # Check that the position of the object isn't too far
    if any(work2obj[:, 3] > 1):
        raise Exception('%s out of bounds'%mesh_props['name'])

    # Reshape Homogeneous transform matrices from 3x4 into 4x4
    mesh_props['work2obj'] = format_htmatrix(work2obj)
    mesh_props['obj2grip'] = format_htmatrix(obj2grip)

    return mesh_props


def plot_bbox(work2obj, bbox, axis=None):
    """Plots the objects bounding box on an (optional) given matplotlib fig."""

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    corners, _ = get_corners_and_planes(bbox)
    for corner in corners:
        new_corner = np.dot(work2obj, cvt4d(corner))[:3]
        axis.scatter(*new_corner, color='r', marker='x', s=125)

    return axis


def plot_candidate(start, end=None, axis=None):
    """Plots grasp candidates the algorithm has identified."""

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    axis.scatter(*start, color='g', marker='o', s=5)

    if end is not None:
        axis.scatter(*end, color='r', marker='*', s=5)
        axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-')

    return axis


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

    for i in xrange(0, len(matrices), 2000):

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


def generate_candidates(mesh_properties, mesh_path, num_samples=10000,
                        augment=True, noise_level=0.05):
    """Generates grasp candidates via surface normals of the object."""

    # Initial offset of gripper from object
    GRIPPER_OFFSET = -0.1
    # Defines the up-vector for the workspace frame
    up_vector = np.asarray([0, 0, -1])
    # Rotate the gripper pose around the local z-direction
    angle_list = [j * np.pi / 4 for j in xrange(8)]

    bbox = mesh_properties['bbox']
    work2obj = mesh_properties['work2obj']

    # Load the object, calculate a new centroid, and transform to workspace frame
    mesh = load_mesh(mesh_path)
    mesh.apply_transform(work2obj)

    #points = sample.sample_surface(mesh, num_samples)
    points = sample.sample_surface_even(mesh, num_samples)
    normals, matrices = [], []

    # Find the normals corresponding to the sampled points
    triangles = mesh.triangles_center
    for p in points:

        face_idx = np.argmin(np.sum((p - triangles)**2, axis=1))
        normal = mesh.triangles_cross[face_idx]
        normal = lib.utils.normalize_vector(normal)

        # Add random noise to the surface normals, centered around 0
        if augment is True:
            normal += (np.random.random(len(normal)) - 0.5) * noise_level
            normal = lib.utils.normalize_vector(normal)

        # Check whether the surface normal would go below the table or not
        if normal[2] * 10 <= 0:
            continue

        # Since we need to set a pose for the gripper, we need to calculate the
        # rotation matrix from a given surface normal
        matrix = lib.utils.get_rot_mat(up_vector, normal)
        matrix[:3, 3] = p

        # Sanity check that our rotation matrix is correct
        mult = np.dot(matrix[:3, :3], np.atleast_2d(up_vector).T).flatten()
        assert np.allclose(mult, normal, 1e-1)

        # Calculate an offset for the gripper from the object.
        matrix[:3, 3] = np.dot(matrix, np.array([0, 0, GRIPPER_OFFSET, 1]).T)[:3]

        # Apply rotation around z-axis of palm
        for angle in angle_list:
            rotz = np.eye(4)
            rotz[:3, :3] = lib.utils.rot_z(angle)
            matrices.append(np.dot(matrix, rotz)[:3].flatten())
            normals.append(normal)

    matrices = np.vstack(matrices)
    normals = np.vstack(normals)
    angles = np.zeros((matrices.shape[0], 6)) # legacy needed for compatibility

    plot_mesh_with_normals(mesh, matrices, up_vector, normals)
    mesh_name = mesh_path.split(os.path.sep)[-1].split('.')[0]
    plt.title('Mesh: ' + mesh_name)
    plt.show()

    return angles, matrices


def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    print 'mesh_path: ', mesh_path
    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh


def main(data_vector, mesh_input_dir, candidate_output_dir, to_keep=-1):

    if not os.path.exists(candidate_output_dir):
        os.makedirs(candidate_output_dir)

    mesh_properties = get_mesh_properties(data_vector)
    mesh_path = os.path.join(mesh_input_dir, mesh_properties['name'] + '.stl')

    angles, matrices = generate_candidates(mesh_properties, mesh_path)

    success = angles.shape[0]
    if success == 0:
        print 'No candidates generated. Try reducing step sizes?'
        return
    print '%s \n# of successful TF: %d'%(mesh_properties['name'], success)

    # Choose how many candidates we want to save (-1 will save all)
    if to_keep == -1 or to_keep > success:
        to_keep = success

    random_idx = np.arange(success)
    np.random.shuffle(random_idx)

    # Save the data
    savefile = os.path.join(candidate_output_dir, mesh_properties['name'] + '.txt')
    csvfile = open(savefile, 'wb')
    writer = csv.writer(csvfile, delimiter=',')

    obj_mtx = mesh_properties['work2obj'][:3].flatten()
    for idx in random_idx[:to_keep]:
        writer.writerow(np.hstack(\
            [mesh_properties['name'], idx, obj_mtx, angles[idx], matrices[idx],
             mesh_properties['com'], mesh_properties['inertia']]))
    csvfile.close()


if __name__ == '__main__':

    np.random.seed(np.random.randint(1, 1234567890))

    # We usually run this in parallel (using gnu parallel), so we pass in
    # a row of information at a time (i.e. from collecting initial poses)
    # This is for test purposes only

    data_vector='remote_poisson_013,4.0678394725546e-05,-0.0016528239939362,0.65648239850998,0.43098190426826,-2.4128098630172e-08,-2.6717591026681e-05,-2.4128127051881e-08,0.43097913265228,0.0010859938338399,-2.6717591026681e-05,0.0010859938338399,1.2742271792376e-05,-0.023338001221418,0.023338001221418,-0.10426650196314,0.10426650196314,-0.0067520001903176,0.0067520001903176,0.99972122907639,-0.02355139143765,-0.0016869015526026,0,0.023550977930427,0.99972259998322,-0.00026543068815954,0,0.0016926848329604,0.00022562852245755,0.99999856948853,0.0065092444419861,-0.99972152709961,0.023552084341645,-0.0016915028681979,0,0.023552497848868,0.99972259998322,-0.0002260458713863,3.929017111659e-10,0.0016857098089531,-0.0002658219600562,-0.99999892711639,0.10705983638763,'
    data_vector = data_vector.split(',')[:-1]
    main(data_vector, config_mesh_dir, config_candidate_dir, to_keep=10000)


    if len(sys.argv) == 1:
        data_vector = pd.read_csv(config_pose_path, delimiter=',', header=None)

        for i in xrange(data_vector.shape[0]):
            dv = (data_vector.values)[i]
            main(dv, config_mesh_dir, config_candidate_dir, to_keep=10000)
    else:
        data_vector = sys.argv[1]
        data_vector = data_vector.split(',')[:-1]
        main(dv, config_mesh_dir, config_candidate_dir, to_keep=10000)
