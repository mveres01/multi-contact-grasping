import os
import csv
import h5py
import numpy as np
import trimesh
from trimesh import transformations as tf

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

from lib.python_config import config_mesh_dir


def calc_mesh_centroid(trimesh_mesh,  center_type='vrep'):
    """Calculates the center of a mesh according to three different metrics."""

    if center_type == 'centroid':
        return trimesh_mesh.centroid
    elif center_type == 'com':
        return trimesh_mesh.center_mass
    elif center_type == 'vrep': # How V-REP assigns object centroid
        maxv = np.max(trimesh_mesh.vertices, axis=0)
        minv = np.min(trimesh_mesh.vertices, axis=0)
        return 0.5*(minv+maxv)


def plot_equal_aspect(vertices, axis):
    """Forces the plot to maintain an equal aspect ratio

    # See:
    # http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """

    max_dim = np.max(np.array(np.max(vertices, axis=0) - np.min(vertices, axis=0)))

    mid = 0.5*np.max(vertices, axis=0) + np.min(vertices, axis=0)
    axis.set_xlim(mid[0] - max_dim, mid[0] + max_dim)
    axis.set_ylim(mid[1] - max_dim, mid[1] + max_dim)
    axis.set_zlim(mid[2] - max_dim, mid[2] + max_dim)

    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    return axis


def plot_mesh(mesh_path, workspace2obj, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_path : path to a given mesh
    workspace2obj : 4x4 transform matrix from the workspace to object
    axis : (optional) a matplotlib axis for plotting a figure
    """

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    # Load the object mesh
    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center

    # Rotate the vertices so they're in the frame of the workspace
    mesh.apply_transform(workspace2obj)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5,0.5,1])
    axis.add_collection3d(poly)

    return plot_equal_aspect(mesh.vertices, axis)


def normalize_vector(vector):
    """Normalizes a vector to have a magnitude of 1."""
    return vector / np.sqrt(np.sum(vector ** 2))


def format_htmatrix(matrix_in):
    """Formats a 3x3 rotation matrix into a 4x4 homogeneous matrix."""

    if isinstance(matrix_in, list):
        matrix_in = np.asarray(matrix_in).reshape(3, 4)
    elif matrix_in.ndim == 1:
        matrix_in = matrix_in.reshape(3, 4)

    ht_matrix = np.eye(4)
    ht_matrix[:3] = matrix_in
    return ht_matrix


def format_point(point):
    """Formats a 3-element [x,y,z] vector as a 4-element vector [x,y,z,1]."""

    return np.hstack((point, 1))


def invert_htmatrix(htmatrix):
    """Inverts a homogeneous transformation matrix."""

    shape_in = htmatrix.shape

    inv = np.eye(4)
    rot_T = htmatrix[:3,:3].T
    inv[:3,:3] = rot_T
    inv[:3, 3] = -np.dot(rot_T, htmatrix[:3,3])
    return inv


def change_frames_as_vec(vec1, vec2):

    mat1 = format_htmatrix(vec1)
    mat2 = format_htmatrix(vec2)
    return np.dot(mat1, mat2)[:3].flatten()


def rot_x(theta):
    """Builds a 3x3 rotation matrix around x.

    Parameters
    ----------
    theta : angle of rotation in rads.
    """

    mat = np.asarray(
            [[1, 0,                0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta),  np.cos(theta)]])
    return mat


def rot_y(theta):
    """Builds a 3x3 rotation matrix around y

    Parameters
    ----------
    theta : angle of rotation in rads
    """

    mat = np.asarray(
            [[np.cos(theta), 0, np.sin(theta)],
             [0,               1, 0],
             [-np.sin(theta),0, np.cos(theta)]])
    return mat


def rot_z(theta):
    """Builds a 3x3 rotation matrix around z.

    Parameters
    ----------
    theta : angle of rotation in rads.
    """

    mat = np.asarray(
             [[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,                0,               1]])
    return mat


def rxyz(thetax, thetay, thetaz, as_degrees=False):
    """Calculates rotation matrices by multiplying in the order x,y,z.

    Parameters
    ----------
    thetax : rotation around x in degrees.
    thetay : rotation around y in degrees.
    thetaz : rotation around z in degrees.
    """

    if as_degrees is True:
        thetax = thetax*math.pi/180.
        thetay = thetay*math.pi/180.
        thetaz = thetaz*math.pi/180.

    # Convert radians to degrees
    rx = tf.rotation_matrix(thetax, [1,0,0])
    ry = tf.rotation_matrix(thetay, [0,1,0])
    rz = tf.rotation_matrix(thetaz, [0,0,1])
    rxyz = tf.concatenate_matrices(rx,ry,rz)

    return rxyz


def get_unique_idx(data_in, n_nbrs=-1, thresh=1e-4, scale=False):
    """Finds the unique elements of a dataset using NearestNeighbors algorithm

    Parameters
    ----------
    data_in : array of datapoints
    n : number of nearest neighbours to find
    thresh : float specifying how close two items must be to be considered
        duplicates

    Notes
    -----
    The nearest neighbour algorithm will usually flag the query datapoint
    as being a neighbour. So we generally need n>1
    """

    from sklearn.neighbors import NearestNeighbors

    if n_nbrs == -1:
        n_nbrs = data_in.shape[0]

    # Scale the data so points are weighted equally/dont get misrepresented
    if scale is True:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(data_in)
        data_in = scaler.transform(data_in)

    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='brute').fit(data_in)

    # This vector will contain a list of all indices that may be duplicated.
    # We're going to use each datapoint as a query.
    exclude_vector = np.zeros((data_in.shape[0],), dtype=bool)
    for i in xrange(data_in.shape[0]):

        # If we've already classified the datapoint at this index as being a
        # duplicate, there's no reason to process it again
        if exclude_vector[i] == True:
            continue

        # Find how close each point is to the query. If we find a point that
        # is less then some threshold, we add it to our exlude list
        distances, indices = nbrs.kneighbors(data_in[i:i+1])
        distances = distances.reshape(-1)
        indices = indices.reshape(-1)

        where = np.bitwise_and(distances <= thresh, indices != i)

        exclude_vector[indices[where]] = True

    # Return a list of indices that represent unique elements of the dataset
    return np.bitwise_not(exclude_vector)


def convert_grasp_frame(frame2matrix, matrix2grasp):
    """Function for converting from one grasp frame to another.

    This is useful as transforming grasp positions requires multiplication of
    4x4 matrix, while contact normals (orientation) are multiplication of 3x3
    components (i.e. without positional components).
    """

    if matrix2grasp.ndim == 1:
        matrix2grasp = np.atleast_2d(matrix2grasp)

    # A grasp is contacts, normals, and forces (3), and has (x,y,z) components
    n_fingers = int(matrix2grasp.shape[1] / 6)
    contact_points = matrix2grasp[0, :n_fingers*3].reshape(3, 3)
    contact_normals = matrix2grasp[0, n_fingers*3:n_fingers*6].reshape(3, 3)

    # Append a 1 to end of contacts for easier multiplication
    contact_points = np.hstack([contact_points, np.ones((3, 1))])

    # Convert positions, normals, and forces to object reference frame
    points = np.zeros((n_fingers, 3))
    normals = np.zeros(points.shape)

    for i in xrange(n_fingers):

        points[i] = np.dot(frame2matrix, contact_points[i:i+1].T)[:3].T
        normals[i] = np.dot(frame2matrix[:3, :3], contact_normals[i:i+1].T).T

    return np.vstack([points, normals]).reshape(1, -1)


def skew_symmetric(x, y, z):
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def get_rot_mat(vector1, vector2):
    """Calculates the rotation needed to bring two axes coincident"""

    # Normalize the vectors to be unit vectors
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)

    # Take the vector crossproduct and normalize
    cross = np.squeeze(np.cross(vector1, vector2))

    frob = np.sqrt(np.sum(cross ** 2))

    # Normalize the cross product, then take the inner dot-product
    inner_prod = np.sum(vector1 * vector2)

    # Build the 3x3 identity matrix and skew symmetric matrices
    skew = skew_symmetric(cross[0], cross[1], cross[2])

    # Get the rotation matrix
    p1 = np.eye(3) + skew
    p2 = np.dot(skew, skew) * (1.0 - inner_prod) / (frob ** 2 + 1e-8)

    output = np.eye(4)
    output[:3, :3] =  p1 + p2

    return output


def reorient_up_direction(work2cam, world2work, direction_up, world2point=None):
    """Calculate an "Up" direction vector that points 'Y' towards origin.

    Notes
    -----
    world2point: (optional) 3d point in world frame we want to focus on
    See http://stackoverflow.com/questions/14250208/three-js-how-to-rotate-an-object-to-lookat-one-point-and-orient-towards-anothe
    """

    if isinstance(work2cam, list) or len(work2cam) == 12:
        work2cam = format_htmatrix(work2cam)
    if isinstance(world2work, list) or len(world2work) == 12:
        world2work = format_htmatrix(world2work)
    if not isinstance(direction_up, list):
        raise Exception('param :direction_up: must be of type <list>')

    world_up = np.asarray(direction_up)

    world2cam = np.dot(world2work, work2cam)
    cam2world = invert_htmatrix(world2cam)

    # Look at a point along the camera normal (z-direction)
    world2cam_z = world2cam[:3, 2]
    if world2point is None:
        point2lookat = format_point(world2cam[:3, 2])
    else:
        point2lookat = format_point(world2point)

    # Calculate Direction (D) and Up vector (U).
    D = np.dot(cam2world[:3, :3], world2cam_z)
    D = normalize_vector(D)

    U = np.dot(cam2world[:3, :3], world_up)
    U = normalize_vector(U)

    right = np.cross(U, D)
    right = normalize_vector(right)

    backwards = np.cross(right, U)
    backwards = normalize_vector(backwards)

    up = np.cross(backwards, right)
    up = normalize_vector(up)

    # This is the rotation matrix that aligns the up-vector
    rotMat = np.array([[right[0], up[0], backwards[0], 0],
                       [right[1], up[1], backwards[1], 0],
                       [right[2], up[2], backwards[2], 0],
                       [0,        0,     0,            1]])

    # Do a local rotation of work2cam matrix
    work2cam = np.dot(work2cam, rotMat)
    world2cam = np.dot(world2work, work2cam)
    cam2world = invert_htmatrix(world2cam)

    # Perform a rotation + translation of camera frame
    direction_vec1 = np.atleast_2d(world_up)
    direction_vec2 = np.dot(cam2world, point2lookat)[:3]
    rotMat = get_rot_mat(direction_vec1, direction_vec2)

    return np.dot(work2cam, rotMat)[:3].flatten()


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
                   local_rot=None, global_rot=None, min_dist=None):
    """Computes a random pose for any frame by varying position + orientation.

    Given an initial frame of WRT the workspace, we choose a random offset
    along the local z-direction according to offset_mag. The frame is then
    rotated according to random local and global rotations, by sampling
    (x, y, z) values specified in local/global_rot.
    """

    if frame_work2pose is None:
        frame = np.eye(4)
    elif isinstance(frame_work2pose, list):
        frame = format_htmatrix(frame_work2pose)
    elif frame_work2pose.ndim == 1:
        frame = format_htmatrix(frame_work2pose)
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
    if min_dist is not None:
        randomized_ht[2, 3] = np.maximum(randomized_ht[2, 3], min_dist)
    return randomized_ht[:3].flatten()


def load_subset(h5_file, object_keys, shuffle=True):
    """Loads a subset of an hdf5 file given top-level keys.

    The hdf5 file was created using object names as the top-level keys.
    Loading a subset of this file means loading data corresponding to
    specific objects.
    """

    if not isinstance(object_keys, list):
        object_keys = [object_keys]

    #  Load the grasps
    grasps = []
    for obj in object_keys:
        grasps.append(h5_file[obj]['grasps'][:, :18])
    grasps = np.vstack(grasps)

    # Load the grasp properties.
    props = {}
    for obj in object_keys:

        # Since there are multiple properties per grasp, we'll do something
        # ugly and store intermediate results into a list, and merge all the
        # properties after we've looped through all the objects.
        for prop in h5_file[obj].keys():

            if prop not in props:
                props[prop] = []
            data = np.atleast_2d(h5_file[obj][prop])

            if prop != 'object_name':
                props[prop].append(data.astype(np.float32))
            else:
                props[prop].append(data.astype(str))

    # Merge the list-of-lists for each property into single arrays
    idx = np.arange(grasps.shape[0])
    if shuffle is True:
        np.random.shuffle(idx)

    grasps = grasps[idx]
    for key in props.keys():
        if key == 'object_name':
            props[key] = (np.hstack(props[key]).T)[idx]
        else:
            props[key] = np.vstack(props[key])[idx]

    return grasps, props
