import os
import sys
sys.path.append('..')
import csv
import numpy as np
import trimesh
from trimesh.io.export import export_mesh

from lib.python_config import (config_object_dir, config_mesh_dir,
                               config_prop_path, project_dir,
                               config_object_mass, config_object_density)


def process_mesh(mesh_input_path, mesh_output_dir):
    """Given the path to a mesh, make sure its watertight & estimate params."""

    # Holds the processed name (1), mass  (1), center of mass (3) & inertia (9)
    processed = np.zeros((1, 14), dtype=object)

    try:
        mesh = trimesh.load_mesh(mesh_input_path)
    except Exception as e:
        print 'Exception: Unable to load mesh %s (%s): '%(mesh_input_path, e)
        return None

    full_mesh_name = mesh_input_path.split(os.path.sep)[-1]
    mesh_name = full_mesh_name.split('.')[0]

    # Can visualize the mesh by uncommenting the line below
    #mesh.show()

    # Fix any issues with the mesh
    if not mesh.is_watertight:
        mesh.process()
        if not mesh.is_watertight:
            print 'Mesh (%s) cannot be made watertight'%mesh_name
            return None

    # Then export as an STL file
    mesh_output_name = os.path.join(mesh_output_dir, mesh_name + '.stl')
    export_mesh(mesh, mesh_output_name, 'stl')

    # Calculate mesh properties using build-in functions
    mesh_properties = mesh.mass_properties()
    com = np.array(mesh_properties['center_mass'])
    inertia = np.array(mesh_properties['inertia'])

    # Need to format the inertia based on object density.
    # We un-do the built-in calculation (density usually considered
    # as '1'), and multiply by our defined density
    inertia /= mesh_properties['density']
    inertia *= config_object_density

    # We don't want an unreasonable inertia
    inertia = np.clip(inertia, -1e-5, 1e-5)
    processed[0, 0] = mesh_name
    processed[0, 1] = config_object_mass
    processed[0, 2:5] = com
    processed[0, 5:14] = inertia.flatten()

    return processed


def main(mesh_input_dir, mesh_output_dir, prop_output_path):
    """Saves a copy of each of the meshes, fixing any issues along the way."""

    valid_xtns = ['.ply', '.stl', '.obj']

    if not os.path.exists(mesh_output_dir):
        os.makedirs(mesh_output_dir)

    # Get a list of the meshes in a directory
    mesh_list = os.listdir(mesh_input_dir)
    mesh_list = [m for m in mesh_list if any(v in m for v in valid_xtns) ]
    if len(mesh_list) == 0:
        raise Exception('No meshes found in dir %s'%mesh_input_dir)

    processed_mesh_list = []
    for name in mesh_list:
        mesh_path = os.path.join(mesh_input_dir, name)
        processed = process_mesh(mesh_path, mesh_output_dir)
        if processed is not None:
            processed_mesh_list.append(processed)
    processed_mesh_list = np.vstack(processed_mesh_list)

    print '%d/%d meshes successfully processed.'%\
            (len(processed_mesh_list), len(mesh_list))

    # Write each row to file
    with open(prop_output_path, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for to_write in processed_mesh_list:
            writer.writerow(to_write)


if __name__ == '__main__':
    main(config_object_dir, config_mesh_dir, config_prop_path)
