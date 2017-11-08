import os
from sys import platform


# Current file
cur_file_path = os.path.abspath(__file__)

# Directory of current file
cur_dir = os.path.abspath(os.path.join(cur_file_path, os.pardir))

# Root project directory
project_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))


scene_name = 'grasp_scene.ttt'

config_object_dir = os.path.join(project_dir, 'data', 'meshes')
config_mesh_dir = os.path.join(project_dir, 'data', 'processed_meshes')
config_candidate_dir = os.path.join(project_dir, 'data', 'candidates')
config_simulation_path = os.path.join(project_dir, 'scenes', scene_name)
config_collected_data_dir = os.path.join(project_dir, 'output', 'collected')
config_processed_data_dir = os.path.join(project_dir, 'output', 'processed')
config_dataset_path = os.path.join(project_dir, 'output', 'grasping.hdf5')
