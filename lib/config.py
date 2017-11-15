import os

# Current file
cur_file_path = os.path.abspath(__file__)

# Directory of current file
cur_dir = os.path.abspath(os.path.join(cur_file_path, os.pardir))

# Root project directory
project_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))


scene_name = 'grasp_scene.ttt'
config_mesh_dir = os.path.join(project_dir, 'data', 'meshes')
config_candidate_dir = os.path.join(project_dir, 'data', 'candidates')
config_simulation_path = os.path.join(project_dir, 'scenes', scene_name)
config_output_dir = os.path.join(project_dir, 'output')
config_output_collected_dir = os.path.join(config_output_dir, 'collected')
config_output_dataset_path = os.path.join(project_dir, 'output', 'grasping.hdf5')
