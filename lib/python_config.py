import os
from sys import platform

if platform == 'linux' or platform == 'linux2':
    project_dir = '/scratch/mveres/grasping-random-pose'
else:
    project_dir = 'C:\\users\\matt\\documents\\grasping-random-pose'

# Set a constant object mass and density
scene_name = 'collect_multiview_grasps.ttt'


# Used for prepare_mesh.py
config_object_mass = 1.0
config_object_density = 1000.
config_object_dir = os.path.join(project_dir, 'data', 'meshes')
config_mesh_dir = os.path.join(project_dir, 'data', 'processed_meshes')

# Used for prepare_candidates.py
config_candidate_dir = os.path.join(project_dir, 'data', 'candidates')
config_prop_path = os.path.join(project_dir, 'data', 'mesh_object_properties.txt')
config_pose_path = os.path.join(project_dir, 'data', 'initial_poses.txt')

# Used for prepare_commands.py
config_compute_nodes = 1 # How many compute nodes are available
config_chunk_size = 500
config_max_trials = 10000
config_command_dir = os.path.join(project_dir, 'data', 'candidates')
config_simulation_path = os.path.join(project_dir, 'scenes', scene_name)

config_collected_data_dir = os.path.join(project_dir, 'output', 'collected')
config_processed_data_dir = os.path.join(project_dir, 'output', 'processed')
config_sample_image_dir = os.path.join(config_processed_data_dir, 'sample_images')
config_sample_pose_dir = os.path.join(config_processed_data_dir, 'sample_poses')

# Used for split_dataset.py, postprocess.py
# train_items contains a list of object classes to be used in train set
config_train_item_list = 'train_items.txt'
config_train_dir = os.path.join(config_processed_data_dir, 'train')
config_test_dir = os.path.join(config_processed_data_dir, 'test')
config_dataset_path = os.path.join(project_dir, 'output', 'grasping.hdf5')
