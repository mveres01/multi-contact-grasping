import os
import csv
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from lib.python_config import (config_command_dir, config_candidate_dir)
from lib.python_config import (config_compute_nodes, config_chunk_size,
                               config_max_trials, config_simulation_path)

from sys import platform
if platform == 'linux' or platform == 'linux2':
    vrep_path = '/scratch/mveres/VREP/vrep.sh'
else:
    vrep_path = 'vrep'



def main():
    """Equally splits grasp candidates to check into equal number of chunks.

    # NOTE: The key is to make sure each chunk only contains a single object.
    # This allows us to avoid continuously loading /similar/different/ objects
    # from memory during the simulation
    """

    if not os.path.exists(config_command_dir):
        os.makedirs(config_command_dir)
    if not os.path.exists(config_candidate_dir):
        os.makedirs(config_candidate_dir)

    files = os.listdir(config_candidate_dir)
    files = [f for f in files if '.txt' in f and ['main' not in f]]

    if len(files) == 0:
        raise Exception('No grasp candidates to distribute across nodes')

    data = [0]*len(files)
    for i, f in enumerate(files):

        fp = os.path.join(config_candidate_dir, f)
        df = pd.read_csv(fp, header=None, index_col=False).values

        # Append the name of the object and number of grasp candidates available
        data[i] = (fp.split(os.path.sep)[-1], df.shape[0])


    # Calculate ranges to equally chunk the grasp candidates
    info = []
    for mesh_object in data:

        # Only going to process a certain number of candidates, but need to
        # make sure each file contains **at most** GLOBAL_MAX_TRIAL's worth
        num_elements= np.minimum(config_max_trials, mesh_object[1])
        n_chunks = int(num_elements / config_chunk_size)
        remainder = num_elements % config_chunk_size

        # This will tell the simulator a range of which lines to use
        indices = [i*config_chunk_size + 1 for i in xrange(n_chunks)]
        indices.append(n_chunks*config_chunk_size+remainder)

        # This will tell the simulator where the lines can be found
        object_name = [mesh_object[0]]*len(indices)
        info += zip(object_name, indices[:-1], indices[1:])


    # Save the command for running the sim. The command is composed of
    # flags such as 'headless mode' (-h), 'quit when done' (-q), 'start'
    # (-s), 'input argument' (-g), and the simulation we will run.
    # Here, we give the input argument as the file contaianing grasp
    # candidates
    info_len = len(info)
    commands = [0]*info_len
    for i, sub_cmd in enumerate(info):

        if info_len >= 10 and i % int(info_len*0.1) == 0:
            print '%d/%d generated'%(i, len(info))
        commands[i] = \
            'export DISPLAY=:1; ulimit -n 4096; %s -h -q -s -g%s -g%s -g%s %s '\
            %(vrep_path, sub_cmd[0], sub_cmd[1], sub_cmd[2], config_simulation_path)


    # To parallelize our data collection routine across different compute nodes,
    # we chunk the commands again. Each compute node will be responsible for
    # running a certain number of commands.  NOTE: If we are performing
    # collection on a single compute node, then num_compute_nodes should be 1.
    file_length = int(len(commands)/config_compute_nodes + 0.5)
    remainder = len(commands) % config_compute_nodes

    for i in xrange(config_compute_nodes):

        if i == config_compute_nodes - 1:
            size = range(i*file_length, (i+1)*file_length + remainder)
        else:
            size = range(i*file_length, (i+1)*file_length)

        # Each 'main' file contains file_length number of chunks of commands
        # that the simulator will need to run.
        main_file = open(os.path.join(config_command_dir, 'main%d.txt'%i), 'wb')
        writer = csv.writer(main_file, delimiter=',')

        for row in size:
            writer.writerow([commands[row]])

        main_file.close()


if __name__ == '__main__':
    main()
