import os
import sys
sys.path.append('..')
from lib.python_config import project_dir

screen_cmd = 'export DISPLAY=:1'
vrep_path = '/scratch/mveres/vrep.sh'
prog_path = os.path.join(project_dir, 'scenes/get_initial_poses.ttt')

try:
    cmd = '%s; %s -h -q -s -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE %s'%\
          (screen_cmd, vrep_path, prog_path)
    os.system(cmd)
except Exception as e:
    print 'Unable to launch get initial poses directly from script.'\
          'Try opening VREP and running manually.'
