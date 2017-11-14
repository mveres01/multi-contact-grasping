# grasping-random-pose

This project implements a simulated grasp-and-lift process in V-REP using the Barrett Hand, and interfaces through a python remote API. The primary goal of this project is to collect information on where individual contacts of a multi-fingered hand can be placed on an object, that will lead to a successful grasps. The emphasis on constraining the grasp to specific contacts is to promote future work in fine manipulation.

# Requirements:

* Python 2.7
* V-REP from http://www.coppeliarobotics.com/downloads.html
* Mesh files in either .obj or .stl format. Sample images below come from objects downloaded from http://grasp-database.dkappler.de/. You will need to sign up in order to download them.
* (optional) an Xserver (such as Xorg) if running V-REP in headless mode. Headless mode is when the simulator runs without any GUI present, and is useful when you don't need any immediate visual info about the scene.

## Initialization

```
pip install -r requirements.txt
```

* Assuming your installation directory for V-REP was C:\Program Files\V-REP3, copy vrep.py and vrepConst.py from 
C:\Program Files\V-REP3\V-REP_PRO_EDU\programming\remoteApiBindings\python\python\ to ./lib. Additionally, you will need to copy the following file (use the 32/64-bit versions according to the version of V-REP you've downloaded):
  * __windows__: path/to/vrep/V-REP_PRO_EDU/programming/remoteApiBindings/lib/lib/64Bit/remoteApi.dll
  * __linux__: path/to/vrep/V-REP_PRO_EDU/programming/remoteApiBindings/lib/lib/64Bit/remoteApi.so

* Download and place meshes into the ./data/processed_meshes folder. 

# Collect Grasp Candidates
A pipeline for pre-processing data and up-to and including collecting data has been saved in the ./src director. Assuming everything is set up properly, collecting data should be as simple as running each file sequentially. These steps are described below:

```cd src```

# Collect Grasping Experience
To get a sense of what the simulation is doing, open the scene in V-REP (./scenes/grasp_scene.ttt). You should see a table, a few walls, and a Barrett Hand. 

Start the scene by clicking the play (triangle) button at the top of the screen. The simulation is now waiting for commands from the main python scripts. Run the main collection script in ./src/collect_grasp.py:

```
python collect_grasps.py
```

This will look in the folder ./data/processed_meshes, and run the first mesh it finds. You should see a mesh being imported into the scene, falling onto the table, then after a short delay a gripper should attempt grasps. When the gripper closes, it will check whether all the fingertips of the Barrett Hand are in contact with the object - if so, it will attempt to lift the object to a position above the table / workspace. Successful grasps (where the object remains in the grippers palm) are recorded and saved in a dataset for that specific object in the ./output/collected folder, in the .hdf5 format.

Once grasping experiments have concluded for the object (or all objects if you're running many experiments), run ./src/postprocess_grasps to merge all data into a single file, remove potential duplicates from the dataset, and remove any extreme outliers. Feel free to modify this file to suit your needs.

# Supplementing the Dataset with Images

Now that we have a dataset of grasps that were successful, we can supplement the data by returning to the simulator and collecting images from the camera. Running:

```
python collect_images.py
```

will open up / connect to a running V-REP scene, and begin collecting images using data from ./output/processed/grasping.hdf5. This script uses the state of the simulator at the time of the grasp (i.e. the object pose, gripper pose, angles, etc ...) and restores those parameters before taking an image. The query function is fairly flexible in the arguments it supports, and includes properties for performing scene randomization through:

* Randomizing the number of lights
* Randomizing position of lights
* Randomizing colour & texture of object
* Randomizing colour & texture of table object
* Randomizing camera pose

Refer to the code to see how these are done. Sample RGB images obtained can be seen below, but note that a depth image and binary mask image (i.e. where the image is in the scene) are also recorded.

<p align="center">
  <img src="./docs/0_0_box_poisson_016.png" width="256"/>
  <img src="./docs/0_1_box_poisson_016.png" width="256"/>
  <img src="./docs/0_2_box_poisson_016.png" width="256"/>
  <img src="./docs/0_3_box_poisson_016.png" width="256"/>
</p>
