# multi-contact-grasping

This project implements a simulated grasp-and-lift process in V-REP using the Barrett Hand, with an interface through a python remote API.

# Overview

The primary goal of this project is to collect information on where individual contacts of a multi-fingered hand can be placed on an object, that will lead to successful grasps. The emphasis on constraining the grasp to specific contacts is to promote future work in learning fine manipulation skills.

This project is an extension of [this repo](https://github.com/mveres01/grasping), largely focusing on simplifying the pipeline, integration through a python remote API, and image-domain randomization.

## Recorded Information

TODO

## Image Domain Randomization

TO FINISH

Randomization of images is done with respect to the following scene elements:

* Number of lights
* Position of lights
* Colour & texture of object
* Colour & texture of table object
* Camera pose

<p align="center">
  <img src="./docs/0_0_box_poisson_016.png" width="256"/>
  <img src="./docs/0_1_box_poisson_016.png" width="256"/>
  <img src="./docs/0_2_box_poisson_016.png" width="256"/>
  <img src="./docs/0_3_box_poisson_016.png" width="256"/>
  <img src="./docs/0_4_box_poisson_016.png" width="256"/>
</p>


# Requirements:

* Python 2.7
* V-REP (http://www.coppeliarobotics.com/downloads.html)
* Mesh files in either .obj or .stl format. Note that meshes with complicated structures will have a negative impact on the dynamics simulation, so pure / convex meshes are preferred.
* (optional) an Xserver if running V-REP in headless mode (i.e. no GUI). 

## Initialization

TODO: Add requirements.txt

```
pip install -r requirements.txt
```

* Copy __vrep.py__ and __vrepConst.py__ from 
_path/to/vrep/V-REP_PRO_EDU/programming/remoteApiBindings/python/python/_ to _./lib_. You will also need to copy the following library (use the 32/64-bit version depending on what version of V-REP you've downloaded):
  * __windows__: path/to/vrep/V-REP_PRO_EDU/programming/remoteApiBindings/lib/lib/64Bit/remoteApi.dll
  * __linux__: path/to/vrep/V-REP_PRO_EDU/programming/remoteApiBindings/lib/lib/64Bit/remoteApi.so

* Download and place meshes into the ./data/meshes folder. A sample mesh (cube.stl) has been provided.

# Collect Grasping Experience
Open _./scenes/grasp_scene.ttt_ in V-REP. You should see a table, a few walls, and a Barrett Hand. 

Start the scene by clicking the play (triangle) button at the top of the screen. Run the main collection script in ./src/collect_grasp.py:

```
cd src
python collect_grasps.py
```

This will look in the folder _./data/meshes_, and run the first mesh it finds. You should see a mesh being imported into the scene, falling onto the table, then after a short delay a gripper should begin to attempt grasping it. When the gripper closes, it will check whether all the fingertips of the Barrett Hand are in contact with the object - if so, it will attempt to lift the object to a position above the table. Successful grasps (where the object remains in the grippers palm) are recorded and saved in an hdf5 dataset for that specific object in the _./output/collected_ folder.

The pose of the object is kept fixed on each grasp attempt. 

Once grasping experiments have concluded for the object (or all objects if you're running many experiments), run 

```
python postprocess_grasps.py
```
to merge all data into a single file, remove potential duplicates from the dataset, and remove any extreme outliers. Feel free to modify this file to suit your needs. Data will be saved to _./output/grasping.hdf5_. 

## A Few things to Note:
1. __The object is static during the pregrasp, and dynamically simulated during the lift__: This avoids potentially moving the object before the fingers come into contact with it.
2. __The same object pose is used for each grasp attempt__: This avoids instances where an object may accidentally fall off the table, but can be removed as a constraint from the main script.
3. __A grasp is successful if the object is in the grippers palm at the height of the lift__: A proximity sensor attached to the palm is used to record whether it detects an object in a nearby vicinity. A threshold is also specified on the number of contacts between the gripper and the object, which helps limit inconsistencies in the simulation dynamics.
4. __Images are captured from a seperate script after simulations have finished__: To avoid introducing additional complexities into the collection script, images are collected after the main grasp collection has finished. This script will ultimately restore the state of the object and gripper during the grasp, and will position a camera randomly to collect images.

# Supplementing the Dataset with Images

Once data collection has finished, we can supplement the dataset by running:

```
cd src
python collect_images.py
```

which will open up or connect to a running V-REP scene, and begin collecting images using data from _./output/processed/grasping.hdf5_. This script uses the state of the simulator at the time of the grasp (i.e. the object pose, gripper pose, angles, etc ...) and restores those parameters before taking an image. 

Refer to the code to see how these are done.
