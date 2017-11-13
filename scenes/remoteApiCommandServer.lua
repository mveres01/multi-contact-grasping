table.copy = function( t, ... )
    -- See: https://gist.github.com/walterlua/978161
    local copyShallow = function( src, dst, dstStart )
        local result = dst or {}
        local resultStart = 0
        if dst and dstStart then
            resultStart = dstStart
        end
        local resultLen = 0
        if "table" == type( src ) then
            resultLen = #src
            for i=1,resultLen do
                local value = src[i]
                if nil ~= value then
                    result[i + resultStart] = value
                else
                    resultLen = i - 1
                    break;
                end
            end
        end
        return result,resultLen
    end

    local result, resultStart = copyShallow( t )

    local srcs = { ... }
    for i=1,#srcs do
        local _,len = copyShallow( srcs[i], result, resultStart )
        resultStart = resultStart + len
    end

    return result
end

--- Randomize the colour of an object using HSV colour space
-- See: http://www.coppeliarobotics.com/helpFiles/en/regularApi/simSetShapeColor.htm
function randomizeColour(object_handle)

    if object_handle == nil then
        print('Provided object handle not in scene, cannot set colour.')
        return
    end

    local colour_components =
        {sim_colorcomponent_ambient_diffuse,
         sim_colorcomponent_specular,
         sim_colorcomponent_emission,
         sim_colorcomponent_auxiliary}

    local rand_rgb = {math.random(), math.random(), math.random()}

    for i = 1, #colour_components, 1 do
        simSetShapeColor(object_handle, nil, colour_components[i], rand_rgb)
    end
end


--- Randomizes the positions, orientation, and activity of lighting sources.
-- Allow the light sources to deviate a small amount from their original
-- position and orientation. We also "turn off" a light source with a
-- probability p_off
function randomizeLighting(h_lights, light_default_pos, light_default_ori, mag, p_off)

    for i = 1, #h_lights, 1 do

        local new_pos = {light_default_pos[i][1] + math.random()*mag,
                         light_default_pos[i][2] + math.random()*mag,
                         light_default_pos[i][3] + math.random()*mag}
        local new_ori = {light_default_ori[i][1] + math.random()*mag,
                         light_default_ori[i][2] + math.random()*mag,
                         light_default_ori[i][3] + math.random()*mag}
        simSetObjectPosition(h_lights[i], -1, new_pos)
        simSetObjectOrientation(h_lights[i], -1, new_ori)

        -- Give simulation a percent chance of randomly turning a light off
        if math.random() < p_off then
            simSetLightParameters(h_lights[i], 0)
        else
            simSetLightParameters(h_lights[i], 1)
        end
    end
end


--- Randomly assigns an object a with a checkerboard texture.
-- This function will load a texture that exists on file, and apply it to an
-- object using random position and orientation offsets, with either a planar,
-- spherical, cylindrical, or cubic projection mapping.
-- See: http://www.coppeliarobotics.com/helpFiles/en/apiFunctions.htm#simCreateTexture
function randomizeTexture(object_handle, texture_path)

    if object_handle == nil then
        print('Provided object handle not in scene, cannot set texture.')
        return
    end

    -- Remove any textures that may be on the object already
    simSetShapeTexture(object_handle, -1, sim_texturemap_cube, 0, {0, 0})

    local options = 1 + 4 + 8
    local scalingUV = {math.random(), math.random()}
    local xy_g = {math.random(), math.random(), math.random()}

    -- Create the texture
    local h_texture, id, _ = simCreateTexture(texture_path, options, nil,
                                              scalingUV, xy_g, nil, nil)

    if h_texture == nil then
        print('Could not create texture.')
        return
    end

    --- Update the texture. If we don't do this (and a texture has previously
    -- been applied to an object) changes won't get picked up by the
    -- POV-Camera, although those using OpenGL seem to be able to.
    local data = simReadTexture(id, 0)
    simWriteTexture(id, 0, data)

    local mapping_modes = {sim_texturemap_plane, sim_texturemap_cylinder,
                           sim_texturemap_sphere, sim_texturemap_cube}
    local mode = mapping_modes[math.random(#mapping_modes)]

    local position = {math.random(), math.random(), math.random()}
    local orientation = {math.random(), math.random(), math.random()}

    -- Then apply the texture to the object.
    simSetShapeTexture(object_handle, id, mode, options, scalingUV,
                       position, orientation)
    simRemoveObject(h_texture)
end


function resetHand(modelBase)
    -- See e.g. http://www.forum.coppeliarobotics.com/viewtopic.php?t=6700
    local allObjectsToExplore = {modelBase}
    while (#allObjectsToExplore > 0) do
       local obj = allObjectsToExplore[1]
       table.remove(allObjectsToExplore, 1)
       simResetDynamicObject(obj)
       local index = 0
       while true do
          child = simGetObjectChild(obj, index)
          if (child == -1) then
             break
          end
          table.insert(allObjectsToExplore, child)
          index = index+1
       end
    end
end


setGripperPose = function(inInts, inFloats, inStrings, inBuffer)

    local reset_config = inInts[1]

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local h_workspace = simGetIntegerSignal('h_workspace')
    local h_gripper_base = simGetIntegerSignal('h_gripper_base')
    local h_gripper_dummy = simGetIntegerSignal('h_gripper_dummy')
    local h_gripper_config_buffer = simGetIntegerSignal('h_gripper_config_buffer')

    -- Reset the configuration of gripper and pose of grasp dummy
    if reset_config == 1 then
        simSetConfigurationTree(h_gripper_config_buffer)
    end

    simSetObjectMatrix(h_gripper_dummy, h_workspace, pose)

    resetHand(h_gripper_base)

    return {}, {}, {}, ''
end


setPoseByName = function(inInts, inFloats, inStrings, inBuffer)

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local h_part = simGetObjectHandle(inStrings[1])
    local h_workspace = simGetIntegerSignal('h_workspace')

    simSetObjectMatrix(h_part, h_workspace, pose)
    return {}, {}, {}, ''
end


getPoseByName = function(inInts, inFloats, inStrings, inBuffer)

    local name = inStrings[1]
    local h_part = simGetObjectHandle(name)
    local h_wrt = simGetIntegerSignal('h_workspace')
    local matrix = simGetObjectMatrix(h_part, h_wrt)

    return {}, matrix, {}, ''
end


setJointPositionByName = function(inInts, inFloats, inStrings, inBuffer)

    local position = inFloats[1]
    local h_part = simGetObjectHandle(inStrings[1])
    simSetJointPosition(h_part, position)
    return {}, {}, {}, ''
end


setJointKinematicsMode = function(inInts, inFloats, inStrings, inBuffer)

    local joints = simGetStringSignal('h_gripper_joints')
    joints = simUnpackInt32Table(joints)
    local mode = simGetJointMode(joints[1])

    if inStrings[1] == 'forward' then
        mode = sim_jointmode_force
    elseif inStrings[1] == 'inverse' then
        mode = sim_jointmode_ik
    end

    for i = 1, #joints, 1 do
        simSetJointMode(joints[i], mode, 0)
    end
    return {}, {}, {}, ''
end


setKinematicTargetPos = function(inInts, inFloats, inStrings, inBuffer)
    --- Reads current pose of fingertips, and sets kinematic Targets to coincident.
    -- The typical scheme for calling this is:
    -- setGripperFingerTips
    -- setKinematicTargetPos
    -- setGripperInverseKinematics ...

    local h_ik_contacts = simGetStringSignal('h_ik_contacts')
    h_ik_contacts = simUnpackInt32Table(h_ik_contacts)

    local h_ik_targets = simGetStringSignal('h_ik_targets')
    h_ik_targets = simUnpackInt32Table(h_ik_targets)

    local zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i = 1, # h_ik_targets, 1 do
        simSetObjectMatrix(h_ik_targets[i], h_ik_contacts[i], zeros)
    end
    return {}, {}, {}, ''
end

setGripperProperties = function(inInts, inFloats, inStrings, inBuffer)

    local model_properties = {sim_modelproperty_not_collidable,
                              sim_modelproperty_not_measurable,
                              sim_modelproperty_not_renderable,
                              sim_modelproperty_not_detectable,
                              sim_modelproperty_not_cuttable,
                              sim_modelproperty_not_dynamic,
                              sim_modelproperty_not_respondable,
                              sim_modelproperty_not_visible}

    local h_gripper_base = simGetIntegerSignal('h_gripper_base')

    if #inInts ~= #model_properties then
        print('Number of model properties != # input properties.')
        print('setGripperProperties requires the following parameters in order:')
        print('sim_modelproperty_not_collidable')
        print('sim_modelproperty_not_measurable')
        print('sim_modelproperty_not_renderable')
        print('sim_modelproperty_not_detectable')
        print('sim_modelproperty_not_cuttable')
        print('sim_modelproperty_not_dynamic')
        print('sim_modelproperty_not_respondable')
        print('sim_modelproperty_not_visible')
    else
        local props = 0
        for i = 1, #inInts, 1 do
            if inInts[i] == 1 then
                props = props + model_properties[i]
            end
        end
        simSetModelProperty(h_gripper_base, props)
    end
    resetHand(h_gripper_base)

    return {}, {}, {}, ''
end

loadObject = function(inInts, inFloats, inStrings, inBuffer)

    local file_format = inInts[1]

    local mesh_path = inStrings[1]

    local com = {inFloats[1], inFloats[2], inFloats[3]}
    local mass = inFloats[4]
    local inertia = {inFloats[5],  inFloats[6],  inFloats[7],
                     inFloats[8],  inFloats[9],  inFloats[10],
                     inFloats[11], inFloats[12], inFloats[13]}

    -- Load the object and set pose, if we're interested in a new object.
    -- There seems to be an issue with memory and simCreateMeshShape when we
    -- try to delete it from the scene. So for now we'll only load when we
    -- need to.
    if mesh_path ~= prev_mesh_path then

        local h_object = simGetIntegerSignal('h_object')

        -- If we already have a mesh object in the scene, remove it
        if h_object ~= nil then
            simRemoveObject(h_object)
            simClearIntegerSignal('h_object')
        end

        -- First need to try and load the mesh (may contain many components), and
        -- then try and create it. If this doesn't work, quit the sim
        vertices, indices, _, _ = simImportMesh(file_format, mesh_path, 0, 0.001, 1.0)

        h_object = simCreateMeshShape(0, 0, vertices[1], indices[1])
        if h_object == nil then
            print('ERROR: UNABLE TO CREATE MESH SHAPE')
            simStopSimulation()
        end
        simSetIntegerSignal('h_object', h_object)

        simSetObjectName(h_object, 'object')
        simSetObjectInt32Parameter(h_object, sim_shapeintparam_respondable, 1)
        simReorientShapeBoundingBox(h_object, -1)
        simSetShapeMaterial(h_object, simGetMaterialId('usr_sticky'))

        -- Set the object to be renderable & detectable by all sensors
        simSetObjectSpecialProperty(h_object,
            sim_objectspecialproperty_renderable +
            sim_objectspecialproperty_detectable_all)

        --- By default, the absolute reference frame is used. We re-orient the
        -- object to be WRT this frame by default, so don't need an extra mtx.
        simSetShapeMassAndInertia(h_object, mass, inertia, com)

        -- Playing with ODE & vortex engines
        simSetEngineFloatParameter(sim_ode_body_friction, h_object, 0.9)
        simResetDynamicObject(h_object)

        simSetEngineFloatParameter(sim_vortex_body_skinthickness,
                                   h_object, 0.05)
        simSetEngineFloatParameter(sim_vortex_body_primlinearaxisfriction,
                                   h_object, 0.75)
        simSetEngineFloatParameter(sim_vortex_body_seclinearaxisfriction,
                                   h_object, 0.75)
        simSetEngineBoolParameter(sim_vortex_body_randomshapesasterrain,
                                  h_object, true)
        simSetEngineBoolParameter(sim_vortex_body_autoslip, h_object, true)
        simResetDynamicObject(h_object)

    end
    prev_mesh_path = mesh_path

    return {h_object}, {}, {}, ''
end



--- Called from a remote client, and returns rendered images of a scene.
-- This function loads an object, sets its pose, and performs a small set of
-- domain randomization to get different augmentations of the scene.
queryCamera = function(inInts, inFloats, inStrings, inBuffer)

    local res = inInts[1] -- resolution
    local randomize_texture = inInts[2]
    local randomize_colour = inInts[3]
    local randomize_lighting = inInts[4]


    local work2cam = {
        inFloats[1], inFloats[2],  inFloats[3],  inFloats[4],
        inFloats[5], inFloats[6],  inFloats[7],  inFloats[8],
        inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local p_light_off = inFloats[13]
    local p_light_mag = inFloats[14]
    local rgbNearClip = inFloats[15]
    local rgbFarClip = inFloats[16]
    local depthNearClip = inFloats[17]
    local depthFarClip = inFloats[18]
    local fov = inFloats[19]

    local texturePath = inStrings[1]


    -- Get object handles
    local h_object = simGetIntegerSignal('h_object')
    local h_workspace = simGetIntegerSignal('h_workspace')
    local h_camera_mask = simGetIntegerSignal('h_camera_mask')
    local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    local h_camera_depth = simGetIntegerSignal('h_camera_depth')
    local h_table_object = simGetIntegerSignal('h_table_object')
    local h_lights = simGetStringSignal('h_lights')
    local light_default_pos = simGetStringSignal('light_default_pos')
    local light_default_ori = simGetStringSignal('light_default_ori')

    h_lights = simUnpackInt32Table(h_lights)
    light_default_pos = simUnpackTable(light_default_pos)
    light_default_ori = simUnpackTable(light_default_ori)

    if randomize_texture == 1 then
        randomizeTexture(h_object, texturePath)
        randomizeTexture(h_table_object, texturePath)
    end
    if randomize_colour == 1 then
        randomizeColour(h_object)
        randomizeColour(h_table_object)
    end
    if randomize_lighting == 1 then
        randomizeLighting(h_lights, light_default_pos,
                          light_default_ori, p_light_mag, p_light_off)
    end


    -- Set the resolution for each camera
    simSetObjectFloatParameter(h_camera_rgb, sim_visionfloatparam_near_clipping, rgbNearClip)
    simSetObjectFloatParameter(h_camera_rgb, sim_visionfloatparam_far_clipping, rgbFarClip)
    simSetObjectFloatParameter(h_camera_mask, sim_visionfloatparam_near_clipping, rgbNearClip)
    simSetObjectFloatParameter(h_camera_mask, sim_visionfloatparam_far_clipping, rgbFarClip)
    simSetObjectFloatParameter(h_camera_depth, sim_visionfloatparam_near_clipping, depthNearClip)
    simSetObjectFloatParameter(h_camera_depth, sim_visionfloatparam_far_clipping, depthFarClip)

    -- Set Field of View (fov), resolution, and object to visualize
    for _, cam in pairs({h_camera_depth, h_camera_rgb, h_camera_mask}) do
        simSetObjectFloatParameter(cam, sim_visionfloatparam_perspective_angle, fov)
        simSetObjectInt32Parameter(cam, sim_visionintparam_resolution_x, res)
        simSetObjectInt32Parameter(cam, sim_visionintparam_resolution_y, res)
        simSetObjectMatrix(cam, h_workspace, work2cam)

        -- Allow camera to capture all renderable objects in scene
        simSetObjectInt32Parameter(cam, sim_visionintparam_entity_to_render, -1)
    end
    simSetObjectInt32Parameter(h_camera_mask, sim_visionintparam_entity_to_render, h_object)

    --- We only need a single picture of the object, so we need to
    -- make sure that the simulation knows to render it now
    simHandleVisionSensor(h_camera_depth)
    simHandleVisionSensor(h_camera_rgb)
    simHandleVisionSensor(h_camera_mask)

    local depth_image = simGetVisionSensorDepthBuffer(h_camera_depth)
    local colour_image = simGetVisionSensorImage(h_camera_rgb)
    local mask_image = simGetVisionSensorImage(h_camera_mask)

    local all_images = table.copy({}, depth_image,  colour_image, mask_image)

    return {}, all_images, {}, ''
end


--- Called from a remote client, and returns rendered images of a scene.
-- This function loads an object, sets its pose, and performs a small set of
-- domain randomization to get different augmentations of the scene.
displayGrasp = function(inInts, inFloats, inStrings, inBuffer)

    local reset_containers = inInts[1]

    local frame_world2work = {
        inFloats[1], inFloats[2],  inFloats[3],  inFloats[4],
        inFloats[5], inFloats[6],  inFloats[7],  inFloats[8],
        inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local frame_work2cam = {
        inFloats[13], inFloats[14], inFloats[15], inFloats[16],
        inFloats[17], inFloats[18], inFloats[19], inFloats[20],
        inFloats[21], inFloats[22], inFloats[23], inFloats[24]}

    local grasp_wrt_cam = {}
    for i = 25, 25 + 17, 1 do
        table.insert(grasp_wrt_cam, inFloats[i])
    end

    -- Get handles to the objects we want to move
    local h_workspace = simGetIntegerSignal('h_workspace')
    local h_camera_mask = simGetIntegerSignal('h_camera_mask')
    local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    local h_camera_depth = simGetIntegerSignal('h_camera_depth')
    local sphereContainer = simGetIntegerSignal('sphereContainer')
    local lineContainerRed = simGetIntegerSignal('lineContainerRed')
    local lineContainerGreen = simGetIntegerSignal('lineContainerGreen')
    local lineContainerBlue = simGetIntegerSignal('lineContainerBlue')
    local lineContainers = {lineContainerRed, lineContainerGreen, lineContainerBlue}

    simSetObjectMatrix(h_camera_mask, h_workspace, frame_work2cam)
    simSetObjectMatrix(h_camera_rgb, h_workspace, frame_work2cam)
    simSetObjectMatrix(h_camera_depth, h_workspace, frame_work2cam)

    if reset_containers == 1 then
        simAddDrawingObjectItem(lineContainerRed, nil)
        simAddDrawingObjectItem(lineContainerGreen, nil)
        simAddDrawingObjectItem(lineContainerBlue, nil)
    end

    local contact0 = {grasp_wrt_cam[1], grasp_wrt_cam[2], grasp_wrt_cam[3]}
    local contact1 = {grasp_wrt_cam[4], grasp_wrt_cam[5], grasp_wrt_cam[6]}
    local contact2 = {grasp_wrt_cam[7], grasp_wrt_cam[8], grasp_wrt_cam[9]}
    local positions = {contact0, contact1, contact2}

    local normal0 = {grasp_wrt_cam[10], grasp_wrt_cam[11], grasp_wrt_cam[12]}
    local normal1 = {grasp_wrt_cam[13], grasp_wrt_cam[14], grasp_wrt_cam[15]}
    local normal2 = {grasp_wrt_cam[16], grasp_wrt_cam[17], grasp_wrt_cam[18]}
    local orientations = {normal0, normal1, normal2}


    -- Draw the contact normal
    for i = 1, #positions, 1 do

        local point2 = {
            positions[i][1]+0.1*orientations[i][1],
            positions[i][2]+0.1*orientations[i][2],
            positions[i][3]+0.1*orientations[i][3]}

        local frame_world2cam = simMultiplyMatrices(frame_world2work, frame_work2cam)

        local world2point1 = simMultiplyVector(frame_world2cam, positions[i])
        local world2point2 = simMultiplyVector(frame_world2cam, point2)

        simAddDrawingObjectItem(lineContainers[i],
               {world2point1[1], world2point1[2], world2point1[3],
                world2point2[1], world2point2[2], world2point2[3]})
    end

    return {}, {}, {}, ''
end



if (sim_call_type == sim_childscriptcall_initialization) then

    simClearStringSignal('grasp_candidate')
    simClearStringSignal('drop_object')

    -- ----------------- SIMULATION OBJECT HANDLES ----------------------------
    --- All object handles will be prefixed with a 'h_' marker.
    --- All handles to be used in the simulation will be collected here, and
    -- stored as "global" variables. Only this local function will have write
    -- access however.
    --- Functions called by a remote client will have access to the global
    -- variables, but all subroutines called by those functions will need to
    -- have these parameters passed in as values.

    --local h_gripper_base = simGetObjectHandle('ROBOTIQ_85')
    local h_gripper_base = simGetObjectHandle('BarrettHand')


    local h_gripper_dummy = simGetObjectHandle('gripper_dummy')
    local h_table_object = simGetObjectHandle('customizableTable_tableTop')
    local h_camera_mask = simGetObjectHandle('kinect_mask')
    local h_camera_rgb = simGetObjectHandle('kinect_rgb')
    local h_camera_depth = simGetObjectHandle('kinect_depth')
    local h_camera_dummy = simGetObjectHandle('camera_dummy')
    local h_gripper_config_buffer = simGetConfigurationTree(h_gripper_base)
    local h_gripper_prox = simGetObjectHandle('BarrettHand_attachProxSensor')
    local h_workspace = h_table_object
    local h_lights = {simGetObjectHandle('DefaultLightA'),
                      simGetObjectHandle('DefaultLightB'),
                      simGetObjectHandle('DefaultLightC'),
                      simGetObjectHandle('DefaultLightD')}

    local h_ik_group = simGetIkGroupHandle('IKGroup')
    local h_ik_contacts = {simGetObjectHandle('dummyTip0'),
                           simGetObjectHandle('dummyTip1'),
                           simGetObjectHandle('dummyTip2')}
    local h_ik_targets = {simGetObjectHandle('dummyTarget0'),
                          simGetObjectHandle('dummyTarget1'),
                          simGetObjectHandle('dummyTarget2')}


    --- Given the name of the root of the gripper model, we traverse through all
    -- components to find the contact points.
    local h_gripper_contacts = {}
    local h_gripper_respondable = {}
    local h_gripper_joints = {}

    -- Record initial gripper configuration so it can be reset for each trial.
    local h_gripper_all = simGetObjectsInTree(h_gripper_base)

    for k = 1, #h_gripper_all, 1 do
        local _, res =simGetObjectInt32Parameter(h_gripper_all[k],
                                                 sim_shapeintparam_respondable)
        if res ~= 0 then
            table.insert(h_gripper_respondable, h_gripper_all[k])
        end

        local name = simGetObjectName(h_gripper_all[k])
        if string.match(name, 'respondableContact') then
            table.insert(h_gripper_contacts, h_gripper_all[k])
        end

        if string.match(name, 'joint') then
            table.insert(h_gripper_joints, h_gripper_all[k])
        end
    end

    -- ------------------- VISUALIZATION HANDLES ---------------------------

    --- In V-REP, there are a few useful modules for drawing to screen, which
    -- makes things like debugging much easier.
    local display_num_points = 5000;
    local display_point_density = 0.001;
    local display_point_size = 0.005;
    local display_vector_width = 0.001;
    local black = {0, 0, 0}; local purple = {1, 0, 1};
    local blue = {0, 0, 1}; local red = {1, 0, 0}; local green = {0, 1, 0};

    -- Used for plotting spherical-type objects
    local sphereContainer = simAddDrawingObject(
        sim_drawing_spherepoints, display_point_size,
        display_point_density, -1, display_num_points,
        black, black, black, light_blue)

    -- Used for plotting line-type objects
    local lineContainer = simAddDrawingObject(
        sim_drawing_lines, display_vector_width,
        display_point_density, -1, display_num_points,
        black, black, black, purple)
    local lineContainerRed = simAddDrawingObject(
        sim_drawing_lines, display_vector_width,
        display_point_density, -1, display_num_points,
        black, black, black, red)
    local lineContainerGreen = simAddDrawingObject(
        sim_drawing_lines, display_vector_width,
        display_point_density, -1, display_num_points,
        black, black, black, green)
    local lineContainerBlue = simAddDrawingObject(
        sim_drawing_lines, display_vector_width,
        display_point_density, -1, display_num_points,
        black, black, black, blue)

    -- Reset the containers so they hold no elements
    simAddDrawingObjectItem(sphereContainer, nil)
    simAddDrawingObjectItem(lineContainer, nil)
    simAddDrawingObjectItem(lineContainerRed, nil)
    simAddDrawingObjectItem(lineContainerGreen, nil)
    simAddDrawingObjectItem(lineContainerBlue, nil)

    -- ------------- MISC GRIPPER / MODEL PARAMETERS --------------------

    local num_collision_thresh = 75

    -- Used for lifting the object
    local max_vel_accel_jerk = {0.2, 0.2, 0.2,
                                0.05, 0.05, 0.05,
                                0.2, 0.2, 0.2};

    gripper_prop_static =
        sim_modelproperty_not_dynamic    +
        sim_modelproperty_not_renderable +
        sim_modelproperty_not_collidable  +
        sim_modelproperty_not_respondable

    gripper_prop_visible =
        sim_modelproperty_not_renderable +
        sim_modelproperty_not_measurable

    gripper_prop_invisible =
        sim_modelproperty_not_collidable  +
        sim_modelproperty_not_renderable  +
        sim_modelproperty_not_visible     +
        sim_modelproperty_not_respondable +
        sim_modelproperty_not_dynamic

    simSetModelProperty(h_gripper_base, gripper_prop_invisible)

    -- ---------------- HANDLES FOR SCENE RANDOMIZATION -----------------------

    -- Finally, always start the lights at these default positions
    local light_default_pos = {{2.685, 0.4237, 4.2},
                               {2.5356, 0.7, 4.256},
                               {2.44, 0.4022, 4.3476},
                               {2.44, 0.402, 4.3477}}

    -- These are the default orientations of the cameras, when reading off of
    -- the screen, but future functions (i.e. simSetObjectOrientation) expects
    -- input angles to be in radians.
    local light_default_ori = {{-122.7, -64, 130.6},
                               {102.8, 1.897, 19.80},
                               {-106.75, 52.73, -145.27},
                               {-180, 0, 0}}

    for i, light in ipairs(light_default_ori) do
        light_default_ori[i] = {(3.1415 * light[1] / 180.),
                                (3.1415 * light[2] / 180.),
                                (3.1415 * light[3] / 180.)}
    end

    -- ------- SAVE HANDLES AS GLOBAL VARIABLES (i.e. as signals) -------------

    simSetIntegerSignal('h_table_object', h_table_object)
    simSetIntegerSignal('h_gripper_dummy', h_gripper_dummy)
    simSetIntegerSignal('h_gripper_palm', h_gripper_dummy)
    simSetIntegerSignal('h_gripper_base', h_gripper_base)
    simSetIntegerSignal('h_gripper_config_buffer', h_gripper_config_buffer)
    simSetIntegerSignal('h_workspace', h_workspace)
    simSetIntegerSignal('h_camera_mask', h_camera_mask)
    simSetIntegerSignal('h_camera_rgb', h_camera_rgb)
    simSetIntegerSignal('h_camera_depth', h_camera_depth)
    simSetIntegerSignal('h_camera_dummy', h_camera_dummy)
    simSetIntegerSignal('h_gripper_prox', h_gripper_prox)
    simSetIntegerSignal('h_ik_group', h_ik_group)

    simSetIntegerSignal('sphereContainer', sphereContainer)
    simSetIntegerSignal('lineContainer', lineContainer)
    simSetIntegerSignal('lineContainerRed', lineContainerRed)
    simSetIntegerSignal('lineContainerGreen', lineContainerGreen)
    simSetIntegerSignal('lineContainerBlue', lineContainerBlue)
    simSetIntegerSignal('gripper_prop_static', gripper_prop_static)
    simSetIntegerSignal('gripper_prop_visible', gripper_prop_visible)
    simSetIntegerSignal('gripper_prop_invisible', gripper_prop_invisible)
    simSetFloatSignal('num_collision_thresh', num_collision_thresh)

    -- Tables of handles are saved as strings (packed ints/floats)
    simSetStringSignal('h_gripper_contacts', simPackInt32Table(h_gripper_contacts))
    simSetStringSignal('h_gripper_respondable', simPackInt32Table(h_gripper_respondable))
    simSetStringSignal('h_gripper_all', simPackInt32Table(h_gripper_all))
    simSetStringSignal('h_gripper_joints', simPackInt32Table(h_gripper_joints))
    simSetStringSignal('h_ik_contacts', simPackInt32Table(h_ik_contacts))
    simSetStringSignal('h_ik_targets', simPackInt32Table(h_ik_targets))
    simSetStringSignal('h_lights', simPackInt32Table(h_lights))
    simSetStringSignal('max_vel_accel_jerk', simPackFloatTable(max_vel_accel_jerk))
    simSetStringSignal('light_default_pos', simPackTable(light_default_pos))
    simSetStringSignal('light_default_ori', simPackTable(light_default_ori))

    -- Check where the data will come from
    local PORT_NUM = simGetStringParameter(sim_stringparam_app_arg1)

    if PORT_NUM == '' then
        PORT_NUM = 19999 -- default
        simExtRemoteApiStart(PORT_NUM)
    end
    print('port num: ', PORT_NUM)
end
