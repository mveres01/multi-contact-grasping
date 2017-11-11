--math.randomseed(os.time())
math.randomseed(1234)

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


function table.val_to_str ( v )
	---From http://lua-users.org/wiki/TableUtils
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end


function table.key_to_str ( k )
	---From http://lua-users.org/wiki/TableUtils
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end


function table.tostring( tbl )
	---From http://lua-users.org/wiki/TableUtils
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end


---	Checks that the grasp is valid: the fingertips are contacting the object,
-- and records the position, force, and normal if everything is good.
function checkContacts(contactPoints, h_object)

    local info = {}
	local na = {'nil', 'nil', 'nil'}
	local bad_info = {{na, na, na}, {na, na, na}, {na, na, na}}

    for i = 1, #contactPoints, 1 do

		-- h = handles of colliding objects
		local h, position, force, normal = simGetContactInfo(
			sim_handle_all, contactPoints[i], 1 + sim_handleflag_extended)

        -- Check if the contact point is in contact with the object
        if (h == nil) or (h[1] ~= contactPoints[i]) or (h[2] ~= h_object) then
			return false, bad_info
		else
            table.insert(info, {position, force, normal})
        end
    end
    return true, info
end


--- Checks how many collisions the gripper is making with the table and object
-- being grasped. If we have a significant amount, it is likely that one of
-- the grasp poses has placed the gripper at some intersection of the object.
function checkCollisions(h_object, h_table_object, h_gripper_respondable,
						 h_gripper_contacts, num_collision_thresh)

	local total_collisions = 0

	for k = 1, #h_gripper_respondable, 1 do

		-- simCheckCollisionEx returns the number of collision points
		local tbl = simCheckCollisionEx(h_gripper_respondable[k], h_table_object)
		local obj = simCheckCollisionEx(h_gripper_respondable[k], h_object)
		total_collisions = total_collisions + tbl + obj


		--- Check whether the contact between object and gripper is from
		-- the fingertips, or other parts of the gripper (don't want this)
		local contact_obj = false

		local flag = (tbl > 0) or (obj > 0)
		if flag then
			for l = 1, #h_gripper_contacts, 1 do
				if h_gripper_respondable[k] == h_gripper_contacts[l] then
					contact_obj = true; break;
				end
			end
		end

		if (contact_obj == false and flag == true) or
			total_collisions > num_collision_thresh then
			return true
		end

	end

	--- If we got this far, the rest of the gripper shouldn't be contacting
	-- anything else, and we can deduce the grasp is successful
	return false
end


--- Checks how many collisions the gripper is making with the table and object
-- being grasped. If we have a significant amount, it is likely that one of
-- the grasp poses has placed the gripper at some intersection of the object.
function countCollisions(h_object, h_table_object, h_gripper_respondable,
						 h_gripper_contacts, num_collision_thresh)

	local total_collisions = 0

	for k = 1, #h_gripper_respondable, 1 do

		-- simCheckCollisionEx returns the number of collision points
		local tbl = simCheckCollisionEx(h_gripper_respondable[k], h_table_object)
		local obj = simCheckCollisionEx(h_gripper_respondable[k], h_object)
		total_collisions = total_collisions + tbl + obj

		if total_collisions > num_collision_thresh then
			return true
		end
	end

	--- If we got this far, the rest of the gripper shouldn't be contacting
	-- anything else, and we can deduce the grasp is successful
	return false
end


--- Collects information on everything in the scene.
-- Return is a header that denotes [object, #values] pairs, and a float table
-- that keeps track of each configuration.
function getGraspInformation(all_in_contact, contactsInfo, object_tree,
							 h_gripper_palm, h_object, h_workspace)

	local names = {}
    local configurations = {}

    if all_in_contact == true then
		all_in_contact={1}
	else
		all_in_contact = {0}
	end

	-- Save the forces, torques, and normals of the contact points
	local world2work = simGetObjectMatrix(h_workspace, -1)
	for i = 1, #contactsInfo, 1 do

		-- Convert contact information from world to workspace frame
		for j, vec in ipairs(contactsInfo[i]) do
			if vec[1] ~= 'nil' and vec[2] ~= 'nil' and vec[3] ~= 'nil' then
				contactsInfo[i][j] = simMultiplyVector(world2work, vec)
			end
		end

        names = table.copy(names,
			{'work2contact'..(i-1), #contactsInfo[i][1],
			 'work2force'..(i-1),   #contactsInfo[i][2],
			 'work2normal'..(i-1),  #contactsInfo[i][3]})

        configurations = table.copy(configurations,
			contactsInfo[i][1],
			contactsInfo[i][2],
			contactsInfo[i][3])
    end

	-- Check the number of collisions and get pose matrices for gripper components
    for _, component in pairs(object_tree) do

        local name = simGetObjectName(component)
        if  string.find(name, 'joint') ~= nil then

            local intrinsic_angle = simGetJointPosition(component)
            names = table.copy(names, {name..'_pos',1})
            configurations = table.copy(configurations, {intrinsic_angle})

        end
    end

	local frame_work2obj = simGetObjectMatrix(h_object, h_workspace)
	local frame_work2palm = simGetObjectMatrix(h_gripper_palm, h_workspace)
	local frame_work2palm_orient = simGetEulerAnglesFromMatrix(frame_work2palm)
	local frame_work2palm_pos = {frame_work2palm[4],
									frame_work2palm[8],
									frame_work2palm[12]}

    local frame_world2work = simGetObjectMatrix(h_workspace, -1)

    -- Also save the mass, inertia, and center of mass of the object
    local mass, inertia, com = simGetShapeMassAndInertia(h_object)
    local material_id  = simGetShapeMaterial(h_object)
    local finger_angle = simGetScriptSimulationParameter(sim_handle_all, 'fingerAngle')

	-- returned in absolute coordinate
	local lin_velocity, ang_velocity = simGetObjectVelocity(h_object)

    -- Add a "header" telling us how much information is in each component
    names = table.copy(names,
		{'all_in_contact', 1},
		{'frame_work2palm', #frame_work2palm},
		{'frame_work2palm_orient', #frame_work2palm_orient},
		{'frame_work2palm_pos', #frame_work2palm_pos},
		{'frame_world2work', #frame_world2work},
		{'frame_work2obj', #frame_work2obj},
		{'mass_wrt_world', 1},
		{'inertia_wrt_world', #inertia},
		{'com_wrt_world', #com},
		{'finger_angle', #finger_angle},
		{'material_id', 1},
		{'objectLinearVelocity', #lin_velocity},
		{'objectAngularVelocity', #ang_velocity}
		)

    -- Add all the data into a single table
    configurations = table.copy(configurations,
		all_in_contact,
		frame_work2palm,
		frame_work2palm_orient,
		frame_work2palm_pos,
		frame_world2work,
		frame_work2obj,
		{mass},
		inertia,
		com,
		finger_angle,
		{material_id},
		lin_velocity,
		ang_velocity
		)

	-- Quick sanity check that len(names) == len(configuration)
	local sum = 0
    for i = 2, #names, 2 do sum = sum + names[i] end
	if sum ~= #configurations then
		print('Mismatch in the number of elements being recorded. ' ..
			  'Check values are being recorded correctly.')
		simStopSimulation()
	end
    return names, configurations
end


-- MAIN FUNCTION
threadCollectionFunction = function()

	-- For for initialization from remoteApiCommandServer
	simWaitForSignal('h_object')

	local h_object = simGetIntegerSignal('h_object')
	local h_workspace = simGetIntegerSignal('h_workspace')
	local h_table_object = simGetIntegerSignal('h_table_object')
	local h_gripper_palm = simGetIntegerSignal('h_gripper_palm')
	local h_gripper_base = simGetIntegerSignal('h_gripper_base')
	local h_gripper_dummy = simGetIntegerSignal('h_gripper_dummy')
	local h_gripper_prox = simGetIntegerSignal('h_gripper_prox')

	local prop_invisible = simGetIntegerSignal('gripper_prop_invisible')
	local prop_visible = simGetIntegerSignal('gripper_prop_visible')

	local h_gripper_all = simGetStringSignal('h_gripper_all')
	local h_gripper_contacts = simGetStringSignal('h_gripper_contacts')
	local h_gripper_respondable = simGetStringSignal('h_gripper_respondable')
	local max_vel_accel_jerk = simGetStringSignal('max_vel_accel_jerk')


	h_gripper_all = simUnpackInt32Table(h_gripper_all)
	h_gripper_contacts = simUnpackInt32Table(h_gripper_contacts)
	h_gripper_respondable = simUnpackInt32Table(h_gripper_respondable)
	max_vel_accel_jerk = simUnpackFloatTable(max_vel_accel_jerk)

	local num_collision_thresh = simGetFloatSignal('num_collision_thresh')



	while simGetSimulationState() ~= sim_simulation_advancing_abouttostop do

		local finger_angle = simGetIntegerSignal('run_grasp_attempt')
        if finger_angle ~= nil then

			simClearIntegerSignal('run_grasp_attempt')
			simSetScriptSimulationParameter(sim_handle_all, 'fingerAngle', finger_angle)

			simSetModelProperty(h_gripper_base, prop_visible)
			simResetDynamicObject(h_gripper_base)
			simSwitchThread()

			-- These will save the results of the grasp attemp; a successful
			-- or failed grasp will be encoded here; otherwise we return
			-- default values
			header = {-1}
			pregrasp = {-1}
			postgrasp = {-1}
			h_object = simGetIntegerSignal('h_object')


			-- Check that only the fingertips are contacting the object,
			local is_collision = checkCollisions(
				h_object, h_table_object, h_gripper_respondable,
				h_gripper_contacts, num_collision_thresh)

			if is_collision == false then

				-- ---------------- GRASP THE OBJECT ---------------------

				simSetIntegerSignal('closeGrasp', 1)
				simWaitForSignal('grasp_done')
				simClearIntegerSignal('grasp_done')
				simWait(3)

				-- Send a signal to hold the grasp while we attempt a lift
				simClearIntegerSignal('closeGrasp')
				simSetIntegerSignal('holdGrasp', 1)
				simWaitForSignal('grasp_done')
				simClearIntegerSignal('grasp_done')
				simSwitchThread()

				-- Check that all gripper fingertips are touching
				local is_contact, contact_where =
					checkContacts(h_gripper_contacts, h_object)

				if is_contact then

					-- Save the Pre-grasp data
					header, pregrasp = getGraspInformation(
						 is_contact, contact_where, h_gripper_all,
						 h_gripper_palm, h_object, h_workspace)
					simSwitchThread()

					-- ---------------- LIFT THE OBJECT -------------------

					-- Make object dynamic
					simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 0)
					simResetDynamicObject(h_object)
					simSwitchThread()

					--- Define a path that leads from the current gripper
					-- position to a final gripper position. To perform
					-- this 'lift' action, we manually follow the path by
					-- manually setting the new position at each time step.
					local p0 = simGetObjectPosition(h_gripper_palm, -1)
					local pt = simGetObjectPosition(h_workspace, -1)
					local targetPosVel = {pt[1], pt[2], pt[3] + 0.5, 0, 0, 0}
					local posVelAccel = {p0[1], p0[2], p0[3], 0, 0, 0, 0, 0, 0, 0}

					local rmlHandle = simRMLPos(3, 0.001, -1, posVelAccel,
												max_vel_accel_jerk,
												{1,1,1}, targetPosVel)

					-- Incrementally move the hand along the trajectory
					local res = 0
					while res == 0 do
						dt = simGetSimulationTimeStep()
						res, posVelAccel, sync = simRMLStep(rmlHandle,dt)
						simSetObjectPosition(h_gripper_palm, -1, posVelAccel)
						simSwitchThread()
					end
					simRMLRemove(rmlHandle)
					simSwitchThread()



					--- Three conditions for a successful grasp:
					-- 1. Proximity sensor detects the object, and
					-- 2. Gripper is not intersecting object
					-- 3. Object is not contacting the table
					local _, contact_where =
						checkContacts(h_gripper_contacts, h_object)

					simHandleProximitySensor(h_gripper_prox)
					is_prox, _, _ = simCheckProximitySensor(h_gripper_prox, h_object)

					local is_collision = countCollisions(
						h_object, h_table_object, h_gripper_respondable,
						h_gripper_contacts, 50)

					local r2 = simCheckCollision(h_object, h_table_object)
					success = (is_prox == 1) and (not is_collision) and (r2 == 0)

					-- Get postgrasp information
					_, postgrasp = getGraspInformation(
						success, contact_where, h_gripper_all,
						h_gripper_palm, h_object, h_workspace)
					simSwitchThread()

					print('Grasp Successful? ', success)
				end -- Lifting object

				-- Let the object fall onto table
				simSetIntegerSignal('clearGrasp', 1)
				simWaitForSignal('grasp_done')
				simClearIntegerSignal('grasp_done')
				simSwitchThread()

				if is_touching then
					simWait(3)
				end

			end  -- Grasping object

			--- Set the object static so it doesn't move before contact
			simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 1)
			simResetDynamicObject(h_object)
			simSwitchThread()

			simSetModelProperty(h_gripper_base, prop_invisible)
			simResetDynamicObject(h_gripper_base)

			simSetStringSignal('header', table.tostring(header))
			simSetStringSignal('pregrasp', simPackFloatTable(pregrasp))
			simSetStringSignal('postgrasp', simPackFloatTable(postgrasp))
		end  -- Check gripper collisions

		simSwitchThread()
    end -- loop infinite
end






simSetBooleanParameter(sim_boolparam_dynamics_handling_enabled, true)

-- Here we execute the regular thread code:
res,err=xpcall(threadCollectionFunction, function(err) return debug.traceback(err) end)
if not res then
	print('Error: ', err)
	simAddStatusbarMessage('Lua runtime error: '..err)
end
print('Done simulation!')
