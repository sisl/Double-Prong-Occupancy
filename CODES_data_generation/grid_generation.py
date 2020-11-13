import numpy as np
import os
import hickle as hkl
import time
import grid_generation_utils
import ground_segmentation

# parameters
master_folder = "../DATA_preprocessed"

res = 1/3 
cells_x = 128
cells_y = 128
buf_veh = 0.5 # m
height_max = 3.0  # m -> To filter out possible LiDAR noise
height_lidar = 0.200

print("--------START--------")
for num in range(0, 32): 
	training_folder_name = "training_00" + "{:02d}".format(num)
	training_folder = os.path.join(master_folder, training_folder_name)
	segment_files_name = sorted(os.listdir(training_folder), 
		key = lambda y: int(y.split("_")[0]))

	print("Number of files in {}".format(training_folder_name), len(segment_files_name))

	for segment_file_name in segment_files_name: 
		print("Segment", segment_file_name)
		start_time_segment = time.time()
		npz_path = os.path.join(training_folder, segment_file_name, "points.npz")
		extrinsic_path = os.path.join(training_folder, segment_file_name, "extrinsic.npz")
		ego_pose_path = os.path.join(training_folder, segment_file_name, "ego_pose.npz")
		
		npz_file = np.load(npz_path, allow_pickle = True)
		static_points = npz_file["static"] # static target labels for all frames in this segment
		dynamic_points = npz_file["dynamic"] # dyrynamic target labels for all frames in this segment
		all_points = npz_file["all"] 
		
		extrinsic_file = np.load(extrinsic_path, allow_pickle = True)
		extrinsic = extrinsic_file["extrinsic"]
		ego_pose_file = np.load(ego_pose_path, allow_pickle = True)
		ego_pose_all = ego_pose_file["ego_pose"]

		full_sensorgrids = []
		static_gseg_sensorgrids = []
		dynamic_sensorgrids = [] 

		for frame_index in range(0, static_points.size): # loop through each frame in a segment
			print("Frame index", frame_index)
			static_target_pc = static_points[frame_index]
			dynamic_target_pc = dynamic_points[frame_index] 
			all_pc = all_points[frame_index]

			# Filter out height > height_max
			all_pc_w_intensity = grid_generation_utils.height_filter(all_pc, height_max)

			# Filter for static target labels + static environment pc (filter out dynamic pc from all_pc)
			static_pc_w_intensity = grid_generation_utils.get_static_env(dynamic_target_pc, all_pc_w_intensity)

			# Ground segmentation
			# Before ground segmentation, convert pc from vehicle frame to lidar sensor frame
			# Convert to homogeneous coordinates (3, n) -> (4,n)
			static_pc_homogeneous = np.concatenate((static_pc_w_intensity[:, :-1].T, np.ones((1, static_pc_w_intensity.shape[0]))), axis = 0 )

			# Transform to lidar sensor frame
			static_lidar = np.matmul(np.linalg.inv(extrinsic), static_pc_homogeneous)
			static_lidar = static_lidar[:-1].T # (n, 3)
			
			static_lidar_gseg = ground_segmentation.ground_seg(static_lidar, height_lidar,  res = 1/3, s = 0.12).T
			static_lidar_gseg = np.concatenate((static_lidar_gseg, 
				np.ones((1, static_lidar_gseg.shape[1]))), axis = 0)

			static_gseg = np.matmul(extrinsic, static_lidar_gseg)
			static_gseg = static_gseg[:-1].T 
			
			static_pc = static_pc_homogeneous[:-1].T # (n, 3)
		
			# Getting full grid: all_pc without ground
			if dynamic_target_pc.shape == (1,) and dynamic_target_pc == -99:
				dynamic_target_pc = np.array([500,500,1,1]).reshape(1, 4) # makeshift pointcloud that is very far away
				print("exception case - no dynamic_target pc")

			if static_target_pc.shape == (1,) and static_target_pc == -99:
				static_target_pc = np.array([500,500,1,1]).reshape(1, 4) # makeshift pointcloud that is very far away
				print("exception case - no static_target pc")

			full_pc_wo_ground = np.concatenate((static_gseg, dynamic_target_pc[:, :-1]), axis = 0)

			#Doing static (target labels + the rest of the environment) ground segmented
			static_gseg_sensorgrid = grid_generation_utils.create_grid(static_gseg, res, cells_x, cells_y, buf_veh, frame_index)

			#Doing dynamic targets
			dynamic_sensorgrid = grid_generation_utils.create_grid(dynamic_target_pc, res, cells_x, cells_y, buf_veh, frame_index)

			#Doing full grid ground segmented
			full_sensorgrid = grid_generation_utils.create_grid(full_pc_wo_ground, res, cells_x, cells_y, buf_veh, frame_index)

			full_sensorgrids.append(full_sensorgrid)
			static_gseg_sensorgrids.append(static_gseg_sensorgrid)
			dynamic_sensorgrids.append(dynamic_sensorgrid) 

		# Save sensorgrids
		print("Saving sensorgrids.hkl")
		hkl_outdir = os.path.join(training_folder, segment_file_name, "sensorgrids.hkl")
		hkl.dump([full_sensorgrids, static_gseg_sensorgrids, dynamic_sensorgrids], 
			hkl_outdir, mode = "w")
		print("One segment completed")
		end_time_segment = time.time()
		print("This segment takes:", end_time_segment - start_time_segment)