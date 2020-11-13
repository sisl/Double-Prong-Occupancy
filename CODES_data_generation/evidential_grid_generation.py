'''
Code modified from EvidentialGridsFullKITTI.py implementation found at the github repository:
	https://github.com/mitkina/EnvironmentPrediction
'''

import os
import hickle as hkl
import numpy as np 
import copy
import time
from copy import deepcopy
import evidential_grid_generation_utils



# User inputs
master_folder = "../DATA_preprocessed"

res = 1/3
alpha = 0.9 # information aging (how much old information is discounted)
cells_x = 128
cells_y = 128

print("-------------START--------------")

for num in range(0, 32):
	training_folder_name = "training_00" + "{:02d}".format(num)
	training_folder = os.path.join(master_folder, training_folder_name)
	save_folder = os.path.join(master_folder, training_folder_name)

	segment_files = sorted(os.listdir(training_folder), 
		key = lambda y: int(y.split("_")[0]))

	print("Number of files in {}".format(training_folder_name), len(segment_files))
	
	for segment_file_name in segment_files:
		print("Segment file", segment_file_name)

		start_time_segment = time.time()
		sensorgrid_filename = os.path.join(training_folder, segment_file_name, "sensorgrids.hkl")
		
		# Load ego pose and sensorgrids
		ego_pose_path = os.path.join(training_folder, segment_file_name, "ego_pose.npz")
		ego_pose_file = np.load(ego_pose_path, allow_pickle = True)
		ego_pose_all = ego_pose_file["ego_pose"]

		full_sensorgrids, static_sensorgrids, dynamic_sensorgrids = hkl.load(sensorgrid_filename)

		# Work on full sensorgrids first
		full_sensorgrids = np.array(full_sensorgrids)
		data = evidential_grid_generation_utils.create_DST_grids(full_sensorgrids)

		Masses = []

		for frame_index in range(0, data.shape[0]):
		
			prev_free = np.zeros(full_sensorgrids.shape[1:]) # 128x128
			prev_occ = np.zeros(full_sensorgrids.shape[1:]) # 128x128

			# initializes a measurement cell array
			meas_free = data[frame_index,0,:,:] 
			meas_occ = data[frame_index,1,:,:]

			x_max_cur = res/2 + (cells_x/2 - 1)*res # local coordinate of the current frame
			y_max_cur = res/2 + (cells_y/2 - 1)*res # local coordinate

			y_min_cur = -y_max_cur
			x_min_cur = -x_max_cur
			
			if frame_index > 0:

				# Get x_min/max and y_min/max values of previous frame in this frame
				xy_minmax_prev_prev = np.array([[x_min_cur, x_max_cur, x_max_cur, x_min_cur],
					[y_min_cur, y_min_cur, y_max_cur, y_max_cur],[0, 0, 0, 0],[1, 1, 1, 1]])

				xy_minmax_global = np.matmul(ego_pose_all[frame_index - 1], xy_minmax_prev_prev)
				xy_minmax_prev_cur = np.matmul(np.linalg.inv(ego_pose_all[frame_index]), xy_minmax_global)

				ind_mapping_dict_reverse, indices_local_grid = evidential_grid_generation_utils.find_overlapping(xy_minmax_prev_cur, 
					xy_minmax_prev_prev, ego_pose_all[frame_index - 1], ego_pose_all[frame_index], res, cells_x, cells_y)
			
				for (j_new, i_new) in indices_local_grid:
					
					j_old, i_old = ind_mapping_dict_reverse[(j_new, i_new)]
					
					prev_free[j_new, i_new] = deepcopy(up_free[j_old, i_old])
					prev_occ[j_new, i_new] = deepcopy(up_occ[j_old, i_old])

			# MassUpdate (stored in grid_cell_array)
			up_free, up_occ = evidential_grid_generation_utils.mass_update(meas_free, meas_occ, prev_free, prev_occ, alpha)

			newMass = evidential_grid_generation_utils.get_mass(up_free, up_occ, full_sensorgrids[frame_index,:,:])

			if (frame_index + 1 ) > 0:
				Masses.append(newMass)

		print("Saving massgrid.hkl")
		hkl_outdir = os.path.join(save_folder, segment_file_name, "massgrid.hkl")
		hkl.dump(Masses, hkl_outdir, mode = 'w')

		end_time_segment = time.time()
		print("Time taken for this segment (s):", end_time_segment - start_time_segment)
