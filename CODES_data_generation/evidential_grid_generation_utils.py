'''
Code modified from EvidentialGridsFullKITTI.py implementation found at the github repository:
	https://github.com/mitkina/EnvironmentPrediction
'''
import numpy as np
import math
import copy

def create_DST_grids(grids, meas_mass = 0.95):
	
	data = []

	for j in range(grids.shape[0]):

		grid = grids[j,:,:]
		free_array = np.zeros(grid.shape)
		occ_array = np.zeros(grid.shape)
		
		# occupied indices
		indices = np.where(grid == 1)
		occ_array[indices] = meas_mass

		# free indices
		indices = np.where(grid == 2)
		free_array[indices] = meas_mass

		# ego vehicle
		indices = np.where(grid == 3)
		occ_array[indices] = 1.

		data.append(np.stack((free_array, occ_array)))

	data = np.array(data)        
		
	return data

def get_mass(up_free, up_occ, meas_grid):

	probO = 0.5*up_occ + 0.5*(1.-up_free)
	newMass = np.stack((up_occ, up_free, probO, meas_grid))

	return newMass

def mass_update(meas_free, meas_occ, prev_free, prev_occ, alpha):
			
		check_values = False

		# predicted mass
		m_occ_pred = np.minimum(alpha * prev_occ, 1. - prev_free)
		m_free_pred = np.minimum(alpha * prev_free, 1. - prev_occ)

		if check_values and (m_occ_pred > 1 or m_occ_pred < 0):
			if m_occ_pred > 1.:
				print("This is m_occ_pred: ", m_occ_pred)
			assert(m_occ_pred <= 1.)
			assert (m_occ_pred >= 0.)
			assert (m_free_pred <= 1. and m_free_pred >= 0.)
			assert (m_occ_pred + m_free_pred <= 1.)

		# combine measurement and prediction to form posterior occupied and free masses
		m_occ_up, m_free_up = update_of(m_occ_pred, m_free_pred, meas_occ, meas_free)

		if check_values and (m_occ_up > 1.001 or m_occ_up < 0.):
			print("mass_occ: ", m_occ_up, "mass_free: ", m_free_up)
			assert(m_occ_up <= 1. and m_occ_up >= 0.)
			assert (m_free_up <= 1. and m_free_up >= 0.)
			assert(m_occ_up + m_free_up <= 1.)
   
		return m_free_up, m_occ_up


def update_of(m_occ_pred, m_free_pred, meas_m_occ, meas_m_free):
	
	# predicted unknown mass
	m_unknown_pred = 1. - m_occ_pred - m_free_pred
	
	# measurement masses: meas_m_free, meas_m_occ
	meas_cell_unknown = 1. - meas_m_free - meas_m_occ
	
	# implement DST rule of combination
	K = np.multiply(m_free_pred, meas_m_occ) + np.multiply(m_occ_pred, meas_m_free)
	
	m_occ_up = np.divide((np.multiply(m_occ_pred, meas_cell_unknown) + np.multiply(m_unknown_pred, meas_m_occ) + np.multiply(m_occ_pred, meas_m_occ)), (1. - K))
	m_free_up = np.divide((np.multiply(m_free_pred, meas_cell_unknown) + np.multiply(m_unknown_pred, meas_m_free) + np.multiply(m_free_pred, meas_m_free)), (1. - K))    
	
	return m_occ_up, m_free_up

def find_nearest(n,v,v0,vn,res):
	"Element in nd array closest to the scalar value `v`"
	idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
	return idx

def find_nearest_index(min_val, val, res):
	
	# values are wrt the local frame
	# works for both val of types int and np array
	ind = np.int16(np.round((val-min_val)/res, decimals = 0))
	
	return ind

def fix_indices(desired_len, val_lims, ind_prev, coord_min, res, num_cells):
	
	val_start, val_end = val_lims
	ind_min, ind_max = ind_prev

	val_min = coord_min + res*ind_min
	val_max = coord_min + res*ind_max

	if abs(val_min - val_start) <= abs(val_max - val_end):
		# fix the minimum index
		ind_min_new = ind_min
		ind_max_new = ind_min + desired_len - 1
		if ind_max_new > num_cells - 1:
			ind_max_new = num_cells - 1
			ind_min_new = ind_maxmin_val, val, res_new - desired_len + 1
	else:
		ind_max_new = ind_max 
		ind_min_new = ind_max_new - desired_len + 1
		if ind_min_new < 0:
			ind_min_new = 0
			ind_max_new = ind_min_new + desired_len - 1

	return [ind_min_new, ind_max_new]

def change_ego_pose(H):
	H[[0,1,2,2], [2,2,0,1]] = 0
	return H


def find_overlapping(xy_minmax_prev, xy_minmax_cur, ego_prev, ego_cur, res, cells_x, cells_y):

	# All coordinates are in the current frame 

	# Find x_min, x_max, y_min, y_max among all 8 corners
	min_val = np.min([np.min(xy_minmax_prev, axis = 1), np.min(xy_minmax_cur, axis = 1)], axis = 0)
	max_val = np.max([np.max(xy_minmax_prev, axis = 1), np.max(xy_minmax_cur, axis = 1)], axis = 0)

	x_min, y_min = min_val[:2]
	x_max, y_max = max_val[:2]

	# Create a big "global" grid_cur
	x_min_grid = x_min - res*5
	x_max_grid = x_max + res*5
	y_min_grid = y_min - res*5
	y_max_grid = y_max + res*5

	num_cells_expand_left = math.ceil((xy_minmax_cur[0, 0] - x_min_grid)/res)
	num_cells_expand_right = math.ceil((x_max_grid - xy_minmax_cur[0, 2])/res)

	num_cells_expand_up = math.ceil((xy_minmax_cur[1, 0] - y_min_grid)/res)
	num_cells_expand_down = math.ceil((y_max_grid - xy_minmax_cur[1, 2])/res)

	x_min_grid_new = xy_minmax_cur[0, 0] - res*num_cells_expand_left
	x_max_grid_new = xy_minmax_cur[0, 2] + res*num_cells_expand_right
	y_min_grid_new = xy_minmax_cur[1, 0] - res*num_cells_expand_up
	y_max_grid_new = xy_minmax_cur[1, 2] + res*num_cells_expand_down
	x_range_grid = np.arange(x_min_grid_new, x_max_grid_new, res)
	y_range_grid = np.arange(y_min_grid_new, y_max_grid_new, res)

	grid_cur = np.ones((cells_y, cells_x))

	grid_cur = np.concatenate((np.zeros((grid_cur.shape[0], num_cells_expand_left)), grid_cur), axis = 1)
	grid_cur = np.concatenate((grid_cur, np.zeros((grid_cur.shape[0], num_cells_expand_right))), axis = 1)
	grid_cur = np.concatenate((np.zeros((num_cells_expand_up, grid_cur.shape[1])), grid_cur), axis = 0)
	grid_cur = np.concatenate((grid_cur, np.zeros((num_cells_expand_down, grid_cur.shape[1]))), axis = 0)

	# Create a big "global" grid for previous local grid (coordinates are in the current frame already)

	grid_prev = np.zeros(grid_cur.shape)
	x_local_range = np.arange(xy_minmax_cur[0, 0], xy_minmax_cur[0, 2], res)
	y_local_range = np.arange(xy_minmax_cur[1, 0], xy_minmax_cur[1, 2], res)
	grid_local_x_prevframe, grid_local_y_prevframe = np.meshgrid(x_local_range, y_local_range) 

	ind_mapping_dict = dict()
	ind_mapping_dict_reverse = dict()
	for j in range(0, len(y_local_range)):
		for i in range(0, len(x_local_range)):

			y_local_prev = grid_local_y_prevframe[j, i]
			x_local_prev = grid_local_x_prevframe[j, i]

			# Convert to current frame
			xy_global = np.matmul(ego_prev, np.array([x_local_prev, y_local_prev, 0, 1]))
			xy_prev_cur = np.matmul(np.linalg.inv(ego_cur), xy_global)

			ind_x_new = find_nearest_index(x_min_grid_new, xy_prev_cur[0], res)
			ind_y_new = find_nearest_index(y_min_grid_new, xy_prev_cur[1], res)
			
			grid_prev[ind_y_new, ind_x_new] = 1
			ind_mapping_dict[(j, i)] = (ind_y_new, ind_x_new)
			ind_mapping_dict_reverse[(ind_y_new - num_cells_expand_up, ind_x_new - num_cells_expand_left)] = (j, i)

	grid_result = np.multiply(grid_cur, grid_prev)
	indices_big_grid = np.argwhere(grid_result >= 1)
	indices_local_grid = copy.deepcopy(indices_big_grid)
	indices_local_grid[:, 0] = indices_big_grid[:, 0] - num_cells_expand_up
	indices_local_grid[:, 1] = indices_big_grid[:, 1] - num_cells_expand_left

	return ind_mapping_dict_reverse, indices_local_grid