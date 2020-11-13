# grid_generation_utils
import numpy as np

def height_filter(pc_input, height_max):

	# To filter out possible noise in the raw LiDAR pc input 
	mask = np.ones(pc_input.shape[0], dtype = bool)
	mask = np.logical_and(mask, pc_input[:, 2] < height_max)
	pc_filtered = pc_input[mask, :]

	return pc_filtered

def get_static_env(dynamic_pc, all_pc):

	# Create structured array
	nrows, ncols = all_pc.shape
	
	# no dynamic pc in this frame
	if dynamic_pc.shape == (1,) and dynamic_pc == -99: # Assigned -99 for non-existent pc previously
		print("exception case - grid_util")
		return all_pc

	dtype = {"names":["f{}".format(i) for i in range(ncols)], 
		"formats":ncols * [all_pc.dtype]}
	static_env_pc = np.setdiff1d(all_pc.view(dtype), dynamic_pc.view(dtype))
	static_env_pc = static_env_pc.view(all_pc.dtype).reshape(-1, ncols)

	return static_env_pc

def find_nearest_index(min_val, val, res):
	
	# works for both val of types int and np array
	ind = np.int16(np.round((val-min_val)/res, decimals = 0))
	
	return ind

# generate the y_range along a line
def linefunction(x_1, y_1, x_2, y_2,x_range):
	m = (y_2 - y_1)/(x_2 - x_1)
	b = y_1 - m*x_1 # should be zero if one point is set at the origin (0, 0)

	return m*x_range + b 

def create_grid(pointcloud, res, cells_x, cells_y, buf_veh, frame_index):
	
	'''
	0 : unknown
	1 : occupied
	2 : free
	3 : ego vehicle
	'''
	
	# Create an empty grid
	grid = np.zeros((cells_y, cells_x))

	if int(cells_x % 2) == 0: # if cells_x is an even number eg. 128
		x_max = res/2 + (cells_x/2 - 1)*res
		# if cells_x is even, the grid should be symmetric
		x_min = -x_max
	if cells_y == cells_x:
		y_max = x_max
		y_min = x_min
	else: # even cells_y
		y_max = res/2 + (cells_y/2 - 1)*res
		# if cells_x is even, the grid should be symmetric
		y_min = -y_max
	
	grid_x_range = np.arange(x_min, x_max, res)
	grid_y_range = np.arange(y_min, y_max, res)
	
	# Fill in target point clouds on the grid -> find which cells the pc correspond to
	xind_pc = find_nearest_index(x_min, pointcloud[:, 0], res)
	yind_pc = find_nearest_index(y_min, pointcloud[:, 1], res)
	
	# Keep indices that correspond to pointcloud coordinates higher than the min val of the grid
	mask = np.logical_and(xind_pc > 0, yind_pc > 0)
	xind_pc = xind_pc[mask]
	yind_pc = yind_pc[mask]
	pointcloud_wn_grid = pointcloud[mask, :]

	# Keep indices that correspond to pointcloud coordinates lower than the max val of the grid
	mask_max = np.logical_and(xind_pc < grid.shape[1], yind_pc < grid.shape[0])
	xind_pc = xind_pc[mask_max]
	yind_pc = yind_pc[mask_max]
	pointcloud_wn_grid = pointcloud_wn_grid[mask_max, :]

	# Fill in grid cells corresponding to pointcloud locations -> occupied
	grid[yind_pc, xind_pc] = 1
	yind_occ, xind_occ = np.where(grid == 1)

	# Fill in the free space 
	# Compute distances from pc obstacles to the center of the ego vehicle
	dist = np.sqrt(np.square(pointcloud_wn_grid[:, 0] - 0.0) + np.square(pointcloud_wn_grid[:, 1] - 0.0))
	min_max_order = np.argsort(dist)
	xind_ordered = xind_pc[min_max_order]
	yind_ordered = yind_pc[min_max_order]

	for row in range(grid.shape[0]): # y
		for col in range(grid.shape[1]): # x
			if ((col == grid.shape[1] - 1) or (row == grid.shape[0] - 1) or (col == 0) or (row == 0)):

				x_coord = grid_x_range[col]
				y_coord = grid_y_range[row]
			
				# generate discrete x points for the line from the center to the edge of the grid
				if x_coord < 0:
					x_range = np.arange(0, x_coord,-res*0.01)
				else:
					x_range = np.arange(0, x_coord, res*0.01)
		
				if (abs(x_coord - 0) < res):
					x_range = x_coord*np.ones(int((grid.shape[1]/2)))
					if y_coord < 0:
						y_range = np.arange(y_min + row*res, 0 ,res)
						y_range = y_range[::-1]
						
					else:
						y_range = np.arange(y_min + row*res, 0 ,-res)
						y_range = y_range[::-1]
				else:
					y_range = linefunction(0, 0, x_coord, y_coord, x_range)

				xtemp = find_nearest_index(x_min, x_range, res)
				ytemp = find_nearest_index(y_min, y_range, res)

				if (np.all(grid[ytemp, xtemp] != 1)):
					grid[ytemp, xtemp] = 2

				else:                   
					# loop through each segment, and find the closest coordinates
					for j in range(x_range.shape[0]): # move from the inside -> out
						if grid[ytemp[j],xtemp[j]] == 1:
							break
						grid[ytemp[j], xtemp[j]] = 2

	# Fill in grid cells corresponding to ego vehicles based on ego vehicle frame at mid of rear wheel axle
	x_front = 4.371 # m
	x_back = -0.8 # m
	y_side = 1 # m

	ego_pos = np.array([[x_front, y_side], [x_front, -y_side], [x_back, y_side], [x_back, -y_side]])

	ego_x_ind = find_nearest_index(x_min, np.array([x_back, x_front]), res)
	ego_y_ind = find_nearest_index(y_min, np.array([-y_side, y_side]), res)

	grid[ego_y_ind[0]:ego_y_ind[1]+1, ego_x_ind[0]:ego_x_ind[1]+1] = 3

	return grid
	

