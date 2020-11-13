'''
Code from GroudSeg.py implementation found at the github repository:
	https://github.com/mitkina/EnvironmentPrediction
    
For the implementation of Random Markov Field ground segmentation described in:
    G. Postica, A. Romanoni, and M. Matteucci. Robust moving objects detection in LiDAR 
    data exploiting visual cues. In IEEE/RSJ International Conference on Intelligent Robots and 
    Systems (IROS), pages 1093-1098, 2016.
'''

import numpy as np
import math
import itertools
import copy

def ground_seg(point_cloud, height_lidar, res=1./3., s=0.09):

	num_points = point_cloud.shape[0]

	# generate 2-D grid of the LiDAR cloud
	max_index = math.sqrt(2.)*(128/3./2.+1.)

	# a 2D array that contains lists of 3D points in point_cloud that map to 
	# a particular grid cell (according to the place of the 3D point in point_cloud)
	filler = np.frompyfunc(lambda x: list(), 1, 1)
	grid = np.empty((int(2 * math.ceil(max_index/res) + 1), int(2 * math.ceil(max_index/res) + 1)), dtype=np.object)
	filler(grid, grid);

	# determine the center coordinate of the 2D grid
	center_x = int(math.ceil(max_index/res))
	center_y = int(math.ceil(max_index/res))

	for i in range(num_points):
		point = point_cloud[i,:]
		x = point[0]
		y = point[1]
		z = point[2]

		if ((math.fabs(x) <= max_index) and (math.fabs(y) <= max_index) and (z <= 3.5)):
		
			grid[int(center_x + round(x/res)), int(center_y + round(y/res))].append(i)

	h_G = np.nan*np.empty((grid.shape))
	
	# iterate radially outwards to compute if a point belongs to the ground (1) on mask grid  
	grid_seg = np.zeros(grid.shape)

	# initialize the center coordinate of the 2D grid to ground
	points_z = np.ndarray.tolist(point_cloud[grid[center_x, center_y],2])
	H = max(points_z or [np.nan])
	
	if not math.isnan(H):
		h_G[center_x, center_y] = H
	else:
		# initialize to the z-height of the LiDAR accroding to the KITTI set-up
		#h_G[center_x, center_y] = -2.184 - 0.05
		h_G[center_x, center_y] = -height_lidar

	# initialize the coordinates of inner circle
	circle_inner = [[center_x, center_y]]

	# identify all the points that were labeled as not ground
	point_cloud_seg = np.empty((0,3))		

	for i in range(1,int(math.ceil(max_index/res))+1):

		# generate indices at the ith inner circle level
		circle_curr = generate_circle(i, center_x, center_y)

		for indices in circle_curr:
			x = indices[0]
			y = indices[1]

			# compute h_hat_G: find max h_G of neighbors
			neigh_indeces = np.array(get_neighbors(x,y,circle_inner))
		
			# compute the min and max z coordinates of each grid cell		
			points_z = np.ndarray.tolist(point_cloud[grid[x,y],2])
			H = max(points_z or [np.nan])
			h = min(points_z or [np.nan])

			h_hat_G = np.nanmax(h_G[neigh_indeces])	

			if ((not np.isnan(H)) and (not np.isnan(h)) and \
				(H - h < s) and (H - h_hat_G < s)):
				grid_seg[x,y] = 1
				h_G[x,y] = copy.deepcopy(H)
							
			else:

				h_G[x,y] = copy.deepcopy(h_hat_G)

				# add to not ground points
				point_locations = grid[x,y]
									
				if point_locations != []:
					point_cloud_seg = np.vstack((point_cloud_seg, point_cloud[point_locations,:]))
				
		# update the inner circle indices
		circle_inner = copy.deepcopy(circle_curr)
	
	return point_cloud_seg

# return the indices of a circle at level i from the center of the grid
def generate_circle(i, center_x, center_y):

	circle_range = range(-1*i,i+1)
	circle = [list(x) for x in itertools.product(circle_range, circle_range)]
	circle = [[item[0]+center_x, item[1]+center_y] for item in circle if ((abs(item[0]) == i) or (abs(item[1]) == i))]		
	
	return circle

# get the inner circle neighbors of a point
def get_neighbors(x,y,circle_inner): 
	neigh_indices = []
	for indices in circle_inner:
		if ((abs(x-indices[0]) < 2) and (abs(y-indices[1]) < 2)):
			neigh_indices.append(indices)

	return neigh_indices