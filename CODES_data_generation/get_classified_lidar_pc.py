import os
import numpy as np
import tensorflow as tf 
import copy
import time
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2
import matplotlib.pyplot as plt 
import pdb
import utils
import classification_utils

print(tf.executing_eagerly()) # Should print True for tensorflow 2.0

master_folder = "../DATA_waymo"
master_save_folder = "../DATA_preprocessed"
first_frame_index = 0
#last_frame_index = 49
last_frame_index = False
speed_threshold_vehicle = 5*1000/3600 # m/s if more than this, classified as dynamic object
speed_threshold_pedestrian = 3*1000/3600 # m/s
update_rate = 0.1 # s (1/Hz)
dist_threshold_vehicle = speed_threshold_vehicle * update_rate
dist_threshold_pedestrian = speed_threshold_pedestrian * update_rate
sign_check = True

print("--------START--------")
for num in range(0, 32):
	training_folder_name = "training_00" + "{:02d}".format(num)
	training_folder_path = os.path.join(master_folder, training_folder_name)
	tfrecord_files_list = sorted(os.listdir(training_folder_path))

	print("Number of files in {}".format(training_folder_name), len(tfrecord_files_list))
	for segment_index in range(len(tfrecord_files_list)):
		start_time_segment = time.time()
		print("Segment_index", segment_index)
		tfrecord_file_name = tfrecord_files_list[segment_index]
		if tfrecord_file_name.endswith(".tfrecord") != True:
			continue

		tfrecord_file_path = os.path.join(training_folder_path, tfrecord_file_name)

		frames = utils.load_dataset(tfrecord_file_path) # list
		print("Length of this segment", len(frames))

		# Check if this segment contains at least one sign and its location
		if sign_check:
			if utils.check_sign_in_segment(frames) == False: # means there is no sign in this segment
				print("No sign in this segment")
				continue # continue to check other segments

		tfrecord_file_name = "{}_".format(segment_index) + tfrecord_file_name # adding segment_index at the front of the old name
		master_save_folder_path = os.path.join(master_save_folder, training_folder_name, tfrecord_file_name[:-9]) #w/o .tfrecord

		if not os.path.isdir(master_save_folder_path):
			os.makedirs(master_save_folder_path)

		if last_frame_index == False:
			last_frame_index = len(frames) - 1
	
		target_labels_all = utils.get_target_labels_dict_frames(frames, last_frame_index, first_frame_index = first_frame_index)
		
		target_dict_classified = classification_utils.get_distance_between_frames(frames, 
				target_labels_all, last_frame_index, first_frame_index = first_frame_index)

		static_points, dynamic_points, all_points = classification_utils.classify_target_lidar(frames, target_dict_classified, last_frame_index, first_frame_index, 
			target_labels_all, dist_threshold_vehicle, dist_threshold_pedestrian, save_folder = os.path.join(master_save_folder_path, "classification", "targets_only"))
		
		print("Saving points.npz files")
		np.savez(os.path.join(master_save_folder_path, "points.npz"), static = static_points, dynamic = dynamic_points, all = all_points)
		print("Done saving npz files")

		# Getting extrinsic for Top Mid-Lidar only
		calibrations = sorted(frames[0].context.laser_calibrations, key=lambda c:c.name)

		for c in calibrations:
			if c.name == dataset_pb2.LaserName.TOP: 
				extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
				
				print("Saving extrinsic.npz files")
				np.savez(os.path.join(master_save_folder_path, "extrinsic.npz"), extrinsic = extrinsic)
				print("Done saving extrinsic.npz files")

		ego_pose = [] # for all frames
		for frame_index in range(first_frame_index, last_frame_index):
			ego_pose_current = np.reshape(np.array(frames[frame_index].pose.transform), [4, 4])
			ego_pose.append(ego_pose_current)
			
		print("Saving npz files")
		np.savez(os.path.join(master_save_folder_path, "ego_pose.npz"), ego_pose = ego_pose)
		print("Done saving ego_pose.npz files")
			
		# Fix last_frame_index back to default = False
		last_frame_index = False # put at the last line before loop back to segment loop 

		end_time_segment = time.time()
		print("Time for this segment (s):", end_time_segment - start_time_segment)