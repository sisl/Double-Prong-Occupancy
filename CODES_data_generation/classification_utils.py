import math
import tensorflow as tf
import numpy as np 
import utils

def get_distance(point1, point2):
	
	x1 = point1[0]
	x2 = point2[0]

	y1 = point1[1]
	y2 = point2[1]

	dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
	
	return dist

def get_distance_between_frames(frames, target_labels_all, last_frame_index, first_frame_index = 0):

	# Compare between two consecutive frames (transform coordinates of the next frame to be of the current frame)
	target_dist_dict = dict()

	for frame_index in range(first_frame_index, last_frame_index): # not last_frame_index + 1 since we can't compare
		for target_id, target_label_all in target_labels_all.items(): # target_label_all: list of all frames

			target_label = target_label_all[frame_index] # target_label for the specific frame
			target_label_next = target_label_all[frame_index + 1]

			if target_id not in target_dist_dict:
				target_dist_dict[target_id] = ["None"]*(last_frame_index)

			if target_label == "None" or target_label_next == "None":
				continue #target object doesn't exist in this frame -> continue to the next target object

			target_pos_current = np.array([target_label.box.center_x, target_label.box.center_y, 
				target_label.box.center_z]).reshape((3, 1))
			target_pos_next = np.array([target_label_next.box.center_x, target_label_next.box.center_y, 
				target_label_next.box.center_z, 1]).reshape((4, 1))
			
			ego_pose_current = np.reshape(np.array(frames[frame_index].pose.transform), [4, 4])
			ego_pose_next = np.reshape(np.array(frames[frame_index + 1].pose.transform), [4, 4])

			target_pos_next_global = np.matmul(ego_pose_next, target_pos_next)
			target_pos_next_current = np.matmul(np.linalg.inv(ego_pose_current), target_pos_next_global)
			target_pos_next_current = target_pos_next_current[:-1] # delete 1 at the end 

			dist = get_distance(target_pos_current, target_pos_next_current)
			target_dist_dict[target_id][frame_index] = dist

	return target_dist_dict

def classify_target_lidar(frames, classified_target_dict, last_frame_index, 
    first_frame_index, target_labels_all, threshold_vehicle, threshold_pedestrian, save_folder = None):
    
    static_points = []
    dynamic_points = []
    all_points = []

    for frame_index in range(first_frame_index, last_frame_index): # loop through each frame (no +1 since no distance calc)
        # get LiDAR points in a frame
        #print("frame_index", frame_index)
        points, intensity = utils.get_lidar(frames, frame_index) # list of lidar sensors
        points_mid_lidar = points[0]
        intensity = intensity.reshape((intensity.shape[0], 1))
        points_mid_lidar_w_intensity = np.hstack((points_mid_lidar, intensity))
        
        counter_static = 1
        counter_dynamic = 1

        static_points_frame = None
        dynamic_points_frame = None

        # target label patches 
        for target_id, target_label_all in target_labels_all.items():
            
            laser_label = target_label_all[frame_index]
            if laser_label == "None":
                continue

            dist = classified_target_dict[target_id][frame_index]
            
            if dist == "None":
                col = "black"   
            elif laser_label.type == 1 and dist <= threshold_vehicle: 
                col = "red" # static vehicle
            elif laser_label.type == 1 and dist > threshold_vehicle:
                col = "green" # moving vehicle
            elif dist <= threshold_pedestrian:
                col = "red" # static everything else (unknown, pedestrian, cyclist)
            else: 
                col = "green" # moving everything else (unknown, pedestrian, cyclist)
            
            # box = center_x, center_y, center_z, length, width, height, heading
            box_extend = 0.4 # m
            box = np.array([laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z, 
            laser_label.box.length + box_extend, laser_label.box.width + box_extend, laser_label.box.height, laser_label.box.heading])
            box = tf.convert_to_tensor(box)
            box = tf.reshape(box, [1,7])
            _, points_w_intensity_in_box = utils.get_points_in_box(tf.convert_to_tensor(points_mid_lidar, dtype = tf.float64), 
                points_mid_lidar_w_intensity, box)

            if col == "red":
                # static case
                if counter_static == 1:
                    static_points_frame = points_w_intensity_in_box
                    counter_static = -1
                else:
                    static_points_frame = np.concatenate((static_points_frame, points_w_intensity_in_box), axis = 0)
            elif col == "green":
                # dynamic case
                if counter_dynamic == 1:
                    dynamic_points_frame = points_w_intensity_in_box
                    counter_dynamic = -1
                else:
                    dynamic_points_frame = np.concatenate((dynamic_points_frame, points_w_intensity_in_box), axis = 0)
            
        if static_points_frame is not None:
            static_points.append(static_points_frame)
        else:
            static_points.append(np.array([-99]))
        if dynamic_points_frame is not None:
            dynamic_points.append(dynamic_points_frame)
        else:
            dynamic_points.append(np.array([-99]))
        all_points.append(points_mid_lidar_w_intensity)

    return static_points, dynamic_points, all_points