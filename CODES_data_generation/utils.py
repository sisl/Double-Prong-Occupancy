import tensorflow as tf 
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils 

def load_dataset(filename):

	dataset = tf.data.TFRecordDataset(filename, compression_type = '')
	frames = []

	for data in dataset:
		frame = open_dataset.Frame()
		frame.ParseFromString(bytearray(data.numpy()))
		frames.append(frame)

	return frames

def check_sign_in_segment(frames):
	for frame_index in range(len(frames)):
		for laser_label in frames[frame_index].laser_labels:
			if laser_label.type == 3:
				return True

	return False # if reach here, that means already exhausted all frames and didn't see any sign labels

def get_target_labels_dict_frames(frames, last_frame_index, first_frame_index = 0):

	'''
	Return: a dictionary of a segment 
		keys: target_id
		values: a list of label_info for all frames 
				element is "None" if the target is not present in that corresponding frame
	'''

	target_label_dict = dict()
	for frame_index in range(first_frame_index, last_frame_index + 1): # loop through frames
		for laser_label in frames[frame_index].laser_labels: # loop through labels in each frame
			if laser_label.type == 3: # type: sign
				continue
			label_id = laser_label.id
			# Check if this label_id already exists in dict key 
			if label_id not in target_label_dict:
				# Create one
				target_label_dict[label_id] = ["None"]*(last_frame_index + 1)
			target_label_dict[label_id][frame_index] = laser_label

	return target_label_dict

def get_points_in_box(points, points_w_intensity, box):
        is_points_in_box = box_utils.is_within_box_3d(points, box) # is a tf.tensor
        is_points_in_box = is_points_in_box.numpy()
        is_points_in_box = is_points_in_box.reshape(-1)
        points = points.numpy()

        points_in_box = points[is_points_in_box, :]
        points_w_intensity_in_box = points_w_intensity[is_points_in_box, :]

        return points_in_box, points_w_intensity_in_box # np array of all points in the box N x 3

# Based on waymo_open_dataset 
def get_lidar(frames, frame_index):
        (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frames[frame_index])

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frames[frame_index],
                range_images, camera_projections, range_image_top_pose)
        intensity_top_lidar_r0 = get_intensity(range_images[open_dataset.LaserName.TOP][0])

        return points, intensity_top_lidar_r0
