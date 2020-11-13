# Process evidential_grid data for double stream network with static/dynamic
'''
Particle filter velocity estimates and DST occupancy grids are processed into static and DOGMa grids that will serve as input into PredNet. 
This file assumes the full training tracking dataset is used.

Code modified from: https://github.com/coxlab/prednet
'''
import os
import numpy as np
import hickle as hkl
from random import seed
from random import randint

seed(8)

# Inputs
master_folder = "../DATA_preprocessed"
master_save_folder_single = "../DATA_evidential_grid_splits/single_prong"
master_save_folder_double = "../DATA_evidential_grid_splits/double_prong"

massgrid_hkl_name = "massgrid.hkl" 
sensorgrids_hkl_name = "sensorgrids.hkl" 

if not os.path.exists(master_save_folder_single): os.makedirs(master_save_folder_single)
if not os.path.exists(master_save_folder_double): os.makedirs(master_save_folder_double)


# Actual folders used to create the evidential_grid_splits data for training the model
# Change these folders per user's preferences

test_training_segfolder_inputs = dict()
test_training_segfolder_inputs["training_0000"] = [2, 3, 6, 9, 16]
test_training_segfolder_inputs["training_0001"] = [2, 6]
test_training_segfolder_inputs["training_0002"] = [2, 18, 19]
test_training_segfolder_inputs["training_0003"] = [3, 16, 23, 24]
test_training_segfolder_inputs["training_0005"] = [15]
test_training_segfolder_inputs["training_0006"] = [1, 8, 13, 20]
test_training_segfolder_inputs["training_0007"] = [1]
# the rest

val_training_segfolder_inputs = dict()
val_training_segfolder_inputs["training_0000"] = [7, 11, 18, 23]
val_training_segfolder_inputs["training_0001"] = [0, 4, 5, 8, 20]
val_training_segfolder_inputs["training_0002"] = [3, 5, 9, 22]
val_training_segfolder_inputs["training_0003"] = [2, 5, 13]
val_training_segfolder_inputs["training_0005"] = [2, 10, 16, 23]

train_training_segfolder_inputs = dict()
train_training_segfolder_inputs["training_0003"] = [14, 15, 17, 22]
train_training_segfolder_inputs["training_0004"] = [1, 12, 17]
train_training_segfolder_inputs["training_0005"] = [11, 12, 18, 24]
train_training_segfolder_inputs["training_0006"] = [16, 17, 19, 23, 25]
train_training_segfolder_inputs["training_0007"] = [3, 4, 19]
train_training_segfolder_inputs["training_0008"] = [9]
# the rest

# get not_train list:
train_trainingfolder_segfolder = []
for k, v in train_training_segfolder_inputs.items():
	segment_files = sorted(os.listdir(os.path.join(master_folder, k)), key = lambda y: int(y.split("_")[0]))
	for seg_index in v:
		train_trainingfolder_segfolder.append(os.path.join(k, segment_files[seg_index]))

val_trainingfolder_segfolder = []
for k, v in val_training_segfolder_inputs.items():
	segment_files = sorted(os.listdir(os.path.join(master_folder, k)), key = lambda y: int(y.split("_")[0]))
	for seg_index in v:
		val_trainingfolder_segfolder.append(os.path.join(k, segment_files[seg_index]))

test_trainingfolder_segfolder = []
for k, v in test_training_segfolder_inputs.items():
	segment_files = sorted(os.listdir(os.path.join(master_folder, k)), key = lambda y: int(y.split("_")[0]))
	for seg_index in v:
		test_trainingfolder_segfolder.append(os.path.join(k, segment_files[seg_index]))

# Create grid datasets
# Processes grids and saves them in train, val, test splits.

splits = {s: [] for s in ['train', 'test', 'val']}
splits['val'] = val_trainingfolder_segfolder
splits['test'] = test_trainingfolder_segfolder
splits['train'] = train_trainingfolder_segfolder
all_segments = set(splits['val'] + splits['test'] + splits['train'])

train_count = len(train_trainingfolder_segfolder)
test_count = len(test_trainingfolder_segfolder)
val_count = len(val_trainingfolder_segfolder)


# User inputs -> to match our data
train_max = 594
test_max = 119
val_max = 79


while val_count < val_max:
	train_folder = randint(0, 31)
	train_folder_name = "training_00" + "{:02d}".format(train_folder)
	segment_files = sorted(os.listdir(os.path.join(master_folder, train_folder_name)), 
		key = lambda y: int(y.split("_")[0]))
	seg_pos = randint(0, len(segment_files) - 1)
	name = os.path.join(train_folder_name, segment_files[seg_pos])
	if name not in all_segments:
		splits["val"].append(name)
		all_segments.add(name)
		val_count += 1

while test_count < test_max:
	train_folder = randint(0, 31)
	train_folder_name = "training_00" + "{:02d}".format(train_folder)
	segment_files = sorted(os.listdir(os.path.join(master_folder, train_folder_name)), 
		key = lambda y: int(y.split("_")[0]))
	seg_pos = randint(0, len(segment_files) - 1)
	name = os.path.join(train_folder_name, segment_files[seg_pos])
	if name not in all_segments:
		splits["test"].append(name)
		all_segments.add(name)
		test_count += 1

for num in range(0, 2): # only training_0000 and training_0001
	training_folder_name = "training_00" + "{:02d}".format(num)
	training_folder = os.path.join(master_folder, training_folder_name)

	segment_files = sorted(os.listdir(training_folder), 
		key = lambda y: int(y.split("_")[0]))

	print("Number of files in {}".format(training_folder_name), len(segment_files))
	
	for segment_file_name in segment_files:
	
		massgrid_trainfolder_segfolder = os.path.join(training_folder_name, segment_file_name)

		if massgrid_trainfolder_segfolder not in all_segments:
			splits["train"].append(massgrid_trainfolder_segfolder)

for split in splits:

	print("Doing split", split)
	
	source_list = []  # corresponds to recording that image came from
	found = False

	for name in splits[split]: # for each segment 
		
		massgrid_filepath = os.path.join(master_folder, name, massgrid_hkl_name)
		massgrid_array = np.array(hkl.load(massgrid_filepath))

		sensorgrids_filepath = os.path.join(master_folder, name, sensorgrids_hkl_name)
		sensorgrids_array = np.array(hkl.load(sensorgrids_filepath)) 
		# The order is : full_sensorgrids, static_gseg_sensorgrids, dynamic_sensorgrids

		# Change shape from (frame_length, 4, 128, 128) to (frame_length, 128, 128, 4, 1)
		evidential_array_current = np.transpose(np.expand_dims(massgrid_array, axis = 4).astype('float16'), (0,2,3,1,4))
		# The order is : up_occ, up_free, probO, meas_grid

		# Change the label from ego vehicle from 3 to 1 (occupied), normalize: 0 - {F,O}, 0.5 - {O}, 1.0 - {F}
		evidential_array_current[np.where(evidential_array_current == 3.)] = 1.

		# Change shape from (3, frame_length, 128, 128) to (frame_length, 128, 128, 3, 1)
		sensorgrids_array_current = np.transpose(np.expand_dims(sensorgrids_array, axis = 4).astype('float16'), (1,2,3,0,4))
		# 0 : unknown, 1 : occupied, 2 : free, 3 : ego vehicle
		sensorgrids_array_current[np.where(sensorgrids_array_current == 3.)] = 0.
		sensorgrids_array_current[np.where(sensorgrids_array_current == 2.)] = 0.

		evidential_all_current = np.concatenate((evidential_array_current[:,:,:,0:2,0], np.expand_dims(sensorgrids_array_current[:,:,:,2,0],axis=-1)), axis = -1)

		if not found:
			evidential_all = evidential_all_current
			found = True

		else:
			evidential_all = np.concatenate((evidential_all, evidential_all_current), axis = 0)
			
		source_list += [name] * evidential_all_current.shape[0]

	if split == 'train':
		hkl.dump(evidential_all, os.path.join(master_save_folder_double, 'X_' + split + '_prefiltered' + '.hkl'))
		hkl.dump(source_list, os.path.join(master_save_folder_double, 'sources_' + split + '_prefiltered'+ '.hkl'))
		
	else:
		hkl.dump(evidential_all, os.path.join(master_save_folder_double, 'X_' + split + '.hkl'))
		hkl.dump(source_list, os.path.join(master_save_folder_double, 'sources_' + split + '.hkl'))
		hkl.dump(evidential_all[:,:,:,0:2], os.path.join(master_save_folder_single, 'X_' + split + '.hkl'))
		hkl.dump(source_list, os.path.join(master_save_folder_single, 'sources_' + split + '.hkl'))
