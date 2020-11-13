'''
Create new input data that contains moving objects in the training data 
from X_train_prefiltered.hkl and sources_train_prefiltered.hkl for both 
single prong and double prong cases 

'''
import os
import hickle as hkl
import numpy as np

# User inputs
time_step = 20
gap = 1 #1 means dif of 1 between each frame so no gap
min_len = 20 
master_data_folder_double = "../DATA_evidential_grid_splits/double_prong"
master_data_folder_single = "../DATA_evidential_grid_splits/single_prong"

split_data = "X_train_prefiltered.hkl"
split_sources = "sources_train_prefiltered.hkl"

data = hkl.load(os.path.join(master_data_folder_double, split_data)) # numpy array
sources = hkl.load(os.path.join(master_data_folder_double, split_sources)) # a list of scene name for each index

print("X_train", data.shape, type(data))
print("source", len(sources), type(sources))

sources_set = set(sources)
unique_sources_list = list(sources_set)
sources = np.array(sources)

skip_scenes_list = []
out_data_list = [] # convert to numpy array at the end before saving 
out_source_list = []

print("no. of unique scenes", len(unique_sources_list))
for scene_name in unique_sources_list:
    print("BEGIN")
    print(scene_name)
    mask = sources==scene_name
    scene_data = data[mask]
    assert len(scene_data) == np.sum(mask)
    dynamic_data = scene_data[:,:,:,-1] # (scene_length, 128, 128)
    # Get an array of frames with dynamic objects
    mask = np.sum(dynamic_data, axis=(1,2))
    frame_indices = sorted(np.argwhere(mask).reshape(-1)) # array containing frames with dynamic objects
    print("frame_indices", len(frame_indices), frame_indices)
    
    if len(frame_indices) < min_len:
        print("Skip scene case", scene_name)
        skip_scenes_list.append(scene_name)
        continue
    
    diff_arr = np.diff(frame_indices)
    mask_gap = diff_arr > gap
    ind_gap = np.argwhere(mask_gap).reshape(-1)

    # if all frame indices are consecutive from the start to the end -> easiest case
    if sum(diff_arr) == len(frame_indices) - 1 or len(ind_gap) == 0 :
        print("Consecutive case")
        out_scene_data = scene_data[frame_indices] # (len(frame_indices), 128, 128, 4)
        assert len(out_scene_data) == len(frame_indices)
        out_scene_source = [scene_name]*len(frame_indices) # list
         
        out_data_list.append(out_scene_data)
        out_source_list += out_scene_source
        continue

    # for Non-consecutive case
    print("Non-consecutive case")

    ind_gap = ind_gap + 1
    frame_pairs = []
    frame_pairs.append([frame_indices[0], frame_indices[ind_gap[0]-1]])
    for idx in range(len(ind_gap)-1):
        ind = ind_gap[idx]
        next_ind = ind_gap[idx+1]
        frame_pairs.append([frame_indices[ind], frame_indices[next_ind-1]])
    frame_pairs.append([frame_indices[ind_gap[-1]], frame_indices[-1]])
    
    print("frame_pairs", frame_pairs)
    # Remove for consecutive block of frames that are less than 20 in length
    len_frame_pairs = np.array([pair[1] -pair[0] +1 for pair in frame_pairs])
    mask = len_frame_pairs >= min_len
    frame_pairs_array = np.array(frame_pairs)[mask] # frame_pairs become array
     
    # Correct frame pairs
    print("frame_pairs", frame_pairs_array)
    
    # Add data and source to out_data_list and out_source_list for each cons block in frame_pairs
    for i in range(len(frame_pairs_array)):
        ind_pair = frame_pairs_array[i]
        ind_next = ind_pair[1]
        if ind_pair[1]-ind_pair[0]+1 < 20:
            ind_next = ind_pair[0] + time_step - 1
            print("New frame pair", ind_pair, ind_next)
        data_block_i = scene_data[ind_pair[0]:ind_next+1]
        #assert data_block_i.shape[0] == ind_pair[1]-ind_pair[0]+1
        out_data_list.append(data_block_i)

        source_name_block_i = scene_name + "/" + str(i)
        source_block_i = [source_name_block_i]*data_block_i.shape[0]
        #out_source_list.append(source_block_i)
        out_source_list += source_block_i

out_data_final = np.concatenate(out_data_list, axis=0)
out_source_final = out_source_list
assert len(out_source_final) == out_data_final.shape[0]
print("Finish generating new input data")
print("Out_data_final", out_data_final.shape)
print("Old_data", data.shape)

print("Saving data")
hkl.dump(out_data_final, os.path.join(master_data_folder_double, "X_train.hkl"), mode="w")
hkl.dump(out_source_final, os.path.join(master_data_folder_double, "sources_train.hkl"), mode="w")

hkl.dump(out_data_final[...,0:2], os.path.join(master_data_folder_single, "X_train.hkl"), mode="w")
hkl.dump(out_source_final, os.path.join(master_data_folder_single, "sources_train.hkl"), mode="w")
print("Successful saving")
