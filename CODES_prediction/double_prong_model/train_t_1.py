'''
Train the double-prong model on t_1 mode
Input data: evidential grids produced from subsampling the waymo dataset for moving objects
'''

import importlib
import os
import tensorflow as tf 
import random as rn 
import matplotlib.pyplot as plt 
import hickle as hkl
import numpy as np
import pdb
import argparse
import math
import datetime
import pytz
from keras import backend as K 
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, add
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from alt_model_checkpoint.keras import AltModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from data_utils import SequenceGenerator

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

def weighted_loss(y_true, y_pred):
    weight_factor = 10
    y_pred_static = y_pred[:, 0]
    y_pred_dynamic = y_pred[:, 1]
    loss_static = mean_abs_error(y_true, y_pred_static)
    loss_dynamic = mean_abs_error(y_true, y_pred_dynamic)
    loss_final = loss_static + weight_factor*loss_dynamic
    return loss_final

def mean_abs_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def separate_input_masks(tensors):
    # Check input shape but should be (?, nt, 128, 128, 3)
    dynamic_mask = tf.expand_dims(tensors[:,:,:,:,-1], axis=-1)
    out_dynamic = tf.multiply(tensors[:,:,:,:,0:2], dynamic_mask)
    out_static = tf.multiply(tensors[:,:,:,:,0:2], 1-dynamic_mask)
    return [out_static, out_dynamic]

# Custom Metrics: return a single tensor value

def err_loss_static(y_true, y_pred):
    loss_static = mean_abs_error(y_true, y_pred[:, 0])
    return loss_static

def err_loss_dynamic(y_true, y_pred):
    loss_dynamic = mean_abs_error(y_true, y_pred[:, 1])
    return loss_dynamic

def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

def get_weight_norm(model):
    with K.name_scope('w_norm'):
        weights = model.trainable_weights
        w_norm = K.sqrt(sum([K.sum(K.square(w)) for w in weights]))
    return w_norm

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="output folder name")
ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
ap.add_argument("-ps", "--predfilestatic", required=True, help="prednet or prednet_dilation")
ap.add_argument("-pd", "--predfiledynamic", required=True, help="prednet or prednet_dilation")

args = vars(ap.parse_args())
prednet_module_static = importlib.import_module(args["predfilestatic"])
PredNet_static = prednet_module_static.PredNet

prednet_module_dynamic = importlib.import_module(args["predfiledynamic"])
PredNet_dynamic = prednet_module_dynamic.PredNet

G = args["gpus"]

# User Inputs
master_data_folder = "../../DATA_evidential_grid_splits/double_prong"
master_save_folder = os.path.join("../../DATA_predictions/double_prong/t_1_mode", args["output"])

if not os.path.exists(master_save_folder): 
    os.makedirs(master_save_folder)
    
save_weights_path = os.path.join(master_save_folder, "weights_t_1.hdf5")
save_json_path = os.path.join(master_save_folder, "model_t_1.json")
save_model = True

# Load train/val splits
train_path = os.path.join(master_data_folder, "X_train.hkl")
train_source_path = os.path.join(master_data_folder, "sources_train.hkl")
val_path = os.path.join(master_data_folder, "X_val.hkl")
val_source_path = os.path.join(master_data_folder, "sources_val.hkl")

# Training parameters
num_tsteps = 20 
num_epoch =  2 #60 
batch_size = 16
samples_per_epoch = 5 #2000
num_seq_val = 780 

K.set_learning_phase(1) # set the learning phase

# Model parameters
n_channels_prong, im_height, im_width = (2, 128, 128)
n_channels_input = 3

# K.image_data_format = "channels_last"
if K.image_data_format() == "channels_first":
        input_shape = (n_channels_input, im_height, im_width)
else:
        input_shape = (im_height, im_width, n_channels_input) # we fall under this case

inputs = Input(shape = (num_tsteps,) + input_shape) # shape of the input is (nt, 128, 128, 3)
input_sep_layer = Lambda(separate_input_masks, trainable=False)
inputs_static, inputs_dynamic = input_sep_layer(inputs)

# Static Prong: 3 layers
stack_sizes_static = (n_channels_prong, 48, 96)
R_stack_sizes_static = stack_sizes_static
A_filt_sizes_static = (3, 3)
Ahat_filt_sizes_static = (3, 3, 3)
R_filt_sizes_static = (3, 3, 3)

prednet_base_static = PredNet_static(stack_sizes_static, R_stack_sizes_static, A_filt_sizes_static, Ahat_filt_sizes_static, R_filt_sizes_static, output_mode = "error", return_sequences = True)
layer_config_base_static = prednet_base_static.get_config()
layer_config_base_static["name"] = "prednet_static"
prednet_static = PredNet_static(**layer_config_base_static)

# Dynamic Prong: 2 layers
stack_sizes = (n_channels_prong, 48)
R_stack_sizes = stack_sizes
A_filt_sizes = (3,)
Ahat_filt_sizes = (3, 3)
R_filt_sizes = (3, 3)

layer_loss_weights_static = np.array([1., 0., 0.])
layer_loss_weights_static = np.expand_dims(layer_loss_weights_static, 1)
time_loss_weights_static = 1./ (num_tsteps - 1) * np.ones((num_tsteps, 1))
time_loss_weights_static[0] = 0.


layer_loss_weights_dynamic = np.array([1., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights_dynamic = np.expand_dims(layer_loss_weights_dynamic, 1)
time_loss_weights = 1./ (num_tsteps - 1) * np.ones((num_tsteps, 1))  # equally weigh all timesteps except the first
time_loss_weights[0] = 0.

prednet_base_dynamic = PredNet_dynamic(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode = "error", return_sequences = True)
layer_config_base = prednet_base_dynamic.get_config()
layer_config_base["name"] = "prednet_dynamic"
prednet_dynamic = PredNet_dynamic(**layer_config_base)

errors_static = prednet_static(inputs_static) # errors will be (batch_size, nt, nb_layers)
errors_dynamic = prednet_dynamic(inputs_dynamic)

# Error_static
errors_by_time_static = TimeDistributed(Dense(1, trainable = False), weights = [layer_loss_weights_static, np.zeros(1)], trainable=False)(errors_static)
errors_by_time_static = Flatten()(errors_by_time_static)  # will be (batch_size, nt)
final_errors_static = Dense(1, weights = [time_loss_weights_static, np.zeros(1)], trainable = False)(errors_by_time_static)  # weight errors by time

# Error_dynamic
errors_by_time_dynamic = TimeDistributed(Dense(1, trainable = False), weights = [layer_loss_weights_dynamic, np.zeros(1)], trainable=False)(errors_dynamic)
errors_by_time_dynamic = Flatten()(errors_by_time_dynamic)  # will be (batch_size, nt)
final_errors_dynamic = Dense(1, weights = [time_loss_weights, np.zeros(1)], trainable = False)(errors_by_time_dynamic)

errors = Concatenate()([final_errors_static, final_errors_dynamic])

with tf.device('/cpu:0'):
    model = Model(inputs = inputs, outputs = errors)
model.compile(loss=weighted_loss, optimizer="adam", metrics=[err_loss_static, err_loss_dynamic])
model.summary()
print("%\n%\n%\n%\n%\n%\n%\n%")
print("\n======== Confirming PredNet class ========\n")
print("prednet_static:", PredNet_static)
print("\nprednet_dynamic:", PredNet_dynamic)
print("%\n%\n%\n%\n%\n%\n%\n%")

# Replicate the model on G GPUs
parallel_model = multi_gpu_model(model, gpus=G)
parallel_model.compile(loss=weighted_loss, optimizer = "adam", metrics=[err_loss_static, err_loss_dynamic])

train_generator = SequenceGenerator(train_path, train_source_path, num_tsteps, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_path, val_source_path, num_tsteps, batch_size=batch_size, N_seq=num_seq_val)

print("Shapes: ", train_generator.X.shape, val_generator.X.shape)
print("train generator", np.amax(train_generator.X), np.amin(train_generator.X))

lr_schedule = lambda epoch: 0.0001 if epoch < 75 else 0.0001 
lr_callback = LearningRateScheduler(lr_schedule)

if save_model:
    weight_callback = AltModelCheckpoint(save_weights_path, model, monitor='val_loss', save_best_only=True)

# Append the "l2 norm of gradients" tensor as a metric sequences for input
parallel_model.metrics_names.append("gradient_norm")
parallel_model.metrics_tensors.append(get_gradient_norm(parallel_model))
parallel_model.metrics_names.append("w_norm")
parallel_model.metrics_tensors.append(get_weight_norm(parallel_model))

# tensorboard
tz = pytz.timezone('America/Los_Angeles')
log_dir_folder = "../../DATA_predictions/double_prong/logs/t_1_mode/"
log_dir = log_dir_folder + args['output'] + "_" + datetime.datetime.now().astimezone(tz).strftime("%Y%m%d-%H:%M") 
tensorboard_callback = TensorBoard(log_dir = log_dir)

history = parallel_model.fit_generator(train_generator, int(math.ceil(samples_per_epoch / batch_size)), num_epoch, callbacks=[tensorboard_callback, weight_callback, lr_callback], validation_data=val_generator, validation_steps=int(math.ceil(num_seq_val / batch_size)), verbose=1, use_multiprocessing=True, workers=12)

# summarize history for loss
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig(os.path.join(master_save_folder, "loss_t_1.png"), dpi = 300)

fig2 = plt.figure()
plt.plot(history.history['gradient_norm'])
plt.plot(history.history['val_gradient_norm'])
plt.title('gradient norm')
plt.ylabel('grad_norm')
plt.xlabel('epoch')
plt.legend(['train_g','val_g'], loc='upper left')
plt.show()
plt.savefig(os.path.join(master_save_folder, "gradient_t_1.png"), dpi=300)

fig3 = plt.figure()
plt.plot(history.history['w_norm'])
plt.plot(history.history['val_w_norm'])
plt.title('weight norm')
plt.ylabel('w_norm')
plt.xlabel('epoch')
plt.legend(['train_w', 'val_w'], loc='upper left')
plt.show()
plt.savefig(os.path.join(master_save_folder, 'weight_t_1.png'), dpi=300)

# save history in a hickle file
hkl.dump(history.history, os.path.join(master_save_folder, "history_t_1.hkl"), mode='w')

if save_model:
    json_string = model.to_json()
    with open(save_json_path, "w") as f:
        f.write(json_string)

