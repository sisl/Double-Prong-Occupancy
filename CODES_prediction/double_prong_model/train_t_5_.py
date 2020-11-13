'''
Train the double-prong model on t_5 mode
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

def weighted_loss(y_true, y_hat):
    print("y_true.shape, y_hat.shape", y_true.shape, y_hat.shape)
    w_factor_full = 10
    w_factor_dynamic = 1
    loss_full = full_loss(y_true, y_hat)
    loss_dynamic = dynamic_loss(y_true, y_hat)
    loss_final = w_factor_full*loss_full + w_factor_dynamic*loss_dynamic
    print("loss_final.shape", loss_final.shape)
    return loss_final

def separate_input_masks(tensors):
    # Check input shape but should be (?, nt, 128, 128, 3)
    dynamic_mask = tf.expand_dims(tensors[:,:,:,:,-1], axis=-1)
    out_dynamic = tf.multiply(tensors[:,:,:,:,0:2], dynamic_mask)
    out_static = tf.multiply(tensors[:,:,:,:,0:2], 1-dynamic_mask)
    return [out_static, out_dynamic]

def full_grid_update(tensors):
    m_static = tensors[0] # should be (None, 20, 128, 128, 2)
    m_dynamic = tensors[1] # should be (None, 20, 128, 128, 2)
    # shpae of m_static_u should be (None, 20, 128, 128)
    m_static_u = 1 - m_static[:,:,:,:,0] - m_static[:,:,:,:,1]
    m_dynamic_u = 1 - m_dynamic[:,:,:,:,0] - m_dynamic[:,:,:,:,1]
    
    K = tf.multiply(m_static[:,:,:,:,0], m_dynamic[:,:,:,:,1]) - tf.multiply(m_static[:,:,:,:,1], m_dynamic[:,:,:,:,0])
    K = tf.minimum(K, 0.99)
    denominator = 1 - K
    
    m_o = tf.divide(tf.multiply(m_static[:,:,:,:,0], m_dynamic[:,:,:,:,0]) + tf.multiply(m_static_u, m_dynamic[:,:,:,:,0]) + tf.multiply(m_static[:,:,:,:,0], m_dynamic_u), denominator)
    m_f = tf.divide(tf.multiply(m_static[:,:,:,:,1], m_dynamic[:,:,:,:,1]) + tf.multiply(m_static_u, m_dynamic[:,:,:,:,1]) + tf.multiply(m_static[:,:,:,:,1], m_dynamic_u), denominator)
    
    m_o = tf.expand_dims(m_o, axis=-1)
    m_f = tf.expand_dims(m_f, axis=-1)
    m_full = tf.concat([m_o, m_f], axis=-1) # shape should be (None, 20, 128, 128, 2)
    print("shape m_full", m_full.shape)
    return m_full

# Custom Metrics

def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

def get_gradient_dynamic_norm(model):
    with K.name_scope('gradient_dyn_norm'):
        grads_dyn = K.gradients(model.total_loss, model.trainable_weights[22:])
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads_dyn]))
    return norm

def get_gradient_static_norm(model):
    with K.name_scope('gradient_static_norm'):
        grads_static = K.gradients(model.total_loss, model.trainable_weights[0:22])
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads_static]))
    return norm

def get_weight_norm(model):
    with K.name_scope('w_norm'):
        weights = model.trainable_weights
        w_norm = K.sqrt(sum([K.sum(K.square(w)) for w in weights]))
    return w_norm

def get_weight_dynamic_norm(model):
    with K.name_scope('w_dyn_norm'):
        weights = model.trainable_weights[22:]
        w_norm = K.sqrt(sum([K.sum(K.square(w)) for w in weights]))
    return w_norm

def get_weight_static_norm(model):
    with K.name_scope('w_static_norm'):
        weights = model.trainable_weights[0:22]
        w_norm = K.sqrt(sum([K.sum(K.square(w)) for w in weights]))
    return w_norm

def full_loss(y_true, y_hat):
    y_true_full = y_true[:, 1:, :, :, :2]
    y_hat_full = y_hat[:, 1:, :, :, :2]
    extrap_loss = 0.5 * K.mean(K.abs(y_true_full - y_hat_full), axis=-1) # output from extrap_loss function
    full_loss = K.mean(extrap_loss)
    return full_loss 

def dynamic_loss(y_true, y_hat):
    y_true_dynamic = tf.multiply(y_true[:,1:,:,:,:2], tf.expand_dims(y_true[:,1:,:,:,-1], axis=-1))
    y_hat_dynamic = y_hat[:, 1:, :, :, 2:4]
    extrap_loss = 0.5 * K.mean(K.abs(y_true_dynamic - y_hat_dynamic), axis=-1) # output from extrap_loss function
    sum_spatial = K.sum(extrap_loss, [-2, -1])
    dyn_loss = K.mean(sum_spatial)
    return dyn_loss

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="output folder name")
ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
ap.add_argument("-f", "--orig", required=True, help="original t_1 prong folder name")
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
master_save_folder = os.path.join("../../DATA_predictions/double_prong/t_5_mode", args["output"])
master_orig_folder = os.path.join("../../DATA_predictions/double_prong/t_1_mode", args["orig"])

if not os.path.exists(master_save_folder): 
    os.makedirs(master_save_folder)
    
save_weights_path = os.path.join(master_save_folder, "weights_t_5.hdf5")
save_json_path = os.path.join(master_save_folder, "model_t_5.json")
save_model = True

# Load train/val splits
train_path = os.path.join(master_data_folder, "X_train.hkl")
train_source_path = os.path.join(master_data_folder, "sources_train.hkl")
val_path = os.path.join(master_data_folder, "X_val.hkl")
val_source_path = os.path.join(master_data_folder, "sources_val.hkl")

orig_weights_file = os.path.join(master_orig_folder, "weights_t_1.hdf5")
orig_json_file = os.path.join(master_orig_folder, "model_t_1.json")

# Training parameters
num_tsteps = 20 
num_epoch =  2 #60
batch_size = 16
samples_per_epoch = 5 #2000 
num_seq_val = 780 # number of sequences for validation?

K.set_learning_phase(1) # set the learning phase

# For t_5 prediction
# Load the original model 
f = open(orig_json_file, "r")
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {"PredNet":PredNet_static, "tf":tf})
orig_model.load_weights(orig_weights_file)

orig_model.summary()

# get layer_config for prednet_static (layer index 2)
layer_config_full = orig_model.layers[2].get_config()
layer_config_full["output_mode"] = "prediction"
layer_config_full["extrap_start_time"] = 5
data_format = layer_config_full["data_format"] if "data_format" in layer_config_full else layer_config_full["dim_ordering"]

prednet_static = PredNet_static(weights=orig_model.layers[2].get_weights(), **layer_config_full)

# get layer_config for prednet_dynamic (layer index 3)
layer_config_dynamic = orig_model.layers[3].get_config()
layer_config_dynamic["output_mode"] = "prediction"
layer_config_dynamic["extrap_start_time"] = 5

prednet_dynamic = PredNet_dynamic(weights=orig_model.layers[3].get_weights(), **layer_config_dynamic)

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

predictions_static = prednet_static(inputs_static)
predictions_dynamic = prednet_dynamic(inputs_dynamic)

# Algorithmic updates to produce full grids
full_grid_update_layer = Lambda(full_grid_update, trainable=False)
predictions_full = full_grid_update_layer([predictions_static, predictions_dynamic])

final_predictions_stack = Concatenate()([predictions_full, predictions_dynamic])

with tf.device('/cpu:0'):
    model = Model(inputs = inputs, outputs = final_predictions_stack)
model.compile(loss=weighted_loss, optimizer="adam", metrics=[full_loss, dynamic_loss])
model.summary()

print("%\n%\n%\n%\n%\n%\n%\n%")
print("\n======== Confirming PredNet class ========\n")
print("prednet_static:", PredNet_static)
print("\nprednet_dynamic:", PredNet_dynamic)
print("%\n%\n%\n%\n%\n%\n%\n%")

# Replicate the model on G GPUs
parallel_model = multi_gpu_model(model, gpus=G)
parallel_model.compile(loss=weighted_loss, optimizer = "adam", metrics=[full_loss, dynamic_loss])

train_generator = SequenceGenerator(train_path, train_source_path, num_tsteps, batch_size=batch_size, shuffle=True, output_mode="prediction")
val_generator = SequenceGenerator(val_path, val_source_path, num_tsteps, batch_size=batch_size, N_seq=num_seq_val, output_mode="prediction")

print("Shapes: ", train_generator.X.shape, val_generator.X.shape)
print("train generator", np.amax(train_generator.X), np.amin(train_generator.X))

lr_schedule = lambda epoch: 0.00001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs

lr_callback = LearningRateScheduler(lr_schedule)
if save_model:
    weight_callback = AltModelCheckpoint(save_weights_path, model, monitor='val_loss', save_best_only=True)

# Append the "l2 norm of gradients" tensor as a metric sequences for input
parallel_model.metrics_names.append("gradient_norm")
parallel_model.metrics_tensors.append(get_gradient_norm(parallel_model))
parallel_model.metrics_names.append("gradient_dyn_norm")
parallel_model.metrics_tensors.append(get_gradient_dynamic_norm(parallel_model))
parallel_model.metrics_names.append("gradient_static_norm")
parallel_model.metrics_tensors.append(get_gradient_static_norm(parallel_model))

parallel_model.metrics_names.append("w_norm")
parallel_model.metrics_tensors.append(get_weight_norm(parallel_model))
parallel_model.metrics_names.append("w_dyn_norm")
parallel_model.metrics_tensors.append(get_weight_dynamic_norm(parallel_model))
parallel_model.metrics_names.append("w_static_norm")
parallel_model.metrics_tensors.append(get_weight_static_norm(parallel_model))

# tensorboard
tz = pytz.timezone('America/Los_Angeles')
log_dir_folder = "../../DATA_predictions/double_prong/logs/t_5_mode/"
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
plt.savefig(os.path.join(master_save_folder, "loss_t_5.png"), dpi = 300)

fig2 = plt.figure()
plt.plot(history.history['gradient_norm'])
plt.plot(history.history['val_gradient_norm'])
plt.title('gradient norm')
plt.ylabel('grad_norm')
plt.xlabel('epoch')
plt.legend(['train_g','val_g'], loc='upper left')
plt.show()
plt.savefig(os.path.join(master_save_folder, "gradient_t_5.png"), dpi=300)

fig3 = plt.figure()
plt.plot(history.history['w_norm'])
plt.plot(history.history['val_w_norm'])
plt.title('weight norm')
plt.ylabel('w_norm')
plt.xlabel('epoch')
plt.legend(['train_w', 'val_w'], loc='upper left')
plt.show()
plt.savefig(os.path.join(master_save_folder, 'weight_t_5.png'), dpi=300)

# save history in a hickle file
hkl.dump(history.history, os.path.join(master_save_folder, "history_t_5.hkl"), mode='w')

if save_model:
    json_string = model.to_json()
    with open(save_json_path, "w") as f:
        f.write(json_string)

