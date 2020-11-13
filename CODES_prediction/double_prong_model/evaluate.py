'''
Evaluate trained PredNet on t_5 mode
Calculates mean-squared error and plots predictions
'''
import importlib
import numpy as np
import tensorflow as tf
import random as rn
import os
import hickle as hkl
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from six.moves import cPickle
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from prednet import PredNet
from data_utils import SequenceGenerator
import evaluate_metrics_util
import pdb
np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

def separate_input_masks(tensors):
    # Check input shape but should be (?, nt, 128, 128, 3)
    dynamic_mask = tf.expand_dims(tensors[:,:,:,:,-1], axis=-1)
    out_dynamic = tf.multiply(tensors[:,:,:,:,0:2], dynamic_mask)
    out_static = tf.multiply(tensors[:,:,:,:,0:2], 1-dynamic_mask)
    return [out_static, out_dynamic]

def full_grid_update(tensors):
    m_static = tensors[0] # should be (None, 20, 128, 128, 2)
    m_dynamic = tensors[1] # should be (None, 20, 128, 128, 2)
    print("m_static.shape", m_static.shape)
    print("m_dynamic.shape", m_dynamic.shape)
    # shpae of m_static_u should be (None, 20, 128, 128)
    m_static_u = 1 - m_static[:,:,:,:,0] - m_static[:,:,:,:,1]
    m_dynamic_u = 1 - m_dynamic[:,:,:,:,0] - m_dynamic[:,:,:,:,1]
    #new
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

# User inputs
n_plot = 5 # example prediction plots
batch_size = 16
nt = 20

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
ap.add_argument("-f", "--orig", required=True, help="original t_5 folder name")
ap.add_argument("-ps", "--predfilestatic", required=True, help="prednet or prednet_dilation")
ap.add_argument("-pd", "--predfiledynamic", required=True, help="prednet or prednet_dilation")

args = vars(ap.parse_args())

prednet_module_static = importlib.import_module(args["predfilestatic"])
PredNet_static = prednet_module_static.PredNet
prednet_module_dynamic = importlib.import_module(args["predfiledynamic"])
PredNet_dynamic = prednet_module_dynamic.PredNet

G = args["gpus"]

master_data_folder = "../../DATA_evidential_grid_splits/double_prong"
master_save_folder = os.path.join("../../DATA_predictions/double_prong/evaluate", args["orig"])
master_orig_folder = os.path.join("../../DATA_predictions/double_prong/t_5_mode", args["orig"])

orig_weights_file = os.path.join(master_orig_folder, "weights_t_5.hdf5")
orig_json_file = os.path.join(master_orig_folder, "model_t_5.json")

if not os.path.exists(master_save_folder):
    os.makedirs(master_save_folder)

test_file = os.path.join(master_data_folder, 'X_test.hkl')
test_sources = os.path.join(master_data_folder, 'sources_test.hkl')

# Load trained model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet_static, 'tf':tf})
train_model.load_weights(orig_weights_file)
train_model.summary()

i_layer_prednet = 3 # 2, 3
layer_config = train_model.layers[i_layer_prednet].get_config()
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()

if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# PREDICT
X_hat = train_model.predict(X_test, batch_size)
print("Finished predicing!")

X_test = X_test.astype(np.float32)
# ProbOccupancy for full grids
X_test_full_probO = np.expand_dims(0.5*(X_test[:,:,:,:,0]) + 0.5*(1.-X_test[:,:,:,:,1]), axis=-1)
X_hat_full_probO = np.expand_dims(0.5*X_hat[:,:,:,:,0] + 0.5*(1-X_hat[:,:,:,:,1]), axis=-1) # (:,20,128,128,1)

# ProbOccupancy for dynamic objects
X_test_masked_probO = np.multiply(X_test_full_probO, np.expand_dims(X_test[...,-1], axis=-1))
X_hat_masked_probO = np.multiply(X_hat_full_probO, np.expand_dims(X_test[...,-1], axis=-1))

hkl.dump([X_test_full_probO, X_test_masked_probO], os.path.join(master_save_folder, "X_test_full_masked_probO.hkl"), mode="w")
hkl.dump([X_hat_full_probO, X_hat_masked_probO], os.path.join(master_save_folder, "X_hat_full_masked_probO.hkl"), mode="w")

# Image similarity metric
avg_score, ms_score = evaluate_metrics_util.ImageSimilarityMetric(X_test_full_probO[:,:,:,:,0], X_hat_full_probO[:,:,:,:,0], start_time = 5)

# MSE metric
mse_score, mse_per_frame = evaluate_metrics_util.MSE(X_test_full_probO[:,:,:,:,0], X_hat_full_probO[:,:,:,:,0], start_time = 5)

print("metrics: IS, MSE:", avg_score, mse_score)
# Plot some prediction
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 4.*aspect_ratio))
gs = gridspec.GridSpec(4, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(master_save_folder, 'example_plots')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test_full_probO.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test_full_probO[i,t,:,:,0], vmin = 0, vmax = 1, cmap = "jet", origin = "lower")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('GT full', fontsize=6)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat_full_probO[i,t,:,:,0], vmin = 0, vmax = 1, cmap = "jet", origin = "lower")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Pred full', fontsize=6)

        plt.subplot(gs[t+2*nt])
        plt.imshow(X_test_masked_probO[i,t,:,:,0], vmin = 0, vmax = 1, cmap = "jet", origin = "lower")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('GT Dyn', fontsize=6)

        plt.subplot(gs[t+3*nt])
        plt.imshow(X_hat_masked_probO[i,t,:,:,0], vmin = 0, vmax = 1, cmap = "jet", origin = "lower")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Pred Dyn', fontsize=6)

    plt.savefig(os.path.join(plot_save_dir, 'plot_' + str(i) + '.png'), dpi = 300)
    plt.clf()

