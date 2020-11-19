# Double-Prong-Occupancy
Implementation of double-prong ConvLSTM for occupancy grid prediction as in "Double-Prong ConvLSTM for Spatiotemporal Occupancy Prediction in Dynamic Environments" ([arXiv](https://arxiv.org/abs/2011.09045)) by Maneekwan Toyungyernsub, Masha Itkina, Ransalu Senanayake, and Mykel J. Kochenderfer.

## Dataset
The LiDAR data used in the experiments are obtained from [Waymo](https://waymo.com/open/). We use their 31 training folders and split the data into training, validation, and test sets. The downloaded training folders should be put under the directory: DATA_waymo/. We also make use of the [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) github repository to preprocess the downloaded data. Please put the downloaded waymo_open_dataset folder under the CODES_data_generation/ directory. 

## Setup
- python 3.6.9
- tensorflow-gpu (1.13.1)
- tensorboard (1.13.1)
- Keras (2.2.4)
- alt-model-checkpoint (2.0.2)
- numpy (1.18.1)
- hickle (3.4.5)

## Data preprocessing 
To generate the evidential grids, please run the following script from the CODES_data_generation/ directory: 
```
generate_data.sh
```

## Training
Run the following script from CODES_prediction/double_prong_model/ directory to train the model:
```
train.sh
```
The input arguments can be modified in the train.sh script to change the number of gpus used for training and the output file name. 

