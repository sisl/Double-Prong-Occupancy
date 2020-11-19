# Double-Prong-Occupancy
Implementation of double-prong ConvLSTM for occupancy grid prediction as in "Double-Prong ConvLSTM for Spatiotemporal Occupancy Prediction in Dynamic Environments" ([arXiv](https://arxiv.org/abs/2011.09045)) by Maneekwan Toyungyernsub, Masha Itkina, Ransalu Senanayake, Mykel J. Kochenderfer.

## Dataset
The LiDAR data used in the experiments are obtained from [Waymo](https://waymo.com/open/). We use their 31 training folders and split the data into training, validation, and test sets. The downloaded training folders should be put under the directory: DATA_waymo/. We also make use of the [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) github repository to preprocess the downloaded data. Please put the downloaded waymo_open_dataset folder under the CODES_data_generation/ directory. 

## Setup
The required dependencies are listed in requirements.txt.

## Data preprocessing 



