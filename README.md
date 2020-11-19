# Double-Prong-Occupancy
Implementation of double-prong ConvLSTM for occupancy grid prediction as in "Double-Prong ConvLSTM for Spatiotemporal Occupancy Prediction in Dynamic Environments" ([arXiv](https://arxiv.org/abs/2011.09045)) by Maneekwan Toyungyernsub, Masha Itkina, Ransalu Senanayake, Mykel J. Kochenderfer.

## Dataset
The LiDAR data used in the experiments are obtained from the ([Waymo Open Dataset](https://waymo.com/open/)). We use their 31 training folders to split into training, validation, and test sets. The downloaded training folders should be put under the directory: DATA_waymo. 
