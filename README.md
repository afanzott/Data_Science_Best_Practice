# RHI-DS-Challenge
Repo for the Data Science Challenge @RHI

# Initialization
Clone the repo and create the conda environment from the "RHI_Challenge_env.yaml" file via "conda env create -f RHI_Challenge_env.yaml" and activate it.

# Model training
To train a model just run train.py. Currently it will use the specified files under the data folder (which won't work on your local device). To train with different data just swap the variables "path_to_feature_file" and "path_to_target_file" in the "config.yaml" to your data location.

The train.py file utilizes a preprocessing pipeline created in pipeline.py to prepare the data for training. Successfully trained models will be stored as .pkl pipelines under trained_models. Therefore you have to change the variable "pipeline_save_file_path" in the "config_yaml" to your desired destination.

# Prediction
For making predictions you have to execute the predict.py file. But first change the variable "predict_pipeline" in the "config.yaml" to the file with your trained pipeline. Additionally you have to serve a data file as an argument when executing. Just add "-d path" where path corresponds to the location of the file, which should be predicted.
