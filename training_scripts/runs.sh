#!/bin/bash

# # create resources
# python azure_create_resources.py --create-workspace --subscription-id [your-subscription_id]
#
# python azure_create_resources.py --create-compute --gups 1
# python azure_create_resources.py --create-compute --gups 2
# python azure_create_resources.py --create-compute --gups 4
#
# python azure_create_resources.py --create-env
#
# python azure_create_resources.py --upload-data
#
# python azure_create_resources.py --create-dataset --dataset-name birdsongs_10sec\
#   --data-path /data/audio_10sec/
#
# python azure_create_resources.py --create-dataset --dataset-name birdsongs_npy\
#   --data-path /data/npy/


# # Runs Kaggle Dataset
# python azure_train.py --model-name cnn1_audin_nmel_1
# python azure_train.py --model-name cnn1_audin_nmel_2
# python azure_train.py --model-name cnn1_audin_nmel_dcbl_1
# python azure_train.py --model-name cnn1_audin_nmel_dcbl_2
# python azure_train.py --model-name cnn1_audin_nffthl_1
# python azure_train.py --model-name cnn1_audin_nffthl_2
# python azure_train.py --model-name cnn1_audin_freq_1
# python azure_train.py --model-name cnn1_audin_freq_2
# python azure_train.py --model-name cnn1_audin_l2reg_1
# python azure_train.py --model-name cnn1_audin_l2reg_2
# python azure_train.py --model-name cnn1_audin_l2reg_3
# python azure_train.py --model-name cnn1_audin_l2reg_4
# python azure_train.py --model-name cnn1_audin_drp_1
# python azure_train.py --model-name cnn1_audin_drp_2
# python azure_train.py --model-name cnn1_audin_drp_3
# python azure_train.py --model-name cnn1_audin_drp_4
# python azure_train.py --model-name milsed_2block_dense
# python azure_train.py --model-name milsed_2block_dense
# python azure_train.py --model-name milsed_3block_dense
# python azure_train.py --model-name milsed_4block_dense
# python azure_train.py --model-name milsed_5block_dense --gpus 2
# python azure_train.py --model-name milsed_6block_dense --gpus 2
# python azure_train.py --model-name milsed_7block_dense --gpus 2
# python azure_train.py --model-name milsed_8block_dense --gpus 2
# python azure_train.py --model-name milsed_7block_dense_drp_1 --gpus 2
# python azure_train.py --model-name milsed_7block_dense_drp_2 --gpus 2
# python azure_train.py --model-name milsed_7block_dense_drp_3 --gpus 2
# python azure_train.py --model-name milsed_7block_dense_drp_4 --gpus 2
# python azure_train.py --model-name milsed_7block_dense_drp_4 --gpus 2
#
# # Use full dataset
# python azure_train.py --model-name milsed_7block_dense --gpus 2 --data-subset all_10sec_wav

# # With Data Augmentation
# python azure_train.py --model-name milsed_7block_dense \
#   --data-subset='kaggle_full_length_npy' --gpus 2 --multithread --augment-position
# python azure_train.py --model-name milsed_7block_dense \
#   --data-subset='kaggle_full_length_npy' --gpus 1 --multithread --augment-position\
#   --augment-pitch
# python azure_train.py --model-name milsed_7block_dense \
#   --data-subset='kaggle_full_length_npy' --gpus 1 --multithread --augment-position\
#   --augment-stretch
# python azure_train.py --model-name milsed_7block_dense \
#   --data-subset='kaggle_full_length_npy' --gpus 1 --multithread --augment-position\
#   --augment-pitch --augment-stretch

# # Data Augmentation on All Data (Careful this has a long runtime)
# python azure_train.py --model-name milsed_7block_dense \
#   --data-subset='all_full_length_npy' --gpus 1 --multithread --augment-position
