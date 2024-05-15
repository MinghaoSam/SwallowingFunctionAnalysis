"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import os
import torch
import time
import ml_collections  # ml_collections is a library of Python data structures

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32            # change this to train larger ACC-UNet model
cosineLR = True         # whether use cosineLR or not
n_channels = 1
n_labels = 1
epochs = 300
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 50
early_stopping_patience = 100

pretrain = False


task_name = 'SagittalTongue_Temporal'



learning_rate = 1e-3
batch_size = 8


# model_name = 'UCTransNet'
# model_name = 'UNet_base'
model_name = 'TTUNet'
# model_name = 'DSCNet'


test_session = "session"


train_dataset = '../datasets/'+ task_name+ '/Train_Folder/'
val_dataset = '../datasets/'+ task_name+ '/Val_Folder/'
test_dataset = '../datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'session'  #time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'




##########################################################################
# Trans configs
##########################################################################
def get_TranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [8, 4, 2, 1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config




# used in testing phase, copy the session name in training phase
