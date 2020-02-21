import os

gpu_flag = False
gpu_name = 'cuda:1'

folder_path = os.path.dirname(os.path.realpath(__file__))

data_root = os.path.join(folder_path,'data')
dataset_mean_value = 0.5
dataset_std_value = 0.5

dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

batch_size = 128
num_epochs = 20
log_step = 50
eval_step = 1

path_model = os.path.join(folder_path,'model')
path_conv_model = os.path.join(path_model,'conv_model.pt')
path_fc_model = os.path.join(path_model,'fc_model.pt')

path_log = os.path.join(folder_path,'log')
path_log_conv = os.path.join(path_log,'log_conv.txt')
path_log_fc = os.path.join(path_log,'log_fc.txt')

path_conv_out = os.path.join(folder_path,'convolution-neural-net.txt')
path_fc_out = os.path.join(folder_path,'multi-layer-net.txt')
