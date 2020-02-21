import os
import torch
import numpy as np

from networks import Conv_Model, Fc_Model
from train import train_model, eval_model
from mnist import get_mnist
import params


train_loader = get_mnist(train = True)
test_loader = get_mnist(train = False)


conv_model = Conv_Model()
fc_model = Fc_Model()

if(params.gpu_flag):
    conv_model.cuda(params.gpu_name)
    fc_model.cuda(params.gpu_name)

if( not os.path.isfile(params.path_conv_model) ):
    conv_model = train_model(conv_model,train_loader,params.path_conv_model, params.path_log_conv)

else:
    conv_model = torch.load(params.path_conv_model)
    if(params.gpu_flag):
        conv_model.cuda(params.gpu_name)

_,_,predicted = eval_model(conv_model,test_loader)
np.savetxt(params.path_conv_out,predicted)


if( not os.path.isfile(params.path_fc_model) ):
    fc_model = train_model(fc_model,train_loader,params.path_fc_model, params.path_log_fc)

else:
    fc_model = torch.load(params.path_fc_model)
    if(params.gpu_flag):
        fc_model.cuda(params.gpu_name)

_,_,predicted = eval_model(fc_model,test_loader)
np.savetxt(params.path_fc_out,predicted)