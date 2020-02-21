import torch
from torch import nn


class Conv_Model(nn.Module):
    def __init__(self):
        super( Conv_Model, self ).__init__()

        self.conv2d_1 = nn.Conv2d( in_channels = 3, out_channels = 16, kernel_size = 3,
                                  )

        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d( in_channels = 16, out_channels = 32, kernel_size = 3,
                                   stride = 2 )
        self.relu_2 = nn.ReLU()

        self.conv2d_3 = nn.Conv2d( in_channels = 32, out_channels = 64, kernel_size = 3,
                                  )
        self.relu_3 = nn.ReLU()

        self.conv2d_4 = nn.Conv2d( in_channels = 64, out_channels = 64, kernel_size = 3,
                                   stride = 2 )
        self.relu_4 = nn.ReLU()

        self.global_pool = nn.AvgPool2d(kernel_size = 4, stride = 4  )

        self.fc = nn.Linear( in_features = 64, out_features = 10)

    def forward(self,input_):

        ret = self.conv2d_1(input_)
        ret = self.relu_1(ret)

        ret = self.conv2d_2(ret)
        ret = self.relu_2(ret)
        
        ret = self.conv2d_3(ret)
        ret = self.relu_3(ret)

        ret = self.conv2d_4(ret)
        ret = self.relu_4(ret)

        ret = self.global_pool(ret)
        ret = ret.reshape(ret.shape[:2])
        ret = self.fc(ret)

        return ret


class Fc_Model(nn.Module):
    def __init__(self):
        super( Fc_Model, self ).__init__()

        self.linear_1 = nn.Linear( in_features = 784, out_features = 128)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear( in_features = 128, out_features = 128)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear( in_features = 128, out_features = 10)

    def forward(self,input_):
        input_ = input_[:,0,:,:]
        input_ = input_.reshape(input_.shape[0],784)
        ret = self.linear_1(input_)
        ret = self.relu_1(ret)

        ret = self.linear_2(ret)
        ret = self.relu_2(ret)
        

        ret = self.linear_3(ret)

        return ret


