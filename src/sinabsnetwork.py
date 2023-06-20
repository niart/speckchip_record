import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork


######### define a weight matrix ##################################################################
class SinabsWeights:
    def __int__(self):
        pass
    def gabor_vertical(self, kernel, input_device_name):
        if input_device_name == "Davis346":
            scaling = 0.5
        elif input_device_name == "None":
            scaling = 0.3
        weights_gabor = torch.tensor([-0.1, -0.8, -0.3, 0.3, 1.0, 0.3, -0.3, -0.8, -0.1, 0], dtype=torch.float32)
        weights_gabor = weights_gabor.repeat(1, 1, kernel, 1)
        weights_zeros2 = torch.zeros((1, 1, kernel, kernel), dtype=torch.float32)
        weights_gabor_on = torch.cat([weights_gabor, weights_zeros2], dim=1)
        weights_gabor_off = torch.cat([weights_zeros2, weights_gabor], dim=1)
        weights_gabor_on = torch.mul(weights_gabor_on, scaling)
        weights_gabor_off = torch.mul(weights_gabor_off, scaling)
        return torch.cat([weights_gabor_on, weights_gabor_off], dim=0)

    def ones(self, kernel):
        ones = torch.ones((2, 2, kernel, kernel), dtype=torch.float32)
        return torch.mul(ones,0.1)

####### define a spiking neural network #######################################################################

class SinabsNeuralNetwork:
    def __int__(self):

        self.network = None

    def one_layer_network(self, input_device_name, weight_matrix, kernel):
        #
        cnn = nn.Sequential(nn.Conv2d(in_channels=2,  # max 1024
                                      out_channels=2,  # max 1024
                                      kernel_size=(kernel, kernel),  # max 16x16
                                      stride=(2, 2),  # 1,2,4,8
                                      padding=(0, 0),  # 0-7
                                      bias=False),
                            nn.ReLU(),
                            )

        weights = SinabsWeights()
        if weight_matrix == "gabor":
            cnn[0].weight.data = weights.gabor_vertical(kernel=kernel, input_device_name= input_device_name)
        elif weight_matrix == "ones":
            cnn[0].weight.data = weights.ones(kernel=kernel)

        # cnn to snn
        input_shape = (2, 128, 128)
        snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model
        # snn to DynapcnnNetwork
        return DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True)

    def two_layer_network(self, input_device_name, weight_matrix, kernel):
        #
        cnn = nn.Sequential(nn.Conv2d(in_channels=2,  # max 1024
                                      out_channels=2,  # max 1024
                                      kernel_size=(kernel, kernel),  # max 16x16
                                      stride=(2, 2),  # 1,2,4,8
                                      padding=(0, 0),  # 0-7
                                      bias=False),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=2,  # max 1024
                                      out_channels=2,  # max 1024
                                      kernel_size=(kernel, kernel),  # max 16x16
                                      stride=(1, 1),  # 1,2,4,8
                                      padding=(0, 0),  # 0-7
                                      bias=False),
                            nn.ReLU(),
                            )

        weights = SinabsWeights()
        if weight_matrix == "gabor":
            cnn[0].weight.data = weights.gabor_vertical(kernel=kernel, input_device_name= input_device_name)
            cnn[2].weight.data = weights.gabor_vertical(kernel=kernel, input_device_name=input_device_name)
        elif weight_matrix == "ones":
            cnn[0].weight.data = weights.ones(kernel=kernel)
            cnn[2].weight.data = weights.ones(kernel=kernel)


        # cnn to snn
        input_shape = (2, 128, 128)
        snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model
        # snn to DynapcnnNetwork
        return DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True)