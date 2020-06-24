from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from util import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def construct_cfg(configFile):
    # Read and pre-process the configuration file by open the file
    # and store the file as a list as well as removing redundant
    # comments
    config = open(configFile,'r')
    file = config.read().split('\n')
    # Get rid of empty lines
    file = [line for line in file if len(line) > 0]
    # Get rid of comments
    file = [line for line in file if line[0] != '#']
    # get rid of fringe whitespaces
    file = [line.lstrip().rstrip() for line in file]
    #Separate network blocks in a list
    networkBlocks = []
    networkBlock = {}

    for x in file:
        # start of a new block
        if x[0] == '[':
            # If block is not empty, storing values of previous block.
            if len(networkBlock) != 0:
                # append the blocks list
                networkBlocks.append(networkBlock)
                # Re-initialize the block
                networkBlock = {}
            networkBlock["type"] = x[1:-1].rstrip()
        else:
            entity , value = x.split('=')
            networkBlock[entity.rstrip()] = value.lstrip()
    networkBlocks.append(networkBlock)

    return networkBlocks




def buildNetwork(networkBlocks):
    # Getting the information from the input and pre-processing
    DNInfo = networkBlocks[0]
    modules = nn.ModuleList([])
    channels = 3
    filterTracker = []

    for i,x in enumerate(networkBlocks[1:]):
        seqModule  = nn.Sequential()
        #check the type of block
        #create a new module for the block
        #append to module_list

        # If it is convolution layer
        if (x["type"] == "convolutional"):

            filters= int(x["filters"])
            pad = int(x["pad"])
            kernelSize = int(x["size"])
            stride = int(x["stride"])

            if pad:
                padding = (kernelSize - 1) // 2
            else:
                padding = 0

            activation = x["activation"]
            try:
                bn = int(x["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True

            conv = nn.Conv2d(channels, filters, kernelSize, stride, padding, bias = bias)
            seqModule.add_module("conv_{0}".format(i), conv)
            #Add the Batch Normalization Layer
            if bn:
                bn = nn.BatchNorm2d(filters)
                seqModule.add_module("batch_norm_{0}".format(i), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                seqModule.add_module("leaky_{0}".format(i), activn)


        elif (x["type"] == "upsample"):
            # Sample of upsampling is at the end of the script
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            seqModule.add_module("upsample_{}".format(i), upsample)
        #If it is a route layer
        elif (x["type"] == "route"):
            x['layers'] = x["layers"].split(',')
            # start of the rout
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end =0

            if start > 0:
                start = start - i
            if end > 0:
                end = end - i

            route = EmptyLayer()
            seqModule.add_module("route_{0}".format(i),route)
            if end < 0:
                filters = filterTracker[i+start] + filterTracker[i+end]
            else:
                filters = filterTracker[i+start]
        #shortcut is skip connection
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            seqModule.add_module("shortcut_{0}".format(i),shortcut)
        # Detection layer
        elif (x["type"] == "yolo"):
            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]
            masks = x["mask"].split(',')
            masks = [int(a) for a in masks]
            anchors = [(anchors[j],anchors[j+1]) for j in range(0,len(anchors),2)]
            anchors = [anchors[j] for j in masks]
            detectorLayer = DetectionLayer(anchors)

            seqModule.add_module("Detection_{0}".format(i),detectorLayer)

        modules.append(seqModule)
        channels = filters
        filterTracker.append(filters)
    return (DNInfo, modules)



class net(nn.Module):
    def __init__(self, cfgfile):
        super(net, self).__init__()
        self.netBlocks = construct_cfg(cfgfile)
        self.DNInfo, self.moduleList = buildNetwork(self.netBlocks)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x, CUDA):
        detections = []
        modules = self.netBlocks[1:]
        layerOutputs = {}


        written_output = 0
        #Iterate throught each module
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            #Upsampling is basically a form of convolution
            if module_type == "convolutional" or module_type == "upsample" :

                x = self.moduleList[i](x)
                layerOutputs[i] = x

            #Add outouts from previous layers to this layer
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                #If layer nummber is mentioned instead of its position relative to the the current layer
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = layerOutputs[i + (layers[0])]

                else:
                    #If layer nummber is mentioned instead of its position relative to the the current layer
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = layerOutputs[i + layers[0]]
                    map2 = layerOutputs[i + layers[1]]


                    x = torch.cat((map1, map2), 1)
                layerOutputs[i] = x

            #ShortCut is essentially residue from resnets
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = layerOutputs[i-1] + layerOutputs[i+from_]
                layerOutputs[i] = x



            elif module_type == 'yolo':

                anchors = self.moduleList[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.DNInfo["height"])

                #Get the number of classes
                num_classes = int (modules[i]["classes"])

                #Output the result
                x = x.data
                print("Size before transform => " ,x.size())

                #Convert the output to 2D (batch x grids x bounding box attributes)
                x = transformOutput(x, inp_dim, anchors, num_classes, CUDA)
                print("Size after transform => " ,x.size())


                #If no detections were made
                if type(x) == int:
                    continue


                if not written_output:
                    detections = x
                    written_output = 1

                else:
                    detections = torch.cat((detections, x), 1)

                layerOutputs[i] = layerOutputs[i-1]


        try:
            return detections
        except:
            return 0


    def load_weights(self, weightfile):
        #Open the weights file
        file = open(weightfile, "rb")

        header = np.fromfile(file, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]


        weights = np.fromfile(file, dtype = np.float32)

        tracker = 0
        for i in range(len(self.moduleList)):
            module_type = self.netBlocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.moduleList[i]
                try:
                    batch_normalize = int(self.netBlocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                convPart = model[0]

                if (batch_normalize):
                    batch_nPart = model[1]
                    #Get the number of weights of Batch Norm Layer
                    biasCount = batch_nPart.bias.numel()
                    #Load the weights
                    batch_nBias = torch.from_numpy(weights[tracker:tracker + biasCount])
                    tracker += biasCount

                    batch_nPart_weights = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount

                    batch_nPart_running_mean = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount

                    batch_nPart_running_var = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    # loaded weights into dims of model weights.
                    batch_nBias = batch_nBias.view_as(batch_nPart.bias.data)
                    batch_nPart_weights = batch_nPart_weights.view_as(batch_nPart.weight.data)
                    batch_nPart_running_mean = batch_nPart_running_mean.view_as(batch_nPart.running_mean)
                    batch_nPart_running_var = batch_nPart_running_var.view_as(batch_nPart.running_var)
                    #Copy data to the model
                    batch_nPart.bias.data.copy_(batch_nBias)
                    batch_nPart.weight.data.copy_(batch_nPart_weights)
                    batch_nPart.running_mean.copy_(batch_nPart_running_mean)
                    batch_nPart.running_var.copy_(batch_nPart_running_var)

                else:
                    #Number of biases
                    biasCount = convPart.bias.numel()
                    #Load the weights
                    convBias = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker = tracker + biasCount
                    #reshape the loaded weights according to the dims of the model weights
                    convBias = convBias.view_as(convPart.bias.data)
                    #Finally copy the data
                    convPart.bias.data.copy_(convBias)

                #load the weights for the Convolutional layers
                weightfile = convPart.weight.numel()

                convWeight = torch.from_numpy(weights[tracker:tracker+weightfile])
                tracker = tracker + weightfile

                convWeight = convWeight.view_as(convPart.weight.data)
                convPart.weight.data.copy_(convWeight)


# Sample code for the upsampling
'''
class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x
'''
