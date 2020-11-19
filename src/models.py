import os

import numpy as np

from torchvision import models
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

def label2onehot(labels, pad_value):

    # input labels to one hot vector
    inp_ = T.unsqueeze(labels, 2)
    one_hot = T.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot[:, 0] = 0

    return one_hot

class MTFoodClassify(nn.Module):
    def __init__(self, lr, inpt_dims, fc1_dims, out_dims, architecture, encoder_dir, model_dir=''):

        super(MTFoodClassify, self).__init__()

        self.lr = lr
        self.inpt_dims = inpt_dims
        self.fc1_dims = fc1_dims
        self.out_dims = out_dims
        self.architecture = architecture
        self.encoder_dir = os.path.join(encoder_dir)
        self.model_dir = os.path.join(model_dir)

        if self.architecture == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
            self.dim = 512
        if self.architecture == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        if self.architecture == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        if self.architecture == 'resnet152':
            self.model = models.resnet152(pretrained=True)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        self.model = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, self.out_dims)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.inpt_dims, self.out_dims)

        #self.initialization()

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print("image classifying on device: ", self.device)

    def forward(self, img_inputs):
        #out = self.fc1(img_inputs)
        #out = F.leaky_relu(out)
        x = self.model.conv1(img_inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def save_checkpoint(self):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.model_dir + 'model')

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.model_dir + 'model'))

class MTFoodFeature(nn.Module):
    def __init__(self, architecture, encoder_dir):
        super(MTFoodFeature, self).__init__()
        self.architecture = architecture
        self.encoder_dir = os.path.join(encoder_dir)
        self.initialization()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print("feature extracting on device: ", self.device)

    def initialization(self):
        net_in = getattr(models, self.architecture)(pretrained=True)
        # take only convolutions for features,
        # always ends with ReLU to make last activations non-negative
        if self.architecture.startswith('alexnet'):
            features = list(net_in.features.children())[:-1]
        elif self.architecture.startswith('vgg'):
            features = list(net_in.features.children())[:-1]
        elif self.architecture.startswith('resnet'):
            features = list(net_in.children())[:-1]
        elif self.architecture.startswith('densenet'):
            features = list(net_in.features.children())
            features.append(nn.ReLU(inplace=True))
        elif self.architecture.startswith('squeezenet'):
            features = list(net_in.features.children())
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(self.architecture))

        self.image_encoder = nn.Sequential(*features)
        print(">> {}: for '{}' custom pretrained features '{}' are used"
              .format(os.path.basename(__file__), self.architecture, os.path.basename(FEATURES[self.architecture])))
        self.image_encoder.load_state_dict(model_zoo.load_url(FEATURES[self.architecture], model_dir=self.encoder_dir))

        # Freeze those weights
        '''for p in self.image_encoder.parameters():
            p.requires_grad = False'''

    def forward(self, img_inputs):
        img_features = self.image_encoder(img_inputs)
        img_features = img_features.view(img_features.size(0), -1)
        return img_features

    def save_checkpoint(self):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.encoder_dir + 'encoder')

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.encoder_dir + 'encoder'))

def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def pretrained_model(num_classes, pretrained = False):

    pass


def model_C(num_classes, pretrained = False):
    pass