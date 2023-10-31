import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import vgg16 as v16
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

class vgg16(nn.Module):
    def __init__(self, style_layers, content_layers):
        super(vgg16, self).__init__()

        self.vgg16 = v16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.vgg16.cuda()
        self.output_nodes = []


        for node in style_layers:
            self.output_nodes.append(node)

        for node in content_layers:
            self.output_nodes.append(node)

        self.vgg16 = create_feature_extractor(self.vgg16, return_nodes=self.output_nodes)

        for vgg_p in self.vgg16.parameters():
            vgg_p.requires_grad = False

    def forward(self, x):

        outputs = self.vgg16(x)
        
        return outputs