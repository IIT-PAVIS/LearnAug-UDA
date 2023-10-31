import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):
    a, b, c, d = input.size()
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    output_tensor = torch.zeros([a, b, b], dtype=torch.float64)

    for idx, tensor in enumerate(input):
        features = tensor.view(b, c * d)

        G = torch.mm(features, features.t())
        G = G.div(b * c * d)

        output_tensor[idx,:,:] = G

    return output_tensor

def random_pooling(x1, x2, output_size):
    """
    Applies random pooling to the input tensors `x1` and `x2`.
    The output tensors will have the shape `output_size`.

    Args:
        x1: A PyTorch tensor with shape [N, C, H, W]
        x2: A PyTorch tensor with shape [N, C, H, W]
        output_size: A tuple of integers (out_H, out_W) specifying the output size.

    Returns:
        Two PyTorch tensors with shape [N, C, out_H, out_W].
    """
    # Compute the stride for downsampling the tensor
    N, C, H, W = x1.size()
    out_H, out_W = output_size
    stride_h = H // out_H
    stride_w = W // out_W

    # Compute the window size
    window_h = stride_h + 1
    window_w = stride_w + 1

    # Compute the output tensors using random pooling
    y1 = torch.zeros([N, C, out_H, out_W])
    y2 = torch.zeros([N, C, out_H, out_W])
    for i in range(out_H):
        for j in range(out_W):
            x1_window = x1[:, :, i*stride_h:i*stride_h+window_h, j*stride_w:j*stride_w+window_w]
            x2_window = x2[:, :, i*stride_h:i*stride_h+window_h, j*stride_w:j*stride_w+window_w]
            x1_window_reshaped = x1_window.reshape(N, C, -1)
            x2_window_reshaped = x2_window.reshape(N, C, -1)
            rand_idx = torch.randint(x1_window_reshaped.shape[-1], [1])
            y1[:, :, i, j] = x1_window_reshaped[:, :, rand_idx].squeeze(dim=-1)
            y2[:, :, i, j] = x2_window_reshaped[:, :, rand_idx].squeeze(dim=-1)

    return y1, y2

class PerceptualLoss(nn.Module):

    def __init__(self, style_layers, content_layers, alpha=1.0, beta=1.0, consider_content=False, style_weights=[1.0,1.0,1.0], content_weights=[1.0,1.0], type_eval='feat'):
        super(PerceptualLoss, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.alpha = alpha
        self.beta = beta
        self.consider_content = consider_content
        self.style_weights = style_weights
        self.content_weights = content_weights
        self.type_eval = type_eval

    def getAvgPMat(self, source_feats, target_feats):
        ksize = source_feats.size(2)//4
        stride = source_feats.size(2)//4
        avgpool = nn.AvgPool2d(kernel_size=ksize, stride=stride)
        return avgpool(source_feats), avgpool(target_feats)

    def forward(self, source, augmented_source, target):

        style_loss = 0
        content_loss = 0

        for idx, style_layer in enumerate(self.style_layers):
            if self.type_eval == 'feat':
               loss = torch.norm(gram_matrix(augmented_source[style_layer]) - gram_matrix(target[style_layer]), p='fro')
            else:
               mod_aug_source, mod_target = self.getAvgPMat(augmented_source[style_layer], 
                                                            target[style_layer])
               loss = F.mse_loss(mod_aug_source,mod_target)

            style_loss += self.style_weights[idx]*loss

        if self.consider_content:
            for idx, content_layer in enumerate(self.content_layers):
                if self.type_eval == 'feat':
                    loss = F.mse_loss(augmented_source[content_layer], source[content_layer])
                else:
                    mod_aug_source, mod_source = self.getAvgPMat(augmented_source[content_layer], 
                                                                 source[content_layer])
                    loss = F.mse_loss(mod_aug_source,mod_source)
                
                content_loss += self.content_weights[idx]*loss

        if self.consider_content:
           return self.alpha*content_loss + self.beta*style_loss
        else:
           return self.beta*style_loss
