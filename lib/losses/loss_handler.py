import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.losses import perceptual_loss

class LossHandler(nn.Module):

    def __init__(
            self, model, StyleAlignmentMod,
            AugModule, style_layers, content_layers,
            weight_decay=0, consider_content=False, style_weights=[1.0,1.0,1.0], content_weights=[1.0,1.0], alpha=1.0, beta=1.0,
            vae_type='UNIT', type_eval='feat', recLoss=False):
        super().__init__()

        self.model = model
        self.StyleAlignmentMod = StyleAlignmentMod
        self.weight_decay = weight_decay
        self.AugModule = AugModule
        self.perceptual_criterion = perceptual_loss.PerceptualLoss(style_layers=style_layers,
                                                                   content_layers=content_layers,
                                                                   alpha=alpha,
                                                                   beta=beta,
                                                                   consider_content=consider_content,
                                                                   style_weights=style_weights,
                                                                   content_weights=content_weights,
                                                                   type_eval=type_eval)
        self.vae_type=vae_type
        self.recLoss = recLoss
        self.mse=torch.nn.MSELoss()

    def manual_weight_decay(self, model, coef):
        return coef * (1. / 2.) * sum([params.pow(2).sum()
                                       for name, params in model.named_parameters()
                                       if not ('_bn' in name or '.bn' in name)])

    def forward(self, x, y, target_imgs, ori_targets, c=None, loss='cls'):

        if loss == 'cls':
            return self.loss_classifier(x, y, c)
        elif loss == 'loss_perceptual':
            return self.loss_perceptual(x, y, target_imgs, ori_targets, c)
        else:
            raise NotImplementedError

    def loss_classifier(self, x, y, c=None):

        # Augment input samples
        with torch.no_grad():
            if self.vae_type == 'UNIT':
                aug_x, _, _ = self.AugModule(x, c)
            elif self.vae_type == 'DED': 
                aug_x = self.AugModule(x, c)
                
        # Calculate classification loss
        pred = self.model(aug_x)
        loss = F.cross_entropy(pred, y)

        res = {'loss cls. step1': loss.item()}

        if self.weight_decay > 0:
            loss += self.manual_weight_decay(self.model, self.weight_decay)

        return loss, res, aug_x


    def loss_perceptual(self, x, y, target_imgs, ori_targets, c=None):

        # Augment input samples
        if self.vae_type == 'UNIT':
            aug_x, mean, _ = self.AugModule(x, c)
        elif self.vae_type == 'DED':
            aug_x = self.AugModule(x, c)
        
        if target_imgs.shape[0] == 1:
            target_imgs = target_imgs.repeat((x.shape[0],1,1,1))

        # Calculate features maps
        source = self.StyleAlignmentMod(x)
        augmented_source = self.StyleAlignmentMod(aug_x)

        if aug_x.shape[0] != target_imgs.shape[0]:
           target_imgs = target_imgs[0:aug_x.shape[0],:,:,:]
        else:
           target_imgs = target_imgs

        target = self.StyleAlignmentMod(target_imgs.cuda())

        # Calculate perceptual loss
        loss_perc = self.perceptual_criterion(source, augmented_source, target)

        # Calculate Reconstruction loss
        if self.vae_type == 'DED' and self.recLoss:
            input_imgs = torch.concat((ori_targets, x[:-ori_targets.size(dim=0)]))
            img_rec = self.AugModule(input_imgs, input_imgs)
            loss_rec = self.mse(input_imgs,img_rec)

        res = {'step2-loss perc.': loss_perc.item()}

        if self.vae_type == 'DED' and self.recLoss:
            return loss_perc + loss_rec, res, target_imgs
        else:
            return loss_perc, res, target_imgs