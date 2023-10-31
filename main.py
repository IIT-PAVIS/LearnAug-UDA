import os
import csv
import torch
import wandb
import random
import logging
import warnings
import datetime
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim
from torch.utils import data
from lib.losses import loss_handler
from torchsummary import summary
from lib.models import build_model
from lib.dataloaders import loader
import torchvision.datasets as datasets
from lib.utils import utils, lr_scheduler
from torchvision.utils import save_image, make_grid

logger = logging.getLogger(__name__)

dataset_dict = {'clipart': 'c',
                'sketch':'sk',
                'real':'r',
                'infograph':'inf',
                'painting':'p',
                'quickdraw':'q',
                'train':'tr',
                'test':'tst',
                'validation':'val'}

domain_benchmark = {'DomainNet': 'DN',
                    'VisDAc':'VD'}

visdac_classes = {'aeroplane':0,
                 'bicycle':1,
                 'bus':2,
                 'car':3,
                 'horse':4,
                 'knife':5,
                 'motorcycle':6,
                 'person':7,
                 'plant':8,
                 'skateboard':9,
                 'train':10,
                 'truck':11}

def getExpName(args):

    time = datetime.datetime.now()

    expName = domain_benchmark[args.benchmark]
    expName = expName + dataset_dict[args.source_dataset]
    expName = expName + '2' + dataset_dict[args.target_dataset]
    expName = expName + time.strftime("%m%d%H%M")

    expName = expName + 'r' + str(args.run)

    return expName

def logExperiments(args, dict_test, acc_source, acc_target):

    args_names = []
    args_values = []

    for k, v in vars(args).items():
        args_names.append(k)
        args_values.append(v)

    if args.benchmark == 'VisDAc':
        for name in dict_test.keys():
            args_names.append(name)
    else:
        args_names.append('acc_source')
        args_names.append('acc_target')

    file = os.path.join(args.log_dir,'experiments.csv')

    if not os.path.exists(file):
       with open(file, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=args_names, delimiter=';')
            writer.writeheader()

    dict_results={}
    
    with open(file, mode='a') as csv_file:
         writer = csv.DictWriter(csv_file, fieldnames=args_names, delimiter=';')

         for idx in range(0,len(args_values)):
             dict_results[args_names[idx]] = args_values[idx]

         if args.benchmark == 'VisDAc':
            for name, val in dict_test.items():
                dict_results[name] = val
         else:
            dict_results['acc_source'] = acc_source
            dict_results['acc_target'] = acc_target

         writer.writerow(dict_results)

def test(device, testloader, net):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print('Acc: %.3f%% (%d/%d)' % (acc, correct, total))

    return acc

def test_class(device, testloader, net):
    net.eval()
    pred = torch.zeros(len(testloader.dataset))
    labels = torch.zeros(len(testloader.dataset))
    begin = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc='Processed samples (%):')):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)

            end = begin + len(inputs)

            pred[begin:end] = predicted
            labels[begin:end] = targets

            begin = end


    dict_acc = getAccuracies(pred, labels)

    return dict_acc

def getAccuracies(pred, labels):

    dict_acc = {}
    acc_sum = 0

    for name, id in visdac_classes.items():
        idx = torch.eq(labels,id)
        correct = torch.count_nonzero(torch.eq(pred[idx], id))
        total = torch.count_nonzero(idx)

        acc = 100 * (correct / total)
        dict_acc[name] = acc.item()
        acc_sum = acc_sum + acc

    dict_acc['mean'] = acc_sum.item() / len(visdac_classes)
        
    return dict_acc

def getStyleWeights(weights_type, num_layers, weights=None):

    weights_style = np.ones(num_layers)

    if weights_type == 'log':
        weights_style = np.logspace(1, 0.1, num_layers, endpoint=True)/10.0
    elif weights_type == 'quad':
        weights_style = np.array([ pow(i,2)/pow(num_layers,2) for i in range(num_layers, 0, -1)])
    elif weights_type == 'udef':
            weights = np.fromiter(map(float, weights.split(',')), float)
            if  weights.shape[0] == num_layers:
                weights_style = weights
            else:
                raise NotImplementedError('Provived style weights are not accepted.')
    else:
        raise NotImplementedError('Style weights type ({weights_type}) not define')

    return weights_style

def main(args):

    logger.info(args)
    # Setup GPU
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise NotImplementedError('CUDA is unavailable.')

    # Dataset
    if args.benchmark == 'DomainNet':

        source_dir = os.path.join(args.dataset_dir,args.source_dataset)
        train_source_list =os.path.join(source_dir,'labeled_source_images_{}.txt'.format(args.source_dataset))
        test_source_list =os.path.join(source_dir,'validation_target_images_{}_3.txt'.format(args.source_dataset))

        if args.target_batch_size == 1 or args.target_batch_size == 3:
            target_dir = os.path.join(args.dataset_dir,args.target_dataset)
            train_target_list =os.path.join(target_dir,'labeled_target_images_{}_{}.txt'.format(args.target_dataset, args.target_batch_size))
            test_target_list =os.path.join(target_dir,'unlabeled_target_images_{}_{}.txt'.format(args.target_dataset, args.target_batch_size))
        elif args.target_batch_size == 2 or (args.target_batch_size > 3 and args.target_batch_size <= 10):
            target_dir = os.path.join(args.dataset_dir,args.target_dataset)
            train_target_list =os.path.join(target_dir,'labeled_target_images_{}_10.txt'.format(args.target_dataset))
            test_target_list =os.path.join(target_dir,'unlabeled_target_images_{}_10.txt'.format(args.target_dataset))
        else:
            raise NotImplementedError('Number of target samples accepted is 1 or 3')

        n_classes = 126
        source_train_loader = loader.createLoader(args.num_workers, args.input_size, source_dir, train_source_list, train=True, batch_size=args.batch_size)
        target_train_loader = loader.createLoader(args.num_workers, args.input_size, target_dir, train_target_list, train=True, batch_size=args.target_batch_size)

        source_eval_loader = loader.createLoader(args.num_workers, args.input_size, source_dir, test_source_list, train=False, batch_size=args.batch_size)
        target_eval_loader = loader.createLoader(args.num_workers, args.input_size, target_dir, test_target_list, train=False, batch_size=args.batch_size)

    elif args.benchmark == 'VisDAc':

        args.source_dataset = 'train'
        args.target_dataset = 'validation'

        source_dir = os.path.join(args.dataset_dir, 'VisDAc', args.source_dataset)
        train_source_list = os.path.join(source_dir,'image_list.txt')
        test_source_list = os.path.join(source_dir,'image_list.txt')


        target_dir = os.path.join(args.dataset_dir, 'VisDAc', args.target_dataset)
        train_target_list =os.path.join(target_dir,'image_list.txt')

        test_dir = os.path.join(args.dataset_dir, 'VisDAc', 'test')
        test_target_list =os.path.join(test_dir,'image_list.txt')

        n_classes = 12
        source_train_loader = loader.createLoader(args.num_workers, args.input_size, source_dir, train_source_list, train=True, batch_size=args.batch_size)
        target_train_loader = loader.createLoader(args.num_workers, args.input_size, target_dir, train_target_list, train=True, batch_size=args.target_batch_size)

        source_eval_loader = loader.createLoader(args.num_workers, args.input_size, source_dir, test_source_list, train=False, batch_size=args.batch_size)
        test_eval_loader = loader.createLoader(args.num_workers, args.input_size, test_dir, test_target_list, train=False, batch_size=args.batch_size)

    else:
        raise NotImplementedError('Benchmark not recognized.')

    # Model
    model = build_model('resnet101', n_classes).to(device)

    pretrained_classifier = os.path.join(args.classifier_dir, args.classifier_name)

    if args.pretrained_model:
        if os.path.exists(pretrained_classifier):
            logger.info('Loading pretrained classifier')
            model.load_state_dict(torch.load(pretrained_classifier)['model'])
        else:
            raise NotImplementedError('Pretrained model not found.')

    model.train()

    # VGG Teacher - Setting Layers to use
    if args.pl_type == '123':
        style_layers = []
        content_layers = ['features.1','features.6','features.11']
        args.consider_content = True
    elif args.pl_type == '345':
        style_layers = []
        content_layers = ['features.11','features.18','features.25']
        args.consider_content = True
    elif args.pl_type == 'full1':
        style_layers = ['features.3','features.8','features.15']
        content_layers = ['features.22', 'features.29']
    elif args.pl_type == 'full2':
        style_layers = ['features.3','features.8','features.15','features.22']
        content_layers = ['features.15']
    else:
        raise NotImplementedError('Perceptual loss type not implemented.')

    StyleAlignmentMod = build_model(model_name="vgg16", style_layers=style_layers, content_layers=content_layers).to(device)
    StyleAlignmentMod.eval()

    # Target Image (1 Random Target)
    targetloader_iter = enumerate(target_train_loader)
    _, batch_t = next(targetloader_iter)
    target_imgs_ori, _ = batch_t
    target_imgs_ori = target_imgs_ori.to(device)

    if args.apply_rndCrop:
        h, w = map(int, args.input_size.split(','))
        RandomCropResize = torchvision.transforms.RandomResizedCrop(size=(h, w), scale=(0.2, 0.5))

    #Save target image
    if args.vis:
        save_image(make_grid(target_imgs_ori), os.path.join(args.log_dir, args.expName, 'target_imgs.png'))

    # Augmentation Module
    if args.vae_type == 'UNIT':
        AugModule = build_model(model_name='UNIT', 
                                device=device, 
                                conditioned=True, 
                                alpha=args.alpha, 
                                beta=args.beta, 
                                UseDConv=args.UseDConv, 
                                Upsampling_type=args.Upsampling_type).to(device)
        #Freezing weights
        AugModule.freezeEncoder()
        AugModule.freezeDecoder()
    elif args.vae_type == 'DED':
        AugModule = build_model(model_name='DED', 
                                device=device,  
                                conditioned=False, 
                                UseDConv=args.UseDConv, 
                                Upsampling_type=args.Upsampling_type).to(device)
        #Freezing weights
        AugModule.freezeStyleEncoder()
        AugModule.freezeContentEncoder()
        AugModule.freezeDecoder()
    else:
        raise NotImplementedError('VAE type not implemented.')
    
    if args.pretrained_style:
        AugModule.load_state_dict(torch.load(os.path.join(args.load_dir,args.load_name)))    

    # Optimizer
    optim_cls = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)
    optim_aug = optim.AdamW(AugModule.parameters(), lr=args.aug_lr, weight_decay=args.aug_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWithLinearWarmup(optim_cls, 5, args.n_epochs)

    if len(style_layers) > 0:
        style_weights = getStyleWeights(args.weights_type, len(style_layers), weights=args.style_weights)
    else:
        style_weights = None

    content_weights = np.fromiter(map(float, args.content_weights.split(',')), float)

    # Objective function
    objective = loss_handler.LossHandler(model, 
                                        StyleAlignmentMod, 
                                        AugModule,
                                        style_layers,
                                        content_layers,
                                        args.weight_decay,
                                        args.consider_content,
                                        style_weights = style_weights,
                                        content_weights = content_weights,
                                        alpha=args.weight_content,
                                        beta=args.weight_style,
                                        vae_type=args.vae_type,
                                        type_eval=args.type_eval,
                                        recLoss=args.recLoss).to(device)

    # Training loop
    st_epoch = 1
    logger.info('training')
    meter = utils.AvgMeter()
    target_imgs = target_imgs_ori.clone().detach()
    #target_imgs = target_imgs.to(device)

    one_hot = torch.zeros(size=(args.target_batch_size,args.target_batch_size)).to(device)
    for i in range(0,args.target_batch_size):
        one_hot[i,i]=1.0

    for epoch in range(st_epoch, args.n_epochs + 1):

        if args.apply_rndCrop:
            target_imgs = target_imgs_ori.clone().detach()
            target_imgs = RandomCropResize(target_imgs)
            save_image(make_grid(target_imgs), os.path.join(args.log_dir, args.expName, 'target_imgs_{}.png'.format(epoch)))

        for i, data in enumerate(source_train_loader):

            torch.cuda.synchronize()
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            idx = torch.randint(low=0, high=args.target_batch_size, size=(args.batch_size,))
            target_imgs = target_imgs.to(device)
            context = target_imgs[idx,:,:,:]

            # Update augmentation
            if i % args.n_inner == 0:

                if args.vae_type == 'DED':
                    objective.AugModule.unfreezeStyleEncoder()
                    objective.AugModule.unfreezeContentEncoder()
                else: #UNIT
                    objective.AugModule.unfreezeEncoder()

                objective.AugModule.unfreezeDecoder()

                optim_aug.zero_grad()
                loss_aug, res, aug_targets = objective(inputs, targets, target_imgs[idx,:,:,:], target_imgs_ori, context, 'loss_perceptual')
                loss_aug.backward()
                optim_aug.step()
                meter.add(res)

                if args.vae_type == 'DED':
                    if not objective.AugModule.IsStyleEncFrozen:
                        objective.AugModule.freezeStyleEncoder()

                    if not objective.AugModule.IsContentEncFrozen:
                        objective.AugModule.freezeContentEncoder()
                else: #UNIT
                    if not objective.AugModule.IsEncFrozen:
                        objective.AugModule.freezeEncoder()

                if not objective.AugModule.IsDecFrozen:
                    objective.AugModule.freezeDecoder()

            # Update target model
            optim_cls.zero_grad()
            loss_cls, res, aug_img = objective(inputs, targets, target_imgs[idx,:,:,:], target_imgs_ori, context, 'cls')
            loss_cls.backward()
            optim_cls.step()

            # Adjust learning rate
            scheduler.step(epoch - 1. + (i + 1.) / len(source_train_loader))

            # Print losses and accuracy
            meter.add(res)
            if (i + 1) % args.print_freq == 0:
                logger.info(meter.state(f'epoch {epoch} [{i+1}/{len(source_train_loader)}]',
                                        f'lr {optim_cls.param_groups[0]["lr"]:.4e}'))

        # Save checkpoint
        state, epoch_dict = meter.mean_state(f'epoch [{epoch}/{args.n_epochs}]',f'lr {optim_cls.param_groups[0]["lr"]:.4e}')
        logger.info(state)

        checkpoint = {'model': model.state_dict(),
                      'objective': objective.state_dict(), # including ema model and replay buffer
                      'optim_cls': optim_cls.state_dict(),
                      'optim_aug': optim_aug.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, os.path.join(args.log_dir, args.expName, 'checkpoint.pth'))

        # Save augmented images
        if args.vis:

            AugModule.eval()

            if inputs.size(dim=0) != args.target_batch_size:
                idx = torch.randint(low=0, high=args.target_batch_size, size=(args.batch_size,))
            else:
                idx = range(0,args.target_batch_size)

            ctxt = target_imgs[idx,:,:,:]

            if args.vae_type == 'UNIT':
                aug_img, _, _ = AugModule(inputs,ctxt)
            elif args.vae_type == 'DED':
                aug_img = AugModule(inputs,ctxt)
            else:
                raise NotImplementedError('VAE type not implemented.')

            AugModule.train()

            save_image(aug_img, os.path.join(args.log_dir, args.expName, f'{epoch}epoch_aug_img.png'))
            save_image(inputs, os.path.join(args.log_dir, args.expName, f'{epoch}epoch_img.png'))

    # Evaluation
    if args.benchmark != 'VisDAc':
        logger.info('Source Evaluation')
        acc_source = test(device, source_eval_loader, model)
        logger.info(f'{args.source_dataset} Accuracy (%) {acc_source} ')

        logger.info('Target Evaluation')
        acc_target = test(device, target_eval_loader, model)
        logger.info(f'{args.target_dataset} Accuracy (%) {acc_target} ')

        logExperiments(args, None, acc_source, acc_target)

    else:
        logger.info('Source Evaluation')
        dict_train = test_class(device, source_eval_loader, model)
        logger.info(f'{args.source_dataset} Accuracies (%) {dict_train} ')

        logger.info('Target Evaluation')
        dict_test = test_class(device, test_eval_loader, model)
        logger.info(f'Test Accuracies (%) {dict_test} ')

        logExperiments(args, dict_test, acc_source, acc_target)

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Input Image Options
    parser.add_argument("--input_size", type=str, default='224,224', help="Comma-separated string with height and width of the images.")
    parser.add_argument("--channels", type=int, default=3, help="Specify the number of channels to use (rgb - 3, grayscale - 1).")

    # Datasets Options
    parser.add_argument("--source_dataset", type=str, default='mnist', help="Source dataset (mnist, usps, svhn).")
    parser.add_argument("--target_dataset", type=str, default='svhn', help="Target dataset (mnist, usps, svhn).")
    parser.add_argument("--benchmark", type=str, default='Digits', help="Benchmark to use (Digits, DomainNet, VisdaC).")
    parser.add_argument('--dataset_dir', default='./dataset', type=str)

    # Experiment Options
    parser.add_argument('--run', default=1, type=int, help='')
    parser.add_argument('--num_workers', '-j', default=8, type=int, help='the number of data loading workers')
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--target_batch_size', default=1, type=int)
    parser.add_argument('--log_dir', default='./experiments', type=str)

    # Classifier Options
    parser.add_argument('--pretrained_model', action='store_true', help='Indicates that a pretrained model for the classifier is to be load.')
    parser.add_argument('--classifier_dir', default='./pretrained', help='Directory to load checkpoint file.')
    parser.add_argument('--classifier_name', default='classifier.cktp', help='Name of checkpoint file.')


    # Optimization
    parser.add_argument('--seed', default=0, type=int) 
    parser.add_argument('--lr', default=0.0001, type=float,help='learning rate')
    parser.add_argument('--aug_lr', default=1e-3, type=float,help='learning rate for augmentation model')
    parser.add_argument('--weight_decay', '-wd', default=5e-4, type=float)
    parser.add_argument('--aug_weight_decay', '-awd', default=1e-2, type=float,help='weight decay for augmentation model')


    # Encoder-Decoder
    parser.add_argument('--load_dir', default='./pretrained', type=str)
    parser.add_argument('--load_name', default='model.ckpt', type=str)
    parser.add_argument('--pretrained_style', action='store_true', help='Indicates that a pretrained model for the Encoder-Decoder is to be load.')

    # Fixed augmentation
    parser.add_argument("--crop_size", type=str, default='20,20', help="Comma-separated string with height and width of the images.")    

    parser.add_argument('--n_inner', default=5, type=int, help='the number of iterations for inner loop (i.e., updating classifier)')

    # Improvement techniques
    parser.add_argument('--epsilon', default=0.1, type=float, help='epsilon for the label smoothing')

    # Debugging Options
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--vis', action='store_true', help='visualize augmented images')
    
    # Perceptual Loss Options
    parser.add_argument('--pl_type', default='full1', type=str, help="Specified the perceptual loss to use (123,345, or full).")
    parser.add_argument('--weights_type', default='udef', type=str, help='Indicates the types of weights used for the perceptual style loss (log, quad, udef')
    parser.add_argument('--style_weights', default='1.0,1.0,1.0', type=str, help='User define style weights \'1.0,1.0,1.0\'.')
    parser.add_argument('--content_weights', default='1.0,1.0', type=str, help='User define style weights \'1.0,1.0,1.0\'.')
    parser.add_argument('--weight_content', default=1.0, type=float, help='Weight for content in the perceptual loss.')
    parser.add_argument('--weight_style', default=1.0, type=float, help='Weight for style in the perceptual loss.')
    parser.add_argument('--type_eval', default='mat_avg', help='Indicates which type of evaluation to use for the the style loss (feat, mat_avg) for comparing the two inputs.')
    parser.add_argument('--consider_content', action='store_true', help='considers the content in the perceptual loss')

    # Data Augmentation Options
    parser.add_argument('--apply_rndCrop', action='store_true', help='Applies random cropping to the target samples as to increase the available target data.')

    # Encoder-Decoder Options
    parser.add_argument('--vae_type', default='DED', type=str, help='Type of Encoder-Decoder to use: VAE, UNIT.')

    # Shared-Encoder Options
    parser.add_argument('--alpha', default=2.0, type=float, help='alpha value for mixup operation.')
    parser.add_argument('--beta', default=2.0, type=float, help='alpha value for mixup operation.')

    # Disentangled Encoder Options
    parser.add_argument('--recLoss', action='store_true', help='Indicates if reconstruction loss is applied')    

    # Decoder Options
    parser.add_argument('--UseDConv', action='store_true', help='Indicates if Deconvolution layers should be use on decoder or Upsampling + Conv')
    parser.add_argument('--Upsampling_type', default='bilinear', type=str, help='Type of upsampling done when using Upsampling + Conv (nearest or bilinear)')



    args = parser.parse_args()

    if args.benchmark == 'VisDAc':
        args.source_dataset = 'train'
        args.target_dataset = 'validation'

    args.expName = getExpName(args)

    if not os.path.exists(os.path.join(args.log_dir, args.expName)):
       os.mkdir(os.path.join(args.log_dir, args.expName))

    utils.setup_logger(os.path.join(args.log_dir, args.expName), False)

    if args.seed != 0:

       torch.manual_seed(args.seed)
       np.random.seed(args.seed)
       random.seed(args.seed)

    else:
       torch.backends.cudnn.benchmark = True

    main(args)
