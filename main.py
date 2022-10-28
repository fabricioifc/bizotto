from skimage import io
import os
import torch
import numpy as np
import pandas as pd
import random
from glob import glob

from dataset import DatasetIcmbio
from trainer import Trainer
from models import SegNet, SegNet_two_pools_test
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from utils import clear, convert_to_color, make_optimizer, visualize_augmentations
from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

if __name__=='__main__':

    # Params
    params = {
        # 'root_dir': 'D:\datasets\Vaihingen',
        'root_dir': 'D:\\datasets\\ICMBIO\\all',
        #  'root_dir': 'D:\\Export_Google_TIF\\20220803\\saida',
        'window_size': (256, 256),
        'cache': True,
        'bs': 16,
        'n_classes': 9,
        'classes': ["Urbano", "Mata", "Piscina", "Sombra", "Regeneracao", "Agricultura", "Formação Rochosa", "Solo Exposto", "Água"],
        'cpu': None,
        'device': 'cuda',
        'precision' : 'full',
        'optimizer_params': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005
        },
        'lrs_params': {
            'type': 'multi',
            'milestones': [25, 35, 45],
            'gamma': 0.1
        },
        'weights': '',
        'maximum_epochs': 100,
        'save_epoch': None,
        'model': {
            'encoder': 'resnet18',
            'encoder_weights': 'imagenet',
            'activation': None,
        }
    }

    params['weights'] = torch.ones(params['n_classes']) 
    
    image_dir = os.path.join(params['root_dir'], 'images')
    label_dir = os.path.join(params['root_dir'], 'label')

    # Load image and label files from .txt
    train_images = pd.read_table('train_images.txt',header=None).values
    train_images = [os.path.join(image_dir, f[0]) for f in train_images]
    train_labels = pd.read_table('train_labels.txt',header=None).values
    train_labels = [os.path.join(label_dir, f[0]) for f in train_labels]
    
    test_images = pd.read_table('test_images.txt',header=None).values
    test_images = [os.path.join(image_dir, f[0]) for f in test_images]
    test_labels = pd.read_table('test_labels.txt',header=None).values
    test_labels = [os.path.join(label_dir, f[0]) for f in test_labels]
    
    # image_folder = os.path.join(params['root_dir'], 'images/000000{}.tif')
    # label_folder = os.path.join(params['root_dir'], 'label/000000{}.tif')
    # all_files = sorted(glob(label_folder.replace('{}', '*')))
    # all_ids = [os.path.split(f)[1].split('.')[0] for f in all_files]
    
    
    # split train/val (80/20)
    # train_images, val_images = np.split(train_images, [int(len(train_images)*0.8)])
    # train_labels, val_labels = np.split(train_labels, [int(len(train_labels)*0.8)])
    # train_images, val_images = train_images.tolist(), val_images.tolist()
    # train_labels, val_labels = train_labels.tolist(), val_labels.tolist()
    
    # Create train and test sets
    train_dataset = DatasetIcmbio(train_images, train_labels, window_size = params['window_size'], cache = params['cache'])
    test_dataset = DatasetIcmbio(test_images, test_labels, window_size = params['window_size'], cache = params['cache'],augmentation=False)

    # # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'], num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'], num_workers=0, shuffle=False)
    
    # visualize_augmentations(train_dataset)
    
    # # Load network model in cuda (gpu)
    # model = SegNet(in_channels = 3, out_channels = params['n_classes'])
    model = SegNet_two_pools_test(in_channels = 3, out_channels = params['n_classes'], pretrained = True, pool_type = 'dwt')
    model.cuda()
    
    # model = smp.UnetPlusPlus(
    #     encoder_name=params['model']['encoder'],
    #     encoder_weights=params['model']['encoder_weights'],
    #     in_channels=3,
    #     classes=params['n_classes'],
    #     activation=params['model']['activation']
    # )
    
    # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    loader = {
        "train": train_loader,
        "test": test_loader,
    }
    
    # checkpoint = torch.load('D:/Projetos/aerialseg_kaggle/results/20221010/segnet256_epoch140_88.16359915384432')
    # model.load_state_dict(checkpoint)
    
    trainer = Trainer(model, loader, params, cbkp='D:/Projetos/aerialseg_kaggle/bizotto/tmp/20221027_focal_dice/segnet_final_60')
    # print(trainer.test(stride = 32, all = False))
    # _, all_preds, all_gts = trainer.test(all=True, stride=32)
    clear()
    
    for epoch in range(1, params['maximum_epochs'] + 1):
        # clear()
        # print('Starting training epoch {}/{}.'.format(epoch, params['maximum_epochs']))
        
        train_metric = trainer.train()
        
        if params['save_epoch'] is not None and epoch % params['save_epoch'] == 0:
            acc = trainer.test(stride = min(params['window_size']), all=False)
            trainer.save('./segnet256_epoch_{}_{:.2f}'.format(epoch, acc))
    
    acc, all_preds, all_gts = trainer.test(all=True, stride=32)
    trainer.save('./segnet_final_{}_{:.2f}'.format(epoch, acc))
    
    input_ids, label_ids = test_loader.dataset.get_dataset()
    all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
    
    for p, id_ in zip(all_preds, all_ids):
        img = convert_to_color(p)
        # plt.imshow(img) and plt.show()
        io.imsave('./tmp/inference_tile_{}.png'.format(id_), img)
        
    