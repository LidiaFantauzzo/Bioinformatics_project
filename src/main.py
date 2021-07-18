import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
from dataset import BraTSData
from train import Trainer
from torch.utils import data
import time

from utils.args import parse_args

from segmentation_net import create_model
from metrics import mIoU


def main():
    ######### Set Seeds 
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ########## Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ########## Dataset

    train_data = BraTSData(task = 'train')
    #train_data[1]
    val_data = BraTSData(task= 'val')
    test_data = BraTSData(task= 'test')

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size,shuffle=True,drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=args.test_batch_size,drop_last=True)
    test_loader = data.DataLoader(test_data, batch_size=args.test_batch_size, drop_last=True)

    ########## Model

    model = create_model(args)
    model = model.to(device)

    ######### Metrics
    val_metrics = mIoU(args.num_classes)
    train_metrics = mIoU(args.num_classes)

    ######### Training
    # create class of training 
    trainer = Trainer(model, device, args)

    losses = []
    train_miou = []
    miou = []
    for ep in range(args.num_epochs):

        start = time.time()
        epoch_losses = trainer.train(cur_epoch=ep, train_loader=train_loader, metrics=train_metrics)
        losses.append(np.mean(epoch_losses))
        end = time.time()

        duration = int(end-start)
        print(f"Training will finish in about {duration * (args.num_epochs - ep) // 60} minutes")
        
        epoch_result = train_metrics.get_results()
        train_metrics.reset()
        train_miou.append(epoch_result['Mean IoU'])

        if (ep + 1) % args.val_interval == 0 or ep + 1 == args.num_epochs:
            _, plt_samples =  trainer.validate(loader=val_loader, metrics=val_metrics, plt_samples_bool = False)
            val_result = val_metrics.get_results()
            miou.append(val_result['Mean IoU'])

    # img = plt_samples[0][0][0,:,:].squeeze()
    # target = plt_samples[0][1]
    # pred = plt_samples[0][2]
    # fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    # ax.imshow(img, cmap ='bone')
    # ax.imshow(np.ma.masked_where(pred == False, pred),cmap='Spectral', alpha=0.6)
    # ax.set_title("Predicted tumor area over T1 modality")
    # plt.savefig("ret_sample_pred_val.png", dpi=240)
    # fig.show()

    ######### PLOT
    plt.figure(figsize=(10,5))
    plt.grid(True)
    plt.plot(list(range(len(losses))), losses)
    plt.plot(losses,'or',markersize = 3.5)
    plt.title('Epoch Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.savefig("epoch_loss.png", dpi=240)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.grid(True)
    plt.plot(list(range(len(miou))), miou )
    plt.plot(miou,'or',markersize = 3.5)
    plt.title('Validation mIoU')
    plt.xlabel('Step')
    plt.ylabel('mIoU')
    plt.savefig("val_miou.png", dpi=240)
    plt.show()

    #########  TEST
    #take test sample to plot
    if not args.dropout:
        
        _, plt_samples = trainer.validate(loader=test_loader, metrics=val_metrics, plt_samples_bool=True)
        test_score = val_metrics.get_results()

        print(F"Test mIoU = {test_score['Mean IoU']}")
        print(f"Test class mIoU = {test_score['Class IoU']}")

        img = plt_samples[0][0][0,:,:].squeeze()
        target = plt_samples[0][1]
        pred = plt_samples[0][2]

        fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        ax.imshow(img, cmap ='bone')
        ax.imshow(np.ma.masked_where(target == False, target),cmap='coolwarm', alpha=0.6)
        ax.set_title("Actual tumor area over T1 modality")
        plt.savefig("plt_sample_act.png", dpi=240)
        fig.show()
            

        fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        ax.imshow(img, cmap ='bone')
        ax.imshow(np.ma.masked_where(pred == False, pred),cmap='Spectral', alpha=0.6)
        ax.set_title("Predicted tumor area over T1 modality")
        plt.savefig("ret_sample_pred.png", dpi=240)
        fig.show()
    else:        
        all_scores = []
        plt_samples_list = []
        for i in range(100):
            _, plt_samples = trainer.validate(loader=test_loader, metrics=val_metrics, plt_samples_bool = True, test = True)
            all_scores.append(val_metrics.get_results())
            plt_samples_list.append(plt_samples)
        #claculate mean and std
        all_miou = []
        for m in all_scores:
            all_miou.append(m['Mean IoU'])
        mean_score = np.mean(all_miou)
        std_score = np.std(all_miou)
        max_score = np.max(all_miou)
        min_score = np.min(all_miou)
        print(f"mean IoU with MC dropout: {mean_score}")
        print(f"std: {std_score}")
        print(f"max: {max_score}")
        print(f"min: {min_score}")

        
        img = plt_samples[0][0][0,:,:].squeeze()
        # target = plt_samples[0][1]

        sum_target = plt_samples_list[0][0][2]
        for i in range(99):
            sum_target += plt_samples_list[i+1][0][2]

        fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        ax.imshow(img, cmap ='bone')
        ax.imshow(np.ma.masked_where(sum_target == False, sum_target),cmap='Spectral', alpha=0.6)
        ax.set_title("MC Dropuout uncertaintity regions")
        plt.savefig("dropout.png", dpi=240)
        fig.show()
        
        #fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        #ax.imshow(img, cmap ='bone')
        #for i in range(100):
        #    pred = plt_samples_list[i][0][2]
        #    ax.imshow(np.ma.masked_where(pred == False, pred),cmap='Spectral', alpha=0.2)
        #ax.set_title("MC Dropuout uncertaintity regions")
        #plt.savefig("dropout.png", dpi=240)
        #fig.show()


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    main()
