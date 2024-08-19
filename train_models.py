import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from dataset import *
# from pytorchtools import EarlyStopping
from Earlystopping import EarlyStopping
# from srnet import srnet, SRNetEncoder
from models import get_net
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, RandomRotate90,
    RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, CoarseDropout,
    # IAAAdditiveGaussianNoise
    GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
from utils import *
from surgery import *
# from albumentations.augmentations.dropout import Cutout
import wandb
import pickle
import random

parser = argparse.ArgumentParser(description='Train a neural network on image data.')
arg = parser.add_argument
arg('--model', type=str, default='mixnet_s', help='Name of the model to use (e.g., efficientnetB2, mixnet_s)')
arg('--batch_size', type=int, help='Batch size for training and validation')
arg('--checkpoint_dir', type=str, help='Directory to save checkpoints')
arg('--save_file', type=str,default=None, help='Path to the saved model checkpoint')
# arg('--use_mish', action='store_true', help='Use Mish activation (default: False)')
arg('--surgery', type=int, default=1, help='modification level')
patience=10
args = parser.parse_args()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# data_dir = '../input/alaska2-image-steganalysis'
# folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']
# class_names = ['Normal', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 
#                'JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95',
#                 'UERD_75', 'UERD_90', 'UERD_95']
# class_labels = { name: i for i, name in enumerate(class_names)}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "59152"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

def save_checkpoint(model, optimizer, epoch, file_path, train_loss=None, val_loss=None, metrics=None, hyperparams=None):
    """
    A function to save the state of the model
    """
    if torch.distributed.get_rank() != 0:
        return
    file_path = os.path.join(file_path, f"{args.model}_ViT")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, f"epoch_{epoch}.pth")
    # file_path = file_path + f"epoch_{epoch}" + ".pth"

    modelToSave = model.module

    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,  # Could be a dictionary of various metrics
        'hyperparams': hyperparams,  # Hyperparameters used for training
    }
    torch.save(state, file_path)


def get_hyperparams(model, optimizer):
    hyperparams = {
        'model': {param_name: param.data.clone() for param_name, param in model.named_parameters()},
        'optimizer': optimizer.state_dict()
    }
    return hyperparams 

def train_model(model, train_loader, val_loader, batch_size, device, checkpoint_dir):


    model = model.to(device)
    flag_tensor = torch.zeros(1).to(device)
    ddp_rank = torch.distributed.get_rank()
    master_process = ddp_rank == 0
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        weight_decay=1e-2,
        lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=1,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            eps=1e-08
        )
    num_epochs = 50
    train_loss, val_loss = [], []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        rank = torch.distributed.get_rank()
        if rank == 0:
            log_file = open(f"logs/train_epoch{epoch}_logs.txt", "w")
            print('Train Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
        model.train()
        running_loss = 0
        correct = 0 
        total_correct = 0 
        total_samples = 0 
        precisions = []
        targets_cpy = []
        precision = 0
        total_accuracy = 0
        should_stop = False
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        
        with tqdm(train_loader, disable=True, total=int(len(train_loader))) as tk0:
        # with tqdm(train_loader, disable=not master_process, total=int(len(train_loader))) as tk0:
        # with tqdm(train_loader, total=int(len(train_loader))) as tk0:
            log_file.write(f"Training Loop: Epoch {epoch}\n")
            for step, (images, targets) in enumerate(tk0):
                # if step == 10:
                #     # Master process checks the condition, e.g., early stopping
                #     should_stop = True
                # stop_tensor = torch.tensor(should_stop, dtype=torch.int64).to(device)

                targets = targets.to(device).float()
                images = images.to(device).float()
                # batch_size = images.shape[0]

                optimizer.zero_grad()
                # print(targets.shape)
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # Synchronize and broadcast the decision
                # dist.barrier()
                # dist.broadcast(stop_tensor, src=0)
                # _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
                # targets_cpy = targets.clone()
                # # Convert one-hot encoded targets to class indices
                # if targets_cpy.dim() > 1 and targets_cpy.size(1) > 1:  # Check if targets are one-hot encoded
                #     targets_cpy = torch.argmax(targets_cpy, dim=1)
                #
                #
                # if predicted.shape != targets_cpy.shape:
                #     raise ValueError(f"Shape mismatch: Predicted shape {predicted.shape}, Targets shape {targets_cpy.shape}")
                #
                # Apply a threshold to obtain the predicted classes
                predicted = (outputs > 0.5).float().squeeze()

                # Squeeze the targets tensor to remove the extra dimension
                targets_cpy = targets.squeeze()

                if predicted.shape != targets_cpy.shape:
                    raise ValueError(f"Shape mismatch: Predicted shape {predicted.shape}, Targets shape {targets_cpy.shape}")

                correct = (predicted == targets_cpy).sum().item()  # Count correct predictions
                total_correct += correct
                total_samples += targets_cpy.size(0)

                # Calculate precision for the batch
                precision = precision_score(targets_cpy.cpu(), predicted.cpu(), average='binary', zero_division=0)
                precisions.append(precision)

                accuracy = 100 * correct / targets.size(0)  # Batch accuracy
                total_accuracy = 100 * total_correct / total_samples  # Cumulative accuracy
                epoch_precision = sum(precisions) / len(precisions)  # Cumulative precision

                final_scores.update(targets_cpy, outputs.squeeze())
                if dist.is_initialized():
                    final_scores.sync_score(device)

                # final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)
                if dist.get_rank() == 0:
                    log_file.write(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {loss}, Accuracy: {accuracy}, total_accuracy: {total_accuracy}, epoch_precision: {epoch_precision}, lr: {optimizer.param_groups[0]['lr']}\n")
                    tk0.set_postfix(loss=loss.item(), wauc=final_scores.avg, batch_acc=accuracy, total_acc=total_accuracy, lr=optimizer.param_groups[0]['lr'])
                    tk0.update()
                # if stop_tensor.item() == 1:
                #     break
                # dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
                # if (flag_tensor == 1):
                #     break
            # dist.barrier()

        final_scores.sync_score(device)
        local_batch_accuracy = torch.tensor([accuracy]).to(device)
        local_total_accuracy = torch.tensor([total_accuracy]).to(device)
        local_epoch_precision = torch.tensor([epoch_precision]).to(device)

        dist.all_reduce(local_batch_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_epoch_precision, op=dist.ReduceOp.SUM)
        sum_tensor = torch.tensor(summary_loss.sum).to(device)
        count_tensor = torch.tensor(summary_loss.count).to(device)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        # Update summary_loss with global sum and count
        summary_loss.sum = sum_tensor.item()
        summary_loss.count = count_tensor.item()
        summary_loss.avg = summary_loss.sum / summary_loss.count
        if dist.get_rank() == 0:
            log_file.close()
            num_processes = dist.get_world_size()
            average_batch_accuracy = (local_batch_accuracy / num_processes).item()
            average_total_accuracy = (local_total_accuracy / num_processes).item()
            average_epoch_precision = (local_epoch_precision / num_processes).item()
            # print("does this even work")
            # reduced_loss = reduce_tensor(loss.data, num_processes)
            sum_loss = summary_loss.avg
            scores = final_scores.avg
            wandb.log({
                "epoch": epoch,
                "train_summary_loss": sum_loss,
                "train_final_scores": scores,
                "train_batch_accuracy": average_batch_accuracy,
                "train_total_accuracy": average_total_accuracy,
                "train_average_epoch_precision": average_epoch_precision
            })
            print('Training Loss: {:.8f}'.format(summary_loss.avg))
        # wandb.log({"epoch": epoch, "summary_loss": summary_loss, "final_scores": final_scores})
        # print("debug")

        log_file = open(f"logs/Val_epoch{epoch}_logs.txt", "w")
        correct = 0 
        total_correct = 0 
        total_samples = 0 
        precisions = []
        targets_cpy = []
        precision = 0
        total_accuracy = 0
        if dist.get_rank() == 0:
            log_file.write(f"Validation Loop: Epoch {epoch}\n")
            print('Val Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
        # with tqdm(unit='it', total=len(val_loader), disable=not master_process) as pbar:
        with tqdm(unit='it', total=len(val_loader), disable=True) as pbar:
        # with tqdm(unit='it', total=len(val_loader), disable=not master_process) as pbar:
        # # with tqdm(desc='Epoch %d - ' % epoch, unit='it', total=len(val_loader)) as pbar:
            model.eval()
            running_loss = 0
            y, preds = [], []
            summary_loss = AverageMeter()
            final_scores = RocAucMeter()
            with torch.no_grad():
                for step, (images, targets) in enumerate(val_loader):
                    targets = targets.to(device).float()
                    images = images.to(device).float()

                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()
                    # _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
                    # targets_cpy = targets.clone()
                    # # Convert one-hot encoded targets to class indices
                    # if targets_cpy.dim() > 1 and targets_cpy.size(1) > 1:  # Check if targets are one-hot encoded
                    #     targets_cpy = torch.argmax(targets_cpy, dim=1)
                    #
                    #
                    # if predicted.shape != targets_cpy.shape:
                    #     raise ValueError(f"Shape mismatch: Predicted shape {predicted.shape}, Targets shape {targets_cpy.shape}")

                    # Apply a threshold to obtain the predicted classes
                    predicted = (outputs > 0.5).float().squeeze()

                    # Squeeze the targets tensor to remove the extra dimension
                    targets_cpy = targets.squeeze()

                    if predicted.shape != targets_cpy.shape:
                        raise ValueError(f"Shape mismatch: Predicted shape {predicted.shape}, Targets shape {targets_cpy.shape}")

                    correct = (predicted == targets_cpy).sum().item()  # Count correct predictions
                    total_correct += correct
                    total_samples += targets_cpy.size(0)

                    # Calculate precision for the batch
                    precision = precision_score(targets_cpy.cpu(), predicted.cpu(), average='binary', zero_division=0)
                    precisions.append(precision)

                    accuracy = 100 * correct / targets.size(0)  # Batch accuracy
                    total_accuracy = 100 * total_correct / total_samples  # Cumulative Accuracy
                    final_scores.update(targets_cpy, outputs.squeeze())
                    summary_loss.update(loss.detach().item(), batch_size)
                    if dist.is_initialized():
                        final_scores.sync_score(device)

                    if dist.get_rank() == 0:
                        log_file.write(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {loss}, Accuracy: {accuracy}, total_accuracy: {total_accuracy}, epoch_precision: {epoch_precision}, lr: {optimizer.param_groups[0]['lr']}\n")
                        # Apply threshold to determine predictions
                        pbar.set_postfix(loss=(loss.item()), wauc=final_scores.avg)
                        pbar.update()



        # Aggregating
        running_loss /= len(val_loader)
        scheduler.step(running_loss)
        final_scores.sync_score(device)
        local_batch_accuracy = torch.tensor([accuracy]).to(device)
        local_total_accuracy = torch.tensor([total_accuracy]).to(device)
        local_epoch_precision = torch.tensor([epoch_precision]).to(device)
        # local_total_accuracy = torch.tensor([total_accuracy]).to(device)
        # local_epoch_precision = torch.tensor([epoch_precision]).to(device)

        dist.all_reduce(local_batch_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_epoch_precision, op=dist.ReduceOp.SUM)
        sum_tensor = torch.tensor(summary_loss.sum).to(device)
        count_tensor = torch.tensor(summary_loss.count).to(device)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(local_batch_accuracy, op=dist.ReduceOp.SUM)
        # dist.all_reduce(local_total_accuracy, op=dist.ReduceOp.SUM)
        # dist.all_reduce(local_epoch_precision, op=dist.ReduceOp.SUM)
        # dist.all_reduce(torch.tensor(summary_loss.sum).to(device), op=dist.ReduceOp.SUM)
        # dist.all_reduce(torch.tensor(summary_loss.count).to(device), op=dist.ReduceOp.SUM)
        # epoch_precision = sum(precisions) / len(precisions)  # Cumulative precision
        if dist.get_rank() == 0:
            log_file.close()
            num_processes = dist.get_world_size()
            average_batch_accuracy = (local_batch_accuracy / num_processes).item()
            average_total_accuracy = (local_total_accuracy / num_processes).item()
            average_epoch_precision = (local_epoch_precision / num_processes).item()
            # reduced_loss = reduce_tensor(loss.data, num_processes)
            summary_loss.update(loss.detach().item(), batch_size)
            sum_loss = summary_loss.avg
            scores = final_scores.avg
            print(
                f'Val Loss: {summary_loss.avg:.3}, Weighted AUC:{scores:.3}')
            wandb.log({
                "epoch": epoch,
                "val_summary_loss": sum_loss,
                "val_final_scores": scores,
                "val_batch_accuracy": average_batch_accuracy,
                "val_total_accuracy": average_total_accuracy,
                "val_average_epoch_precision": average_epoch_precision
            })
        early_stopping(summary_loss.avg, model)
    
        early_stop = torch.tensor(early_stopping.early_stop, dtype=torch.bool).cuda()
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if rank == 0:
            hyperparams = get_hyperparams(model, optimizer)
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, train_loss, val_loss, metrics=final_scores.avg, hyperparams=hyperparams)

        #torch.save(model.state_dict(),
        #           f"epoch_{epoch}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth")

def main(rank, world_size):

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    check_unused_parameters = False

    #model = get_net(args.model, args.use_mish).to(device)
    model = get_net(args.model, args.surgery)
    if "mixnet" in args.model:
        check_unused_parameters = False
    else:
        check_unused_parameters = True

    # Create checkpoint directory if it does not exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
            
    
    # seed_everything(args.random_seed)
    QFs = ['75','90', '95']
    Classes = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    IL_train = []
    IL_val = []
    dataset = []
    
    for QF in QFs:
        with open('splits/IL_train_'+QF+'.p', 'rb') as handle:
            IL_train.extend(pickle.load(handle))
        with open('splits/IL_val_'+QF+'.p', 'rb') as handle:
            IL_val.extend(pickle.load(handle))

    # for label, kind in enumerate(Classes):
    for kind in Classes: 
        label = 1 if kind != 'Cover' else 0
        for path in IL_train:
            dataset.append({
                'kind': kind,
                'image_name': path,
                'label': label,
                'fold':1,
            })
    for kind in Classes: 
        label = 1 if kind != 'Cover' else 0
        for path in IL_val:
            dataset.append({
                'kind': kind,
                'image_name': path,
                'label': label,
                'fold':0,
            })
            
    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)

    # train_df = pd.read_csv('/workspace/Steg-Research/Datasets/training_data.csv')
    # val_df = pd.read_csv('/workspace/Steg-Research/Datasets/testing_data.csv')
    model = model.to(device)
    model = DDP(model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False)

    # if args.save_file is Not None:
    #     chkpt = torch.load(args.save_file, map_location=device)
    #     model_state = chkpt['model_state']

    train_dataset = TrainRetriever(
        kinds=dataset[dataset['fold'] != 0].kind.values,
        image_names=dataset[dataset['fold'] != 0].image_name.values,
        labels=dataset[dataset['fold'] != 0].label.values,
        transforms=get_train_transforms()
    )

    valid_dataset = TrainRetriever(
        kinds=dataset[dataset['fold'] == 0].kind.values,
        image_names=dataset[dataset['fold'] == 0].image_name.values,
        labels=dataset[dataset['fold'] == 0].label.values,
        transforms=get_valid_transforms()
    )
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     valid_dataset, num_replicas=world_size, rank=rank
    # )
    # val_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               shuffle=False)
                                               # shuffle=False, sampler=val_sampler)

    if rank == 0:
        wandb.init(project=args.model+'ViT_log', group=f"{args.model}_{args.batch_size}_group", config=args)
    train_model(model, train_loader, val_loader, args.batch_size, device, args.checkpoint_dir)

if __name__ == "__main__":
    world_size = 1
    
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )
