####################################################
# Adapted from https://github.com/Rudrabha/Wav2Lip #
####################################################
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.model import SyncTransformer
from sklearn.metrics import f1_score
import torch
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from glob import glob
import os, random, cv2, argparse
from hparams_gtx3070 import hparams
import pdb
import sys
import soundfile as sf
import torch.multiprocessing
from sklearn.metrics import f1_score
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import load_checkpoint, force_cudnn_initialization, set_parameters
from custom_datasets import Dataset as MyDataset

"""
training based on sync error
E.L. Benaroya 2023 - IRCAM
"""


parser = argparse.ArgumentParser(description='Code to train/test the expert lip-sync discriminator on Macron DB')
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

# little trick avoid RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED using pytorch
# solution,
# https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
force_cudnn_initialization()

v_context = 5
BATCH_SIZE = 1
n_classes = 8
n_max = n_classes*4
n_samples = 25

def trainit(device, model, train_data_loader, val_data_loader, optimizer, checkpoint_dir, checkpoint_interval, nepochs):

    batch_size = 20  # arbitrary ?
    audio_fps = 16000/hparams.hop_size  # float, 80.
    video_fps = hparams.fps  # 25.
    mel_step_size = int(v_context / video_fps * audio_fps)-1  # 16
    # samplewise_acc_k5 = []
    loss_fct = torch.nn.CrossEntropyLoss()
    model.train()

    # n_step_percent = 0.4
    # n_valid_percent = 1.
    # total_step = len(train_data_loader)
    # n_step = int(total_step*n_step_percent)
    # n_valid_step = int(total_step*n_valid_percent)

    onehot_target0 = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=n_classes).float().squeeze(0).to(device)
    onehot_target1 = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=n_classes).float().squeeze(0).to(device)
    onehot_target2 = torch.nn.functional.one_hot(torch.tensor([2]), num_classes=n_classes).float().squeeze(0).to(device)

    n_epochs = 10
    for epoch in range(n_epochs):
        print('] EPOCH %d [' % epoch)
        total_loss = 0.0
        total_corrects = 0
        total_sum = 0
        preds_list = []
        for step, (vid, aud, lastframe) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

            optimizer.zero_grad()

            lastframe = lastframe.item()
            vid = vid.view(1, lastframe, 3, 48, 96)
            # print('vid size ', vid.size())  # torch.Size([1, 325, 3, 48, 96])
            # print('aud size ', aud.size())  # torch.Size([1, 1074, 1, 80])
            delta = 10
            # take context along dimension 1 and batch the windows
            aud_batch = [aud[:, i: i + int(n_samples*audio_fps/video_fps), :, :]
                         for i in [int(item+delta) for item in np.linspace(0, n_max, n_classes)]]
            audio_batch = torch.cat(aud_batch, 0).permute(0, 2, 3, 1).to(device)
            B = audio_batch.size(0)
            #
            img_batch = vid[:, delta:delta+n_samples, :, :, :].view(1, -1, 48, 96).repeat(B, 1, 1, 1).to(device)

            # print()
            # print(f'img_batch size {img_batch.size()}')
            # print(f'audio_batch size {audio_batch.size()}')
            # print('calc loss')
            raw_sync_scores = model(img_batch, audio_batch)
            # print('raw_sync_scores.size() ', raw_sync_scores.size())
            # print(torch.nn.functional.softmax(raw_sync_scores, 0))
            loss0 = loss_fct(raw_sync_scores, onehot_target0)
            loss1 = loss_fct(raw_sync_scores, onehot_target1)
            loss2 = loss_fct(raw_sync_scores, onehot_target2)

            loss = loss0 + 0.95*loss1 + 0.93*loss2
            # print('loss = ', loss)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(raw_sync_scores).item()
            # print(torch.nn.functional.softmax(raw_sync_scores))
            # print(preds.item())
            total_corrects += np.sum(preds==0)+np.sum(preds==1)+np.sum(preds==2)
            total_sum += 1
            preds_list.append(preds)
            # Add these lines to obtain f1_score

        # if step % n_step == 0:    # print
        preds_np = np.array(preds_list)
        f1_score_out = f1_score(np.zeros_like(preds_np), preds_np, average='micro')
        # print(f1_score_out)
        print(f'[TRAIN], loss: {total_loss / total_sum:.3f}, acc: {total_corrects/ total_sum*100.:.1f}')
        # , f1 score: {f1_score_out*100.:.1f}')

        # if step % n_valid_step == 0:
        valid(device, model, val_data_loader)

def valid(device, model, val_data_loader):

    audio_fps = 16000/hparams.hop_size  # float, 80.
    video_fps = hparams.fps  # 25.
    mel_step_size = int(v_context / video_fps * audio_fps)-1  # 16
    loss_fct = torch.nn.CrossEntropyLoss()
    model.eval()

    onehot_target0 = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=n_classes).float().squeeze(0).to(device)
    onehot_target1 = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=n_classes).float().squeeze(0).to(device)
    onehot_target2 = torch.nn.functional.one_hot(torch.tensor([2]), num_classes=n_classes).float().squeeze(0).to(device)

    total_loss = 0.0
    preds_list = []
    total_corrects = 0
    total_elts = 0
    # n_max_step = len(val_data_loader)//4-1
    for step, (vid, aud, lastframe) in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
        with torch.no_grad():

            # print('vid size ', vid.size())  # torch.Size([1, 325, 3, 48, 96])
            # print('aud size ', aud.size())  # torch.Size([1, 1074, 1, 80])

            lastframe = lastframe.item()
            vid = vid.view(1, -1, 3, 48, 96)

            # take context along dimension 1 and batch the windows
            delta = 10
            # take context along dimension 1 and batch the windows
            aud_batch = [aud[:, i: i + int(n_samples*audio_fps/video_fps), :, :]
                         for i in [int(item+delta) for item in np.linspace(0, n_max, n_classes)]]
            audio_batch = torch.cat(aud_batch, 0).permute(0, 2, 3, 1).to(device)
            B = audio_batch.size(0)
            #
            img_batch = vid[:, delta:delta+n_samples, :, :, :].view(1, -1, 48, 96).repeat(B, 1, 1, 1).to(device)

            # im_batch = [vid[:, i: i + n_samples, :, :, :].view(1, -1, 48, 96)
            #             for i in [int(item+delta) for item in np.linspace(0, n_max, n_classes)]]
            # img_batch = torch.cat(im_batch, 0).to(device)
            # B = img_batch.size(0)
            # audio_batch = aud[:, :int(n_samples*audio_fps/video_fps), :, :].repeat(B, 1, 1, 1).permute(0, 2, 3, 1).to(device)

            raw_sync_scores = model(img_batch, audio_batch)
            loss0 = loss_fct(raw_sync_scores, onehot_target0)
            loss1 = loss_fct(raw_sync_scores, onehot_target1)
            loss2 = loss_fct(raw_sync_scores, onehot_target2)

            loss = loss0 + 0.95*loss1 + 0.93*loss2

        preds = torch.argmax(raw_sync_scores).item()
        preds_list.append(preds)

        total_corrects += np.sum(preds == 0) + np.sum(preds == 1) + np.sum(preds == 2)
        total_elts += 1
        total_loss += loss.item()

        # if step == n_max_step:
        #     break

    preds_np = np.array(preds_list)
    f1_score_out = f1_score(np.zeros_like(preds_np), preds_np, average='micro')

    print(f'[VALIDATION], loss: {total_loss / total_elts:.3f},'
          f' acc: {total_corrects/ total_elts*100.:.1f}')  # , f1 score: {f1_score_out*100.:.1f}')


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    # Dataset and Dataloader setup
    print('Dataset and Dataloader setup')
    device = torch.device("cuda" if use_cuda else "cpu")
    train_dataset = MyDataset('test', overwrite=True, device='cpu')  # use test set !!!!!!
    val_dataset = MyDataset('val', overwrite=True, device='cpu')

    print('Data loader setup')
    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=1,  num_workers=2, shuffle=True)
    val_data_loader = data_utils.DataLoader(val_dataset, batch_size=1, num_workers=2)

    # Model
    print('build model')
    model = SyncTransformer().to(device)
    params = set_parameters(model)
    optimizer = optim.Adam(params, lr=1e-4)

    if checkpoint_path is not None:
        print('load checkpoint')
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=True, use_cuda=use_cuda)
    else:
        global_step = 0
        global_epoch = 0

    trainit(device, model, train_data_loader, val_data_loader, optimizer, checkpoint_dir=checkpoint_dir,
          checkpoint_interval=10, nepochs=650)
