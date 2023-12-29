####################################################
# Adapted from https://github.com/Rudrabha/Wav2Lip #
####################################################
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.model_multiclass import SyncTransformer
from sklearn.metrics import f1_score
import torch
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from random import shuffle
from glob import glob
import os, random, cv2, argparse
from hparams_gtx3070 import hparams
import pdb
import sys
import soundfile as sf
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import load_checkpoint, force_cudnn_initialization, set_parameters_multiclass
from custom_datasets import Dataset as MyDataset, collate_fn

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

# parameters
BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
# optim
learning_rate = 1e-4  # THIS MIGHT BE AN ISSUE
weight_decay = 0.1
# windowing
n_classes = 6
# test location at audio rate (80 fps)
# audio_location = sorted([0, 3, 5, 10, 20, 30, 50, 100])  # custom
audio_location = np.linspace(0, 140, n_classes)
loss_weights = [1.0, 0.8, 0.5]
loss_weights = [w / sum(loss_weights) for w in loss_weights]  # normalize
n_locations = len(loss_weights)

assert len(audio_location) == n_classes

n_samples_img = 50  # nombre d'images dans les extraits

audio_fps = 16000 / hparams.hop_size  # float, 80.
video_fps = hparams.fps  # 25.

n_samples_audio = int(np.floor(n_samples_img * audio_fps / video_fps))  # nombre de trames pour le melspec

loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')


def trainit(device, model, train_data_loader, val_data_loader, optimizer, 
            checkpoint_dir, checkpoint_interval, n_epochs=10):

    # initial score !!!
    print('INITIAL SCORE')
    valid(device, model, val_data_loader, loss_weights, delta_audio=None)
    print()

    for epoch in range(n_epochs):
        model.train()
        print(':) EPOCH %d (:' % epoch)
        total_loss = 0.0
        total_corrects = 0
        total_sum = 0
        preds_list = []
        audio_location_k = audio_location.copy()
        for step, (vid_batch, aud_batch, lastframe_batch) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

            optimizer.zero_grad()
            audio_batch = []
            img_batch = []
            audio_location_idx_list = []

            for vid, aud, lastframe in zip(vid_batch, aud_batch, lastframe_batch):
                # print('vid size ', vid.size())  # torch.Size([1, 325, 3, 48, 96])
                # print('aud size ', aud.size())  # torch.Size([1, 1074, 1, 80])

                # take context along dimension 1 and batch the window
                # put data to gpu
                vid = vid.view(1, -1, 3, 48, 96).to(device)
                aud = aud.unsqueeze(0).to(device)  # cast to size 1 batch
                # do windowing
                # 1) audio

                shuffle(audio_location_k)
                sorted_location_indices = sorted(range(n_classes), key=lambda k: audio_location_k[k])[:n_locations]
                audio_location_idx_list.append(sorted_location_indices)
                aud_bat = [aud[:, int(i): int(i) + n_samples_audio, :, :]
                           for i in audio_location_k]
                audio_bat = torch.cat(aud_bat, 0)  # n_classes x n_samples_audio x 1 x 80 (mel)
                audio_batch.append(audio_bat)

                # 2) images
                # B = n_classes  # audio_bat.size(0)
                img_bat = vid[:, 0: n_samples_img, :, :, :]  # 1 x n_samples_img x 3 x 48 x 96 (img crop)
                img_bat = img_bat.view(1, -1, 48, 96)  # fuse n_samples_img and RGB channels
                img_bat = img_bat.repeat(n_classes, 1, 1, 1)  # repeat on dim 0, n_classes x n_samples_img*3, x 48 x 96
                img_batch.append(img_bat)

                # print(f'img_bat size {img_bat.size()}')
                # print(f'audio_bat size {audio_bat.size()}')
            audio_location_idx_batch = torch.tensor(audio_location_idx_list).to(device)
            # print('audio_location_idx_batch size ', audio_location_idx_batch.size())
            # final data, fuse on first dimension
            audio_batch = torch.stack(audio_batch, 0)  # create a new dimension with batch_size
            img_batch = torch.stack(img_batch, 0)  # create a new dimension with batch_size
            # group BATCH_SIZE with n_classes (dim 0 and 1)
            img_batch = img_batch.view(BATCH_SIZE, n_classes*n_samples_img*3, 48, 96)
            audio_batch = audio_batch.view(BATCH_SIZE, n_classes*n_samples_audio, 1, 80)   #
            # IL PEUT Y AVOIR UN PROBLEME ICI AUSSI
            audio_batch = audio_batch.permute(0, 2, 1, 3)  # permute(0, 2, 3, 1) [wiered]
            # print('audio_batch.size() ', audio_batch.size())

            # go through model
            raw_sync_scores = model(img_batch, audio_batch)  # BATCH_SIZE x n_classes
            # probabilities = torch.nn.functional.softmax(raw_sync_scores.view(BATCH_SIZE, n_classes), 1)
            # entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            # loss
            loss = torch.tensor([0.], device=device)
            for v, w in enumerate(loss_weights):
                loss += w*loss_fct(raw_sync_scores, audio_location_idx_batch[:, v])
            # print('loss = ', loss)
            # loss += 10*entropy

            # gradient descent
            loss.backward()
            optimizer.step()

            # indicators
            total_loss += loss.item()
            preds = torch.argmax(raw_sync_scores, 1).clone().cpu().numpy()
            # print(torch.nn.functional.softmax(raw_sync_scores))

            # print('preds.shape ', preds.shape)
            for v, w in enumerate(loss_weights):
                total_corrects += w*np.sum(preds == audio_location_idx_batch[:, v].clone().cpu().numpy())
            total_sum += BATCH_SIZE
            preds_list.append(preds)

        # Add these lines to obtain f1_score
        # preds_np = np.array(preds_list)
        # print(preds_np.shape)
        # preds_np = preds_np.flatten()
        # print(preds_np.shape)
        # f1_score_out = f1_score(np.zeros_like(preds_np), preds_np, average='micro')
        # print(f1_score_out)

        # I'm not sure about the multiplication by BATCH_SIZE
        print(f'[TRAIN], loss: {total_loss / total_sum*BATCH_SIZE:.3f}, acc: {total_corrects/ total_sum*100.:.1f}')
        # , f1 score: {f1_score_out*100.:.1f}')

        valid(device, model, val_data_loader, loss_weights, delta_audio=None)
        print()


def valid(device, model, val_data_loader, loss_weights, delta_audio):

    model.eval()

    total_loss = 0.0
    preds_list = []
    total_corrects = 0
    total_elts = 0
    audio_location_k = audio_location.copy()
    for step, (vid_batch, aud_batch, lastframe_batch) in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
        audio_batch = []
        img_batch = []
        audio_location_idx_list = []
        with torch.no_grad():
            for vid, aud, lastframe in zip(vid_batch, aud_batch, lastframe_batch):
                # print('vid size ', vid.size())  # torch.Size([1, 325, 3, 48, 96])
                # print('aud size ', aud.size())  # torch.Size([1, 1074, 1, 80])

                # take context along dimension 1 and batch the window
                # put data to gpu
                vid = vid.view(1, -1, 3, 48, 96).to(device)
                aud = aud.unsqueeze(0).to(device)  # cast to size 1 batch
                # do windowing
                # 1) audio
                shuffle(audio_location_k)
                sorted_location_indices = sorted(range(n_classes), key=lambda k: audio_location_k[k])[:n_locations]
                audio_location_idx_list.append(sorted_location_indices)
                aud_bat = [aud[:, int(i): int(i) + n_samples_audio, :, :]
                           for i in audio_location_k]

                audio_bat = torch.cat(aud_bat, 0)  # n_classes x n_samples_audio x 1 x 80 (mel)
                audio_batch.append(audio_bat)

                # 2) images
                # B = n_classes  # audio_bat.size(0)
                img_bat = vid[:, 0: n_samples_img, :, :, :]  # 1 x n_samples_img x 3 x 48 x 96 (img crop)
                img_bat = img_bat.view(1, -1, 48, 96)  # fuse n_samples_img and RGB channels
                img_bat = img_bat.repeat(n_classes, 1, 1, 1)  # repeat on dim 0, n_classes x n_samples_img*3, x 48 x 96
                img_batch.append(img_bat)

                # print(f'img_bat size {img_bat.size()}')
                # print(f'audio_bat size {audio_bat.size()}')
                # final data, fuse on first dimension
            audio_location_idx_batch = torch.tensor(audio_location_idx_list).to(device)
            # pdb.set_trace()
            # print('audio_location_idx_batch size ', audio_location_idx_batch.size())
            audio_batch = torch.stack(audio_batch, 0)  # create a new dimension with batch_size
            img_batch = torch.stack(img_batch, 0)  # create a new dimension with batch_size
            # group BATCH_SIZE with n_classes (dim 0 and 1)
            img_batch = img_batch.view(TEST_BATCH_SIZE, n_classes*n_samples_img * 3, 48, 96)
            audio_batch = audio_batch.view(TEST_BATCH_SIZE, n_classes*n_samples_audio, 1, 80)  #
            # IL PEUT Y AVOIR UN PROBLEME ICI AUSSI
            audio_batch = audio_batch.permute(0, 2, 1, 3)  # .permute(0, 2, 3, 1) -> the weired one
            raw_sync_scores = model(img_batch, audio_batch)  # TEST_BATCH_SIZE x n_classes

            loss = torch.tensor([0.], device=device)
            for v, w in enumerate(loss_weights):
                loss += w*loss_fct(raw_sync_scores, audio_location_idx_batch[:, v])

        preds = torch.argmax(raw_sync_scores, 1).clone().cpu().numpy()
        preds_list.append(preds)

        for v, w in enumerate(loss_weights):
            total_corrects += w*np.sum(preds == audio_location_idx_batch[:, v].clone().cpu().numpy())
        total_elts += TEST_BATCH_SIZE
        total_loss += loss.item()

    # preds_np = np.array(preds_list)
    # f1_score_out = f1_score(np.zeros_like(preds_np), preds_np, average='micro')

    print(torch.nn.functional.softmax(raw_sync_scores, 1))
    # I'm not sure about the multiplication by TEST_BATCH_SIZE
    print(f'[VALIDATION], loss: {total_loss / total_elts*TEST_BATCH_SIZE:.3f},'
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
    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE,  num_workers=4, shuffle=True,
                                              collate_fn=collate_fn, drop_last=True)
    val_data_loader = data_utils.DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, num_workers=2, shuffle=False,
                                            collate_fn=collate_fn, drop_last=True)

    # Model
    print('build model')
    model = SyncTransformer(n_classes).to(device)
    if checkpoint_path is not None:
        print('load checkpoint')
        load_checkpoint(checkpoint_path, model, None, reset_optimizer=True, use_cuda=use_cuda)
    else:
        global_step = 0
        global_epoch = 0
    params = set_parameters_multiclass(model, n_classes, device)
    # optimizer = optim.Adam(params, lr=learning_rate)
    # AdamW !!!
    optimizer = optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay)

    trainit(device, model, train_data_loader, val_data_loader, optimizer, checkpoint_dir=checkpoint_dir,
            checkpoint_interval=-1, n_epochs=5)
