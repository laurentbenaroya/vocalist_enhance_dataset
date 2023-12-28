####################################################
# Adapted from https://github.com/Rudrabha/Wav2Lip #
####################################################
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.model import SyncTransformer
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchaudio.transforms import MelScale
from glob import glob
import os, random, cv2, argparse
from hparams_gtx3070 import hparams, get_image_list
import sys
from natsort import natsorted
import soundfile as sf
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import load_checkpoint, calc_pdist, force_cudnn_initialization

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

# RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED using pytorch
# solution,
# https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
force_cudnn_initialization()

num_audio_elements = 3200  # 6400  # 16000/25 * syncnet_T
tot_num_frames = 25  # buffer
v_context = 5
BATCH_SIZE = 1
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
# melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, f_max=hparams.fmax,
#                     n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney').to(0)
logloss = nn.BCEWithLogitsLoss()

import pdb
def makeit(device, model, test_data_loader, optimizer, checkpoint_dir, checkpoint_interval, nepochs):

    batch_size = 20  # arbitrary ?
    audio_fps = 16000/hparams.hop_size  # float, 80.
    video_fps = hparams.fps  # 25.
    mel_step_size = int(v_context / video_fps * audio_fps)-1  # 16
    # samplewise_acc_k5 = []
    prog_bar = enumerate(test_data_loader)
    for step, (vid, aud, lastframe) in prog_bar:
        model.eval()
        with torch.no_grad():
            lastframe = lastframe.item()
            vid = vid.view(1, lastframe, 3, 48, 96)
            print('vid size ', vid.size())  # torch.Size([1, 325, 3, 48, 96])
            print('aud size ', aud.size())  # torch.Size([1, 1074, 1, 80])

            # take context along dimension 1 and batch the windows
            im_batch = [vid[:, i: i + lastframe//3, :, :, :].view(1, -1, 48, 96)
                        for i in range(0, 100, 5)]
            img_batch = torch.cat(im_batch, 0).to(device)
            B = img_batch.size(0)
            audio_batch = aud[:, :int(lastframe//3*audio_fps/video_fps), :, :].repeat(B, 1, 1, 1).permute(0, 2, 3, 1).to(device)

            print()
            print(f'img_batch size {img_batch.size()}')
            print(f'audio_batch size {audio_batch.size()}')
            print('calc pdist')
            raw_sync_scores = model(img_batch, audio_batch)
            print(raw_sync_scores.size())
            print(torch.nn.functional.softmax(raw_sync_scores, 0))

def get_audio_duration_with_soundfile(audio_file_path):
    with sf.SoundFile(audio_file_path, 'r') as audio_file:
        # Get the number of frames in the audio file
        num_frames = len(audio_file)

        # Get the frame rate of the audio file
        frame_rate = audio_file.samplerate

        # Calculate the duration in seconds
        duration = float(num_frames) / frame_rate

        return duration


class DatasetTest(object):
    """
    dataset class used for testing, which we will use also for training, on a limited number of videos
    """
    def __init__(self, split='test', overwrite=False, device=None):

        self.split = split
        self.all_videos = get_image_list('', split)
        # self.all_videos = [item.strip() for item in self.all_videos]
        self.datalen = len(self.all_videos)

        # precompute mels
        print('precompute mels')  # not really helpful
        for idx in tqdm(range(self.datalen)):
            vidname = self.all_videos[idx]  # dirname(self.all_videos[idx])
            melpath = join(vidname, "mel.pt")
            wavpath = join(vidname, "audio.wav")
            if not os.path.isfile(melpath) or overwrite:
                melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin,
                                    f_max=hparams.fmax,
                                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney').to(device)
                wav = self.get_wav(wavpath)
                aud_tensor = torch.FloatTensor(wav).to(device)
                spec = torch.stft(aud_tensor, n_fft=hparams.n_fft, hop_length=hparams.hop_size,
                                  win_length=hparams.win_size,
                                  window=torch.hann_window(hparams.win_size), return_complex=True)
                melspec = melscale(torch.abs(spec.detach().clone()).float())
                melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
                # NORMALIZED MEL
                normalized_mel = torch.clip(
                    (2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                    -hparams.max_abs_value, hparams.max_abs_value)
                mels = normalized_mel.unsqueeze(0).permute(2, 0, 1).cpu()
                torch.save(mels, melpath)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath):
        return sf.read(wavpath)[0]

    def get_window(self, img_name, len_img_names):
        """
        get all frames after (inclusive) img_name id
        :param img_name:
        :param len_img_names:
        :return: frame list
        """
        start_frame_id = self.get_frame_id(img_name)
        vidname = dirname(img_name)

        window_fnames = []
        for frame_id in range(start_frame_id, len_img_names):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                print(f'{vidname, "{}.jpg".format(frame_id)} is not a file')
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        return 1  # len(self.all_videos)  # *10

    def __getitem__(self, idx):

        # idx = 0
        # idx = idx%len(self.all_videos)
        vidname = self.all_videos[idx]
        print(vidname)
        wavpath = join(vidname, "audio.wav")
        melpath = join(vidname, "mel.pt")
        mels = torch.load(melpath)
        img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())

        # all this to try avoiding the -1
        # get img_names a list len multiple of fps (25)
        # the problem came from '1.jpg' instead of str(np.min(img_nums)) + '.jpg' (0.jpg here)
        len_img_names = len(img_names) - len(img_names) % 25
        img_names = img_names[:len_img_names]
        img_nums = [int(basename(item)[:-4]) for item in img_names]

        # take floor of sound duration in sec
        num_video_frames = min(len(img_names), np.floor(get_audio_duration_with_soundfile(wavpath)) * 25)
        lastframe = num_video_frames  # -1

        img_name = os.path.join(vidname, str(np.min(img_nums)) + '.jpg')
        window_fnames = self.get_window(img_name, len(img_names))
        assert window_fnames is not None

        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                all_read = False
                break

            window.append(img)

        assert all_read, 'not all frames where read...'

        # H, W, T, 3 --> T*3
        vid = np.concatenate(window, axis=2) / 255.
        vid = vid.transpose(2, 0, 1)
        # take the lower part of the image NOT EXPLICITLY THE LIPS (48 = img_size/2)
        vid = torch.FloatTensor(vid[:, 48:])

        assert not torch.any(torch.isnan(vid))
        assert not torch.any(torch.isnan(mels))

        assert vid is not None
        assert mels is not None

        return vid, mels, lastframe


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    # Dataset and Dataloader setup
    print('Dataset and Dataloader setup')
    device = torch.device("cuda" if use_cuda else "cpu")
    test_dataset = DatasetTest('test', overwrite=True, device='cpu')

    print('Data loader setup')
    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=1,  num_workers=1)

    print('dry test')
    for batch in test_data_loader:
        vid, mels, lastframe = batch
        lastframe = lastframe.item()  # single elt in batch
        print('lastframe ', lastframe)

    # Model
    print('build model')
    model = SyncTransformer().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=5e-5)

    if checkpoint_path is not None:
        print('load checkpoint')
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False, use_cuda=use_cuda)
    else:
        global_step = 0
        global_epoch = 0

    makeit(device, model, test_data_loader, optimizer, checkpoint_dir=checkpoint_dir,
          checkpoint_interval=10, nepochs=650)
