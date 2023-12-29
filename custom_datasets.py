import os
from tqdm import tqdm
from glob import glob
from os.path import dirname, basename, join, isfile
import torch
import numpy as np
import soundfile as sf
from natsort import natsorted
from hparams_gtx3070 import hparams, get_image_list
from torchaudio.transforms import MelScale
import cv2
from utils import get_audio_duration_with_soundfile


TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))


class Dataset(object):
    """
    dataset class used for testing, which we will use also for training, on a limited number of videos
    """
    def __init__(self, split='test', overwrite=False, device='cpu'):

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
        return len(self.all_videos)  # *10

    def __getitem__(self, idx):

        # idx = 0
        # idx = idx%len(self.all_videos)
        vidname = self.all_videos[idx]
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


def collate_fn(batch):
    a, b, c = zip(*batch)
    return a, b, c
