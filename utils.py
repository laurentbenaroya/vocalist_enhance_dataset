import torch
from tqdm import tqdm
import soundfile as sf

def _load(checkpoint_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, use_cuda=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return


def set_parameters(model):
    # set trainable parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mem_transformer.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    print([name for name, p in model.named_parameters() if p.requires_grad])
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return params
import pdb

def calc_pdist(model, img, mels, vshift=15, device='cpu'):
    win_size = vshift * 2 + 1

    melspad = torch.nn.functional.pad(mels.permute(1, 2, 3, 0).contiguous(), (vshift, vshift)).permute(3, 0, 1, 2).contiguous()

    dists = []
    label_est = []
    num_rows_dist = len(img)
    for i in tqdm(range(0, num_rows_dist-5-win_size, win_size)):
        distsdelta = []
        melswindow = melspad[i:i+win_size].to(device)
        print('melspad size ', melspad.size())
        print('melswindow size ', melswindow.size())
        for delta in range(5):
            imgwindow_delta = img[delta+i:(delta+i+win_size)].to(device)  # .unsqueeze(0).repeat(win_size, 1, 1, 1).to(device)
            print('img size ', img.size())
            print('imgwindow_delta size ', imgwindow_delta.size())
            # pdb.set_trace()
            # print('going through model...')
            # try:
            raw_sync_scores = model(imgwindow_delta, melswindow)
            # except Exception as e:
            #     print(e)
            #     pdb.set_trace()
            dist_measures = raw_sync_scores.clone().cpu()
            # if i in range(vshift):
            #     dist_measures[0:vshift-i] = torch.tensor(-1000, dtype=torch.float).to(device)
            # elif i in range(num_rows_dist-vshift, num_rows_dist):
            #     dist_measures[vshift+num_rows_dist-i:] = torch.tensor(-1000, dtype=torch.float).to(device)
            distsdelta.append(dist_measures)

        dists.append(distsdelta)

    return dists


def get_audio_duration_with_soundfile(audio_file_path):
    with sf.SoundFile(audio_file_path, 'r') as audio_file:
        # Get the number of frames in the audio file
        num_frames = len(audio_file)

        # Get the frame rate of the audio file
        frame_rate = audio_file.samplerate

        # Calculate the duration in seconds
        duration = float(num_frames) / frame_rate

        return duration

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
