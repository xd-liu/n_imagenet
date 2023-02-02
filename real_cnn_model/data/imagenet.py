import torch
from torch_scatter import scatter_max, scatter_min
from torch.utils.data import Dataset
import torch.nn as nn
from pathlib import Path
import numpy as np
import random
import torch.nn.functional as F
from os.path import join, dirname, isfile

SENSOR_H = 180
SENSOR_W = 240
IMAGE_H = 224
IMAGE_W = 224
EXP_TAU = 0.3
TIME_SCALE = 1000000

CLIP_COUNT = False
CLIP_COUNT_RATE = 0.99
DISC_ALPHA = 3.0

# ValueLayer for EST


class ValueLayer(nn.Module):

    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[
            -1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[
            0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init",
                    "trilinear_init.pth")
        if isfile(path):
            # print("loading ValueLayer Successfully!")
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            print("loading ValueLayer Failed!")
            self.init_kernel(num_channels)

        self.mlp = self.mlp.float()

    def forward(self, x):
        x = x.float()
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]
        # print("VL test 1")
        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))
        # print("VL test 2")

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


# Parsing Modules


def load_event(event_path, cfg):
    # Returns time-shifted numpy array event from event_path
    event = np.load(event_path, allow_pickle=True)
    if getattr(cfg, 'compressed', True):
        event = event['arr_0'].item()['event_data']
        event = np.vstack([event['x'], event['y'], event['t'],
                           event['p'] + 2]).T
    else:
        event = np.vstack([
            event['x_pos'], event['y_pos'], event['timestamp'],
            event['polarity'].astype(np.uint8)
        ]).T

    event = event.astype(np.float)

    # Account for int-type timestamp
    event[:, 2] /= TIME_SCALE

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    # print("Load Event:", event[:, 0].max(), event[:, 0].min(), \
    #         event[:, 1].max(), event[:, 1].min(), \
    #         event[:, 3].max(), event[:, 3].min())

    return event


def slice_event(event, cfg):
    slice_method = getattr(cfg, 'slice_method', 'idx')
    if slice_method == 'idx':
        start = getattr(cfg, 'slice_start', None)
        end = getattr(cfg, 'slice_end', None)
        event = event[start:end]
    elif slice_method == 'time':
        start = getattr(cfg, 'slice_start', None)
        end = getattr(cfg, 'slice_end', None)
        event = event[(event[:, 2] > start) & (event[:, 2] < end)]
    elif slice_method == 'random':
        length = getattr(cfg, 'slice_length', None)
        slice_augment = getattr(cfg, 'slice_augment', False)

        if slice_augment and cfg.mode == 'train':
            slice_augment_width = getattr(cfg, 'slice_augment_width', 0)
            length = random.randint(length - slice_augment_width,
                                    length + slice_augment_width)

        if len(event) > length:
            start = random.choice(range(len(event) - length + 1))
            event = event[start:start + length]

    return event


def reshape_event_with_sample(event, orig_h, orig_w, new_h, new_w):
    # Sample events
    sampling_ratio = (new_h * new_w) / (orig_h * orig_w)

    new_size = int(sampling_ratio * len(event))
    idx_arr = np.arange(len(event))

    sampled_arr = np.random.choice(idx_arr, size=new_size, replace=False)
    sampled_event = event[np.sort(sampled_arr)]

    # Rescale coordinates
    sampled_event[:, 0] *= (new_w / orig_w)
    sampled_event[:, 1] *= (new_h / orig_h)

    return sampled_event


def reshape_event_no_sample(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    return event


def reshape_event_unique(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    coords = event[:, :2].astype(np.int64)
    timestamp = (event[:, 2] * TIME_SCALE).astype(np.int64)
    min_time = timestamp[0]
    timestamp -= min_time

    key = coords[:, 0] + coords[:, 1] * new_w + timestamp * new_h * new_w
    _, unique_idx = np.unique(key, return_index=True)

    event = event[unique_idx]

    return event


def parse_event(event_path, cfg):
    event = load_event(event_path, cfg)

    event = torch.from_numpy(event)

    # Account for density-based denoising
    denoise_events = getattr(cfg, 'denoise_events', False)
    denoise_bins = getattr(cfg, 'denoise_bins', 10)
    denoise_timeslice = getattr(cfg, 'denoise_timeslice', 5000)
    denoise_patch = getattr(cfg, 'denoise_patch', 2)
    denoise_thres = getattr(cfg, 'denoise_thres', 0.5)
    denoise_density = getattr(cfg, 'denoise_density', False)
    denoise_hot = getattr(cfg, 'denoise_hot', False)
    denoise_time = getattr(cfg, 'denoise_time', False)
    denoise_neglect_polarity = getattr(cfg, 'denoise_neglect_polarity', True)

    reshape = getattr(cfg, 'reshape', False)
    # print("before reshape x", event[:,0].max(), event[:,0].min())
    # print("before reshape y", event[:,1].max(), event[:,1].min())
    # print("before reshape t", event[:,3].max(), event[:,3].min())
    if reshape:
        reshape_method = getattr(cfg, 'reshape_method', 'no_sample')

        if reshape_method == 'no_sample':
            event = reshape_event_no_sample(event, SENSOR_H, SENSOR_W, IMAGE_H,
                                            IMAGE_W)
        elif reshape_method == 'sample':
            event = reshape_event_with_sample(event, SENSOR_H, SENSOR_W,
                                              IMAGE_H, IMAGE_W)
        elif reshape_method == 'unique':
            event = reshape_event_unique(event, SENSOR_H, SENSOR_W, IMAGE_H,
                                         IMAGE_W)

    # print("after reshape x", event[:,0].max(), event[:,0].min())
    # print("after reshape y", event[:,1].max(), event[:,1].min())
    # print("after reshape t", event[:,3].max(), event[:,3].min())
    # Account for slicing
    slice_events = getattr(cfg, 'slice_events', False)

    if slice_events:
        event = slice_event(event, cfg)

    # print("after slice x", event[:,0].max(), event[:,0].min())
    # print("after slice y", event[:,1].max(), event[:,1].min())
    # print("after slice t", event[:,3].max(), event[:,3].min())

    return event


# Aggregation Modules


# Reimplement EST
def est_aggregation(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a (2 * C) * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # print("ini x", event_tensor[:,0].max(), event_tensor[:,0].min())
    # print("ini y", event_tensor[:,1].max(), event_tensor[:,1].min())
    # print("ini p", event_tensor[:,3].max(), event_tensor[:,3].min())

    # print("event_tensot shape:", event_tensor.shape)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)
    C = kwargs.get('channel_num', 9)

    mlp_layers = kwargs.get('value-layer_mlp', [1, 30, 30, 1])

    value_layer = ValueLayer(mlp_layers,
                             activation=nn.LeakyReLU(negative_slope=0.1),
                             num_channels=C)

    num_voxels = 2 * C * H * W
    vox = event_tensor[0].new_full([
        num_voxels,
    ], fill_value=0)

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]
    # event_tensor[: 2] = (event_tensor[: 2] - start_time) / time_length
    x = event_tensor[:, 0].long()
    y = event_tensor[:, 1].long()
    t = (event_tensor[:, 2] - start_time) / time_length
    p = event_tensor[:, 3].long()

    idx_before_bins = x \
                    + W * y \
                    + 0 \
                    + W * H * C * (p + 1) / 2

    for i_bin in range(C):
        values = t * value_layer.forward(t - i_bin / (C - 1))

        # draw in voxel grid
        idx = idx_before_bins + W * H * i_bin
        vox.put_(idx.long(), values, accumulate=True)

    vox = vox.view(2, C, H, W)
    vox = torch.cat([vox[0, ...], vox[1, ...]], 0)
    vox = vox.float()

    return vox


# Reimplement EST with positive polarity only
def est_aggregation_positive(event_tensor, augment=None, **kwargs):
    '''
    Accumulate only positive events to create a (1 * C) * H * W image
    Args:
        event_tensor: (N, 4) tensor
        augment: augmentation function
        kwargs: other arguments
    Returns:
        vox: (1 * C, H, W) tensor
    '''

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # print("ini x", event_tensor[:,0].max(), event_tensor[:,0].min())
    # print("ini y", event_tensor[:,1].max(), event_tensor[:,1].min())
    # print("ini p", event_tensor[:,3].max(), event_tensor[:,3].min())

    # print("event_tensot shape:", event_tensor.shape)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)
    C = kwargs.get('channel_num', 9)

    mlp_layers = kwargs.get('value-layer_mlp', [1, 30, 30, 1])

    value_layer = ValueLayer(mlp_layers,
                             activation=nn.LeakyReLU(negative_slope=0.1),
                             num_channels=C)

    # num_voxels = 2 * C * H * W # original
    num_voxels = 1 * C * H * W  # for positive polarity only
    vox = event_tensor[0].new_full([
        num_voxels,
    ], fill_value=0)

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]
    # event_tensor[: 2] = (event_tensor[: 2] - start_time) / time_length
    x = event_tensor[:, 0].long()
    y = event_tensor[:, 1].long()
    t = (event_tensor[:, 2] - start_time) / time_length
    p = event_tensor[:, 3].long()

    # Only positive polarity
    mask = p > 0
    x = x[mask].reshape(-1, 1)
    y = y[mask].reshape(-1, 1)
    t = t[mask].reshape(-1, 1)
    p = p[mask].reshape(-1, 1)

    idx_before_bins = x \
                    + W * y \
                    + 0 \
                    + W * H * C * p # for positive polarity
    # + W * H * C * (p + 1) / 2 # for both polarity

    for i_bin in range(C):
        values = t * value_layer.forward(t - i_bin / (C - 1))

        # draw in voxel grid
        idx = idx_before_bins + W * H * i_bin
        vox.put_(idx.long(), values, accumulate=True)

    vox = vox.view(1, C, H, W)
    vox = vox[0, ...]  # for positive polarity
    # vox = torch.cat([vox[0, ...], vox[1, ...]], 0) # for both polarity
    vox = vox.float()

    return vox


# Reimplement EST, ignore polarity
def est_aggregation_nop(event_tensor, augment=None, **kwargs):
    '''
    Accumulate events to create a (1 * C) * H * W image (ignore polarity)
    Args:
        event_tensor: (N, 4) tensor
        augment: augmentation function
        kwargs: other arguments
    Returns:
        vox: (1 * C, H, W) tensor
    '''

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # print("ini x", event_tensor[:,0].max(), event_tensor[:,0].min())
    # print("ini y", event_tensor[:,1].max(), event_tensor[:,1].min())
    # print("ini p", event_tensor[:,3].max(), event_tensor[:,3].min())

    # print("event_tensot shape:", event_tensor.shape)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)
    C = kwargs.get('channel_num', 9)

    mlp_layers = kwargs.get('value-layer_mlp', [1, 30, 30, 1])

    value_layer = ValueLayer(mlp_layers,
                             activation=nn.LeakyReLU(negative_slope=0.1),
                             num_channels=C)

    num_voxels = 1 * C * H * W
    vox = event_tensor[0].new_full([
        num_voxels,
    ], fill_value=0)

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]
    # event_tensor[: 2] = (event_tensor[: 2] - start_time) / time_length
    x = event_tensor[:, 0].long()
    y = event_tensor[:, 1].long()
    t = (event_tensor[:, 2] - start_time) / time_length

    # ignore polarity
    p[p < 0] = 1
    p = event_tensor[:, 3].long()

    idx_before_bins = x \
                    + W * y \
                    + 0 \
                    + W * H * C * p   # ignore polarity
    # + W * H * C * (p + 1) / 2 # original

    for i_bin in range(C):
        values = t * value_layer.forward(t - i_bin / (C - 1))

        # draw in voxel grid
        idx = idx_before_bins + W * H * i_bin
        vox.put_(idx.long(), values, accumulate=True)

    vox = vox.view(1, C, H, W)
    vox = vox[0, ...]  # ignore polarity
    # vox = torch.cat([vox[0, ...], vox[1, ...]], 0) # original
    vox = vox.float()

    return vox


def reshape_then_acc(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    pos_max = pos_count.max().float()
    pos_count = pos_count / pos_max

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    neg_max = neg_count.max().float()
    neg_count = neg_count / neg_max

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    result = torch.stack([pos_count, pos_out, neg_count, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_time(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)

    pos_min_out, _ = scatter_min(norm_pos_time,
                                 pos_idx,
                                 dim=-1,
                                 dim_size=H * W)
    pos_min_out = pos_min_out.reshape(H, W)
    neg_min_out, _ = scatter_min(norm_neg_time,
                                 neg_idx,
                                 dim=-1,
                                 dim_size=H * W)
    neg_min_out = neg_min_out.reshape(H, W)

    result = torch.stack([pos_min_out, pos_out, neg_min_out, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Account for empty events
    if len(event_tensor) == 0:
        event_tensor = torch.zeros([10, 4]).float()
        event_tensor[:, 2] = torch.arange(10) / 10.
        event_tensor[:, -1] = 1

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    result = torch.stack([pos_count, pos_out, neg_count, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)

    result = torch.stack([pos_count, neg_count], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count_only(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 1 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    # Get pos, neg counts
    event_count = torch.bincount(event_tensor[:, 0].long() +
                                 event_tensor[:, 1].long() * W,
                                 minlength=H * W).reshape(H, W)

    result = torch.unsqueeze(event_count, -1)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_all(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 6 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    if event_tensor.shape[0] == 0:
        return torch.zeros([6, IMAGE_H, IMAGE_W])

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)

    pos_min_out, _ = scatter_min(norm_pos_time,
                                 pos_idx,
                                 dim=-1,
                                 dim_size=H * W)
    pos_min_out = pos_min_out.reshape(H, W)
    neg_min_out, _ = scatter_min(norm_neg_time,
                                 neg_idx,
                                 dim=-1,
                                 dim_size=H * W)
    neg_min_out = neg_min_out.reshape(H, W)

    result = torch.stack(
        [pos_count, neg_count, pos_out, neg_out, pos_min_out, neg_min_out],
        dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_flat(event_tensor, augment=None, **kwargs):
    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    if augment is not None:
        event_tensor = augment(event_tensor)

    coords = event_tensor[:, :2].long()
    event_image = torch.zeros([H, W])
    event_image[(coords[:, 1], coords[:, 0])] = 1.0

    event_image = torch.unsqueeze(event_image, -1)

    event_image = event_image.permute(2, 0, 1)
    event_image = event_image.float()

    return event_image


def reshape_then_flat_pol(event_tensor, augment=None, **kwargs):
    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    if augment is not None:
        event_tensor = augment(event_tensor)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    pos_coords = pos[:, :2].long()
    neg_coords = neg[:, :2].long()

    pos_image = torch.zeros([H, W])
    neg_image = torch.zeros([H, W])

    pos_image[(pos_coords[:, 1], pos_coords[:, 0])] = 1.0
    neg_image[(neg_coords[:, 1], neg_coords[:, 0])] = 1.0

    event_image = torch.stack([pos_image, neg_image], dim=2)
    event_image = event_image.permute(2, 0, 1)
    event_image = event_image.float()

    return event_image


def reshape_then_acc_exp(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    pos_out_exp = torch.exp(-(1 - pos_out) / EXP_TAU)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    neg_out_exp = torch.exp(-(1 - neg_out) / EXP_TAU)

    result = torch.stack([pos_out_exp, neg_out_exp], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_time_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Account for empty events
    if len(event_tensor) == 0:
        event_tensor = torch.zeros([10, 4]).float()
        event_tensor[:, 2] = torch.arange(10) / 10.
        event_tensor[:, -1] = 1

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)

    result = torch.stack([pos_out, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_sort(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W (or 2 if neglect_polarity) image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Get sorted indices of event tensor
    if kwargs['global_time']:
        time_idx = (event_tensor[:, 2] * TIME_SCALE).long()
        mem, cnt = torch.unique_consecutive(time_idx, return_counts=True)
        time_idx = torch.repeat_interleave(torch.arange(mem.shape[0]), cnt)
        event_tensor[:, 2] = time_idx
    else:
        # If global_time is False, split time sorting across polarity
        time_idx = (event_tensor[:, 2] * TIME_SCALE).long()
        pos_time_idx = time_idx[event_tensor[:, 3] > 0]
        neg_time_idx = time_idx[event_tensor[:, 3] < 0]

        pos_mem, pos_cnt = torch.unique_consecutive(pos_time_idx,
                                                    return_counts=True)
        pos_time_idx = torch.repeat_interleave(torch.arange(pos_mem.shape[0]),
                                               pos_cnt)

        neg_mem, neg_cnt = torch.unique_consecutive(neg_time_idx,
                                                    return_counts=True)
        neg_time_idx = torch.repeat_interleave(torch.arange(neg_mem.shape[0]),
                                               neg_cnt)

        event_tensor[:, 2] = time_idx

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)
    if kwargs['neglect_polarity']:
        # Get counts
        count = torch.bincount(event_tensor[:, 0].long() +
                               event_tensor[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
        count = count.float()

        if kwargs['use_image']:
            # Trimmed Event Image
            coords = event_tensor[:, :2].long()
            event_image = torch.zeros([H, W])
            event_image[(coords[:, 1], coords[:, 0])] = 1.0

            if kwargs['denoise_image']:
                event_image = density_filter_event_image(event_image, count)

        # Get time
        if kwargs[
                'strict']:  # If strict is True, sorts once more to get a 'rigorous' sorted image
            idx = event_tensor[:, 0].long() + event_tensor[:, 1].long() * W
            scatter_result, scatter_idx = scatter_max(event_tensor[:, 2],
                                                      idx,
                                                      dim=-1,
                                                      dim_size=H * W)

            idx_mask = torch.zeros(event_tensor.shape[0], dtype=torch.bool)
            idx_mask[scatter_idx[scatter_idx < idx.shape[0]]] = True
            event_tensor = event_tensor[idx_mask]

            final_mem, final_cnt = torch.unique_consecutive(event_tensor[:, 2],
                                                            return_counts=True)
            # One is added to ensure that sorted values are greater than 1
            final_scatter = torch.repeat_interleave(
                torch.arange(final_mem.shape[0]), final_cnt).float() + 1
            if final_scatter.max() != final_scatter.min():
                final_scatter = (final_scatter - final_scatter.min()) / (
                    final_scatter.max() - final_scatter.min())
            else:
                final_scatter.fill_(0.0)

            event_sort = torch.zeros(H, W)
            coords = event_tensor[:, :2].long()

            event_sort[(coords[:, 1], coords[:, 0])] = final_scatter
            event_sort = event_sort.reshape(H, W)

        else:
            idx = event_tensor[:, 0].long() + event_tensor[:, 1].long() * W
            event_sort, _ = scatter_max(event_tensor[:, 2],
                                        idx,
                                        dim=-1,
                                        dim_size=H * W)
            hot_event_sort = event_sort[event_sort > 0.0]

            if hot_event_sort.max() != hot_event_sort.min():
                hot_event_sort = (hot_event_sort - hot_event_sort.min()) / (
                    hot_event_sort.max() - hot_event_sort.min())
            else:
                hot_event_sort = hot_event_sort - hot_event_sort.min()

            event_sort = event_sort.reshape(H, W)

        if kwargs['denoise_sort']:
            event_sort = density_filter_event_image(event_sort, count)

            if event_sort.max() != event_sort.min():
                event_sort = (event_sort - event_sort.min()) / (
                    event_sort.max() - event_sort.min())

        if kwargs['quantize_sort'] is not None:
            if type(kwargs['quantize_sort']
                    ) == int:  # If we want single quantization
                event_sort = torch.round(
                    event_sort *
                    kwargs['quantize_sort']) / kwargs['quantize_sort']
            elif type(kwargs['quantize_sort']
                      ) == list:  # If we want multiple quantization
                event_sort_list = []
                for quantize_size in kwargs['quantize_sort']:
                    event_sort_list.append(
                        torch.round(event_sort * quantize_size) /
                        quantize_size)
                event_sort = torch.stack(event_sort_list, dim=2)

        if kwargs['use_image']:
            if len(event_sort.shape) == 2:
                result = torch.stack([event_image, event_sort], dim=2)
            else:
                result = torch.cat([event_image.unsqueeze(-1), event_sort],
                                   dim=2)
        else:
            if len(event_sort.shape) == 2:
                result = event_sort.unsqueeze(-1)
            else:
                result = event_sort

        result = result.permute(2, 0, 1)
        result = result.float()

    else:
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]

        if pos.shape[0] == 0:
            pos = torch.zeros(1, 4)
            pos[:, -1] = 1
        if neg.shape[0] == 0:
            neg = torch.zeros(1, 4)
            neg[:, -1] = 1

        # Get pos, neg counts
        pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                                   minlength=H * W).reshape(H, W)
        pos_count = pos_count.float()

        neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                                   minlength=H * W).reshape(H, W)
        neg_count = neg_count.float()

        if kwargs['use_image']:
            # Trimmed Event Image
            pos_coords = pos[:, :2].long()
            pos_event_image = torch.zeros([H, W])
            pos_event_image[(pos_coords[:, 1], pos_coords[:, 0])] = 1.0

            neg_coords = neg[:, :2].long()
            neg_event_image = torch.zeros([H, W])
            neg_event_image[(neg_coords[:, 1], neg_coords[:, 0])] = 1.0

            if kwargs['denoise_image']:
                pos_event_image = density_filter_event_image(
                    pos_event_image, pos_count)
                neg_event_image = density_filter_event_image(
                    neg_event_image, neg_count)

        # Get pos, neg time
        if kwargs['strict']:
            # Get pos sort
            pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
            pos_scatter_result, pos_scatter_idx = scatter_max(pos[:, 2],
                                                              pos_idx,
                                                              dim=-1,
                                                              dim_size=H * W)

            pos_idx_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
            pos_idx_mask[pos_scatter_idx[
                pos_scatter_idx < pos_idx.shape[0]]] = True
            tmp_pos = pos[pos_idx_mask]

            pos_final_mem, pos_final_cnt = torch.unique_consecutive(
                tmp_pos[:, 2], return_counts=True)
            # One is added to ensure that sorted values are greater than 1
            pos_final_scatter = torch.repeat_interleave(
                torch.arange(pos_final_mem.shape[0]),
                pos_final_cnt).float() + 1

            if pos_final_scatter.max() != pos_final_scatter.min():
                pos_final_scatter = (
                    pos_final_scatter - pos_final_scatter.min()) / (
                        pos_final_scatter.max() - pos_final_scatter.min())
            else:
                pos_final_scatter.fill_(0.0)

            pos_sort = torch.zeros(H, W)
            pos_coords = tmp_pos[:, :2].long()

            pos_sort[(pos_coords[:, 1], pos_coords[:, 0])] = pos_final_scatter
            pos_sort = pos_sort.reshape(H, W)

            # Get neg_sort
            neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
            neg_scatter_result, neg_scatter_idx = scatter_max(neg[:, 2],
                                                              neg_idx,
                                                              dim=-1,
                                                              dim_size=H * W)

            neg_idx_mask = torch.zeros(neg.shape[0], dtype=torch.bool)
            neg_idx_mask[neg_scatter_idx[
                neg_scatter_idx < neg_idx.shape[0]]] = True
            tmp_neg = neg[neg_idx_mask]

            neg_final_mem, neg_final_cnt = torch.unique_consecutive(
                tmp_neg[:, 2], return_counts=True)
            # One is added to ensure that sorted values are greater than 1
            neg_final_scatter = torch.repeat_interleave(
                torch.arange(neg_final_mem.shape[0]),
                neg_final_cnt).float() + 1
            if neg_final_scatter.max() != neg_final_scatter.min():
                neg_final_scatter = (
                    neg_final_scatter - neg_final_scatter.min()) / (
                        neg_final_scatter.max() - neg_final_scatter.min())
            else:
                neg_final_scatter.fill_(0.0)

            neg_sort = torch.zeros(H, W)
            neg_coords = tmp_neg[:, :2].long()

            neg_sort[(neg_coords[:, 1], neg_coords[:, 0])] = neg_final_scatter
            neg_sort = neg_sort.reshape(H, W)

        else:
            pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
            neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
            pos_sort, _ = scatter_max(pos[:, 2],
                                      pos_idx,
                                      dim=-1,
                                      dim_size=H * W)
            hot_pos_sort = pos_sort[pos_sort > 0.0]

            if hot_pos_sort.max() != hot_pos_sort.min():
                hot_pos_sort = (hot_pos_sort - hot_pos_sort.min()) / (
                    hot_pos_sort.max() - hot_pos_sort.min())
            else:
                hot_pos_sort = hot_pos_sort - hot_pos_sort.min()
            pos_sort = pos_sort.reshape(H, W)

            neg_sort, _ = scatter_max(neg[:, 2],
                                      neg_idx,
                                      dim=-1,
                                      dim_size=H * W)
            hot_neg_sort = neg_sort[neg_sort > 0.0]

            if hot_neg_sort.max() != hot_neg_sort.min():
                hot_neg_sort = (hot_neg_sort - hot_neg_sort.min()) / (
                    hot_neg_sort.max() - hot_neg_sort.min())
            else:
                hot_neg_sort = hot_neg_sort - hot_neg_sort.min()
            neg_sort = neg_sort.reshape(H, W)

        if kwargs['denoise_sort']:
            pos_sort = density_filter_event_image(pos_sort, pos_count)
            neg_sort = density_filter_event_image(neg_sort, neg_count)

            if pos_sort.max() != pos_sort.min():
                pos_sort = (pos_sort - pos_sort.min()) / (pos_sort.max() -
                                                          pos_sort.min())
            if neg_sort.max() != neg_sort.min():
                neg_sort = (neg_sort - neg_sort.min()) / (neg_sort.max() -
                                                          neg_sort.min())

        if kwargs['quantize_sort'] is not None:
            if type(kwargs['quantize_sort']
                    ) == int:  # If we want single quantization
                pos_sort = torch.round(
                    pos_sort *
                    kwargs['quantize_sort']) / kwargs['quantize_sort']
                neg_sort = torch.round(
                    neg_sort *
                    kwargs['quantize_sort']) / kwargs['quantize_sort']
            elif type(kwargs['quantize_sort']
                      ) == list:  # If we want multiple quantization
                pos_sort_list = []
                neg_sort_list = []
                for quantize_size in kwargs['quantize_sort']:
                    pos_sort_list.append(
                        torch.round(pos_sort * quantize_size) / quantize_size)
                    neg_sort_list.append(
                        torch.round(neg_sort * quantize_size) / quantize_size)
                pos_sort = torch.stack(pos_sort_list, dim=2)
                neg_sort = torch.stack(neg_sort_list, dim=2)

        if kwargs['use_image']:
            if len(pos_sort.shape) == 2:
                result = torch.stack(
                    [pos_event_image, pos_sort, neg_event_image, neg_sort],
                    dim=2)
            else:
                result = torch.cat([
                    pos_event_image.unsqueeze(-1), pos_sort,
                    neg_event_image.unsqueeze(-1), neg_sort
                ],
                                   dim=2)
        else:
            if len(pos_sort.shape) == 2:
                result = torch.stack([pos_sort, neg_sort], dim=2)
            else:
                result = torch.cat([pos_sort, neg_sort], dim=2)

        result = result.permute(2, 0, 1)
        result = result.float()

    return result


def reshape_then_acc_intensity(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 1 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    pos_count = pos_count.float()

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    neg_count = neg_count.float()

    intensity = pos_count - neg_count
    intensity = (intensity - intensity.min()) / (intensity.max() -
                                                 intensity.min())

    result = intensity.unsqueeze(0)
    result = result.float()
    return result


def reshape_then_acc_adj_sort(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    pos_count = pos_count.float()

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W,
                               minlength=H * W).reshape(H, W)
    neg_count = neg_count.float()

    # clip count
    pos_unique_count = torch.unique(pos_count, return_counts=True)[1]
    pos_sum_subset = torch.cumsum(pos_unique_count, dim=0)
    pos_th_clip = pos_sum_subset[pos_sum_subset < H * W *
                                 CLIP_COUNT_RATE].shape[0]
    pos_count[pos_count > pos_th_clip] = pos_th_clip

    neg_unique_count = torch.unique(neg_count, return_counts=True)[1]
    neg_sum_subset = torch.cumsum(neg_unique_count, dim=0)
    neg_th_clip = neg_sum_subset[neg_sum_subset < H * W *
                                 CLIP_COUNT_RATE].shape[0]
    neg_count[neg_count > neg_th_clip] = neg_th_clip

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length

    # Get pos, neg time
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_min_out, _ = scatter_min(norm_pos_time,
                                 pos_idx,
                                 dim=-1,
                                 dim_size=H * W)
    pos_out = pos_out.reshape(H, W).float()
    pos_min_out = pos_min_out.reshape(H, W).float()
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_min_out, _ = scatter_min(norm_neg_time,
                                 neg_idx,
                                 dim=-1,
                                 dim_size=H * W)
    neg_out = neg_out.reshape(H, W).float()
    neg_min_out = neg_min_out.reshape(H, W).float()

    pos_min_out[pos_count == 0] = 1.0
    neg_min_out[neg_count == 0] = 1.0

    # Get temporal discount
    pos_disc = torch.zeros_like(pos_count)
    neg_disc = torch.zeros_like(neg_count)

    patch_size = 5

    pos_neighbor_count = patch_size**2 * torch.nn.functional.avg_pool2d(
        pos_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)
    neg_neighbor_count = patch_size**2 * torch.nn.functional.avg_pool2d(
        neg_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)

    pos_disc = (torch.nn.functional.max_pool2d(pos_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) +
        torch.nn.functional.max_pool2d(-pos_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (pos_neighbor_count)
    neg_disc = (torch.nn.functional.max_pool2d(neg_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) +
        torch.nn.functional.max_pool2d(-neg_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (neg_neighbor_count)

    pos_out[pos_count > 0] = (pos_out[pos_count > 0] -
                              DISC_ALPHA * pos_disc.squeeze()[pos_count > 0])
    pos_out[pos_out < 0] = 0
    pos_out[pos_neighbor_count.squeeze() == 1.0] = 0
    neg_out[neg_count > 0] = (neg_out[neg_count > 0] -
                              DISC_ALPHA * neg_disc.squeeze()[neg_count > 0])
    neg_out[neg_out < 0] = 0
    neg_out[neg_neighbor_count.squeeze() == 1.0] = 0

    pos_out = pos_out.reshape(H * W)
    neg_out = neg_out.reshape(H * W)

    pos_val, pos_idx = torch.sort(pos_out)
    neg_val, neg_idx = torch.sort(neg_out)

    pos_unq, pos_cnt = torch.unique_consecutive(pos_val, return_counts=True)
    neg_unq, neg_cnt = torch.unique_consecutive(neg_val, return_counts=True)

    pos_sort = torch.zeros_like(pos_out)
    neg_sort = torch.zeros_like(neg_out)

    pos_sort[pos_idx] = torch.repeat_interleave(torch.arange(
        pos_unq.shape[0]), pos_cnt).float() / pos_unq.shape[0]
    neg_sort[neg_idx] = torch.repeat_interleave(torch.arange(
        neg_unq.shape[0]), neg_cnt).float() / neg_unq.shape[0]

    pos_sort = pos_sort.reshape(H, W)
    neg_sort = neg_sort.reshape(H, W)

    result = torch.stack([pos_sort, neg_sort], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()

    return result


# Augmentation Modules


def random_shift_events(event_tensor, max_shift=20, resolution=(224, 224)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2, ))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (
        event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor


def random_flip_events_along_x(event_tensor, resolution=(224, 224), p=0.5):
    H, W = resolution

    if np.random.random() < p:
        event_tensor[:, 0] = W - 1 - event_tensor[:, 0]

    return event_tensor


def random_time_flip(event_tensor, resolution=(224, 224), p=0.5):
    if np.random.random() < p:
        event_tensor = torch.flip(event_tensor, [0])
        event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
        event_tensor[:,
                     3] = -event_tensor[:,
                                        3]  # Inversion in time means inversion in polarity
    return event_tensor


def base_augment(mode):
    assert mode in ['train', 'eval']

    if mode == 'train':

        def augment(event):
            event = random_time_flip(event, resolution=(IMAGE_H, IMAGE_W))
            event = random_flip_events_along_x(event)
            event = random_shift_events(event)
            return event

        return augment

    elif mode == 'eval':
        return None


class ImageNetDataset(Dataset):

    def __init__(self, cfg, mode='train'):
        super(ImageNetDataset, self).__init__()
        self.mode = mode
        self.train_file = open(cfg.train_file, 'r').readlines()
        self.val_file = open(cfg.val_file, 'r').readlines()

        self.train_file = [(Path(s.strip())) for s in self.train_file]
        self.val_file = [(Path(s.strip())) for s in self.val_file]

        if mode == 'train':
            self.map_file = self.train_file
        elif mode == 'val':
            self.map_file = self.val_file
        elif mode == 'test':
            self.map_file = self.val_file

        self.labels = [
            s.split()[1].strip() for s in open(cfg.label_map, 'r').readlines()
        ]
        self.labels = sorted(self.labels[:1000])

        if getattr(cfg, 'trim_class_count', None) is not None:
            self.labels = self.labels[:cfg.trim_class_count]
            self.map_file = list(
                filter(lambda s: s.parent.stem in self.labels, self.map_file))

        self.label_map = {s: idx for idx, s in enumerate(self.labels)}

        self.cfg = cfg
        self.augment_type = getattr(cfg, 'augment_type', None)
        self.loader_type = getattr(cfg, 'loader_type', None)
        self.parser_type = getattr(cfg, 'parser_type', 'normal')
        assert self.parser_type in ['normal']

        # Choose parser (event path -> (N, 4) event tensor)
        if self.parser_type == 'normal':
            self.event_parser = self.augment_parser(parse_event)

        # Choose loader ((N, 4) event tensor -> Network input)
        if self.loader_type is None or self.loader_type == 'reshape_then_acc':
            self.loader = reshape_then_acc
        elif self.loader_type == 'reshape_then_acc_time':
            self.loader = reshape_then_acc_time
        elif self.loader_type == 'reshape_then_acc_count':
            self.loader = reshape_then_acc_count
        elif self.loader_type == 'reshape_then_acc_all':
            self.loader = reshape_then_acc_all
        elif self.loader_type == 'reshape_then_flat_pol':
            self.loader = reshape_then_flat_pol
        elif self.loader_type == 'reshape_then_flat':
            self.loader = reshape_then_flat
        elif self.loader_type == 'reshape_then_acc_time_pol':
            self.loader = reshape_then_acc_time_pol
        elif self.loader_type == 'reshape_then_acc_count_pol':
            self.loader = reshape_then_acc_count_pol
        elif self.loader_type == 'reshape_then_acc_exp':
            self.loader = reshape_then_acc_exp
        elif self.loader_type == 'reshape_then_acc_sort':
            self.loader = reshape_then_acc_sort
        elif self.loader_type == 'reshape_then_acc_intensity':
            self.loader = reshape_then_acc_intensity
        elif self.loader_type == 'reshape_then_acc_adj_sort':
            self.loader = reshape_then_acc_adj_sort
        elif self.loader_type == 'est_aggregation':
            self.loader = est_aggregation
        elif self.loader_type == 'est_aggregation_nop':
            self.loader = est_aggregation_nop
        elif self.loader_type == 'est_aggregation_positive':
            self.loader = est_aggregation_positive

    def augment_parser(self, parser):

        def new_parser(event_path):
            return parser(event_path, self.cfg)

        return new_parser

    def __getitem__(self, idx):
        event_path = self.map_file[idx]
        label = self.label_map[event_path.parent.stem]

        # Load and optionally reshape event from event_path
        event = self.event_parser(event_path)
        augment_mode = 'train' if self.mode == 'train' else 'eval'
        event = self.loader(event,
                            augment=base_augment(augment_mode),
                            neglect_polarity=getattr(self.cfg,
                                                     'neglect_polarity',
                                                     False),
                            global_time=getattr(self.cfg, 'global_time', True),
                            strict=getattr(self.cfg, 'strict', False),
                            use_image=getattr(self.cfg, 'use_image', True),
                            denoise_sort=getattr(self.cfg, 'denoise_sort',
                                                 False),
                            denoise_image=getattr(self.cfg, 'denoise_image',
                                                  False),
                            filter_flash=getattr(self.cfg, 'filter_flash',
                                                 False),
                            filter_noise=getattr(self.cfg, 'filter_noise',
                                                 False),
                            quantize_sort=getattr(self.cfg, 'quantize_sort',
                                                  None))

        return event, label

    def __len__(self):
        return len(self.map_file)
