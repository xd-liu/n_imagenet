import numpy as np
import os
import skvideo.io
import tqdm


def event_format_convert(input_fn):
    '''
    Args:
        input_fn (str): path to the input file
        output_fn (str): path to the output file
    Returns:
        res (dict): dictionary containing the events:
            - t: [N]
            - x: [N], x coordinate of the event
            - y: [N], y coordinate of the event
            - p: [N], polarity of the event
    '''
    content = np.fromfile(input_fn, dtype=np.uint8).astype(np.int32)

    res = {}
    res['x'] = content[0:][::5]
    res['y'] = content[1:][::5]
    res['p'] = np.right_shift(content[2:][::5], 7) + 1
    res['t'] = np.left_shift(np.bitwise_and(content[2:][::5], 127), 16)
    res['t'] = res['t'] + np.left_shift(content[3:][::5], 8)
    res['t'] = res['t'] + content[4:][::5]

    return res


def event_processor(x, y, p, red, blue, output_frame):
    '''
    Args:
        x (np.ndarray): [N], x coordinate of the event
        y (np.ndarray): [N], y coordinate of the event
        p (np.ndarray): [N], polarity of the event
        red (np.ndarray): [3], red color
        blue (np.ndarray): [3], blue color
        output_frame (np.ndarray): [H, W, 3], output frame
        
    Returns:
        output_frame (np.ndarray): [H, W, 3], output frame
    '''
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 1:
            output_frame[y_, x_] = blue
        else:
            output_frame[y_, x_] = red

    return output_frame


def save_to_video(target_path, shape, data_path, fps=30):
    '''
    Args:
        target_path (str): path to save the video
        shape (tuple): (height, width) of the saved video
        data (dict): dictionary containing the events:
            - t: [N]
            - x: [N], x coordinate of the event
            - y: [N], y coordinate of the event
            - p: [N], polarity of the event
        fps (int): FPS of the saved video
    '''
    # convert each non-overlapping time-window to a frame
    data = event_format_convert(data_path)
    tmin, tmax = data['t'][[0, -1]]
    tmin, tmax = tmin.item(), tmax.item()
    # t here are in unit 1e-6 s
    t0 = np.arange(tmin, tmax, 1e6 // 100)
    t1, t0 = t0[1:], t0[:-1]
    idx0 = np.searchsorted(data['t'], t0)
    idx1 = np.searchsorted(data['t'], t1)


    path = os.path.join(target_path, "events.mp4")
    writer = skvideo.io.FFmpegWriter(path, inputdict={'-framerate': str(fps)})

    red = np.array([255, 0, 0], dtype="uint8")
    blue = np.array([0, 0, 255], dtype="uint8")

    pbar = tqdm.tqdm(total=len(idx0))
    for i0, i1 in zip(idx0, idx1):
        sub_data = {
            k: v[i0:i1].astype("int32") # for ndarray
            # k: v[i0:i1].cpu().numpy().astype("int32") # for torch tensor
            for k, v in data.items()
        }
        frame = np.full(shape=shape + (3, ), fill_value=255, dtype="uint8")
        event_processor(sub_data['x'], sub_data['y'], sub_data['p'], red, blue,
                        frame)
        writer.writeFrame(frame)
        pbar.update(1)
    writer.close()

    return path


def test_event2video():
    # load events
    input_fn = "/home/xudong99/scratch/cy6cvx3ryv-1/Caltech101/accordion/image_0001.bin"
    # save to video
    shape = (256, 256)
    save_to_video("./video/", shape, input_fn)


if __name__ == '__main__':
    test_event2video()
