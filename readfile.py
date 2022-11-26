import numpy as np
from os import listdir
import os

def convert(input_fn, output_fn):
    content = np.fromfile(input_fn, dtype=np.uint8).astype(np.int32)

    res = {}
    res['x'] = content[0:][::5]
    res['y'] = content[1:][::5]
    res['p'] = np.right_shift(content[2:][::5], 7) * 2 - 3
    res['t'] = np.left_shift(np.bitwise_and(content[2:][::5], 127), 16)
    res['t'] = res['t'] + np.left_shift(content[3:][::5], 8)
    res['t'] = res['t'] + content[4:][::5]

    event = {'event_data': res}
    
    np.save(output_fn, event)

if __name__ == "__main__":
    # label mapping file
    root_dir = 'Caltech101'
    labels = sorted(listdir(root_dir))
    with open("mapping.txt", "w") as f:
        for i, label in enumerate(labels):
            f.write(str(i) + " " + label)
            if i != len(labels) - 1:
                f.write("\n")

    # event folder structure
    # root dir
    new_root_dir = 'Caltech101-NIN'
    if not os.path.exists(new_root_dir):
        os.makedirs(new_root_dir)
    
    # sub dir
    for label in labels:
        sub_dir = os.path.join(new_root_dir, label)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    
    # event list
    event_fnames = []
    new_event_fnames = []
    for label in labels:
        fns = sorted(listdir(os.path.join(root_dir, label)))
        new_fns = [fn.split('.')[0] + '.npy' for fn in fns]

        abs_fns = [os.path.join(root_dir, label, fn) for fn in fns]
        new_abs_fns = [os.path.join(new_root_dir, label, fn) for fn in new_fns]

        event_fnames += abs_fns
        new_event_fnames += new_abs_fns

    # convert from .bin to .npy
        for input_fn, out_fn in zip(event_fnames, new_event_fnames):
            convert(input_fn, out_fn)
    
    # split train and val
    choices = np.random.choice([0, 1], size=(len(new_event_fnames),), p=[.15, .85])
    train_list = new_event_fnames[choices==1]
    val_list = new_event_fnames[choices==0]

    train_fn = "train_list.txt"
    val_fn = "val_list.txt"
    with open(train_fn, "w") as f:
        for i, fn in enumerate(train_list):
            f.write(fn)
            if i != len(train_fn) - 1:
                f.write("\n")
    
    with open(val_fn, "w") as f:
        for i, fn in enumerate(val_list):
            f.write(fn)
            if i != len(val_list) - 1:
                f.write("\n")




    






