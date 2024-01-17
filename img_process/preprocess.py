import io
import random
import torch
import os.path as osp
import base64
from PIL import Image
import torchvision.transforms as T
import numpy as np


def base64_to_image(base64_code):
    """将base64的数据转换成rgb格式的图像矩阵"""
    img_data = base64.b64decode(base64_code)
    # img_array = np.frombuffer(img_data, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = Image.open(io.BytesIO(img_data))
    return img


def get_transform():
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        normalize_transform
    ])
    return eval_transforms


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_names(fpath):
    names = []
    with open(fpath, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def get_indices(num, seq_len):
    """
    Evenly sample seq_len items from num items.
    """
    indices_list = []
    if num >= seq_len:
        r = num % seq_len
        stride = num // seq_len
        if r != 0:
            stride += 1
        for i in range(stride):
            indices = np.arange(i, stride * seq_len, stride)
            indices = indices.clip(max=num - 1)
            indices_list.append(indices)
    else:
        # if num is smaller than seq_len, simply replicate the last image
        # until the seq_len requirement is satisfied
        indices = np.arange(0, num)
        num_pads = seq_len - num
        indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
        indices_list.append(indices)

    if len(indices_list) > 50:
        indices_list = indices_list[:50]

    return indices_list


def process_data(track_test, test_names, root, home_dir):
    tracklets = []
    num_imgs_per_tracklet = []
    min_seq_len = 1
    num_tracklets = track_test.shape[0]
    for tracklet_idx in range(num_tracklets):
        data = track_test[tracklet_idx, ...]
        start_index, end_index, pid, camid = data
        if pid == -1: continue  # junk images are just ignored

        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        img_names = test_names[start_index - 1:end_index]

        # make sure image names correspond to the same person
        pnames = [img_name[:4] for img_name in img_names]
        assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

        # make sure all images are captured under the same camera
        camnames = [img_name[5] for img_name in img_names]
        assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

        # append image names with directory information
        img_paths = [osp.join(root, home_dir, img_name[:4], img_name) for img_name in img_names]
        if len(img_paths) >= min_seq_len:
            img_paths = tuple(img_paths)
            tracklets.append((img_paths, pid, camid))
            num_imgs_per_tracklet.append(len(img_paths))

    num_tracklets = len(tracklets)

    return tracklets, num_tracklets
