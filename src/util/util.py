import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import cv2


def make_ids(path):
    ids = []
    frames = os.listdir(path)
    for frame in frames:
        (filename, extension) = os.path.splitext(frame)
        ids.append(int(filename))
    ids = sorted(ids)
    return ids


def read_target(file_name, output_size):
    pil_target = Image.open(file_name)
    if pil_target.size[0] != output_size:
        pil_target = transforms.Resize((output_size, output_size), interpolation=Image.BILINEAR)(pil_target)
    img_numpy = np.asarray(pil_target)
    TARGET = 2.0 * transforms.ToTensor()(img_numpy.astype(np.float32)) / 255.0 - 1.0
    return TARGET


def load_coeffs(input_file_name):
    file = open(input_file_name, "r")
    coeffs = [float(line) for line in file]
    file.close()
    return coeffs


def load_landmarks(file_name):
    landmarks = []
    file = open(file_name, 'r')
    for line in file:
        s1 = line.split(' ')
        landmarks.append([float(s1[0]), float(s1[1])])
    file.close()
    return landmarks


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def tensor2im(input_image, imtype=np.uint8, bs=0):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0)
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[bs].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def show_mask(mask, bs=0):
    image_tensor = mask.data
    image_tensor = image_tensor[bs:bs+1, ...].cpu()
    mask_image = torch.ones(image_tensor.shape, dtype=torch.float32)
    mask_image = torch.where(image_tensor, torch.ones_like(mask_image), torch.zeros_like(mask_image))
    mask_image = mask_image.cpu().squeeze(0).numpy()
    mask_image = mask_image * 255
    mask_image = mask_image.astype(np.uint8)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    return mask_image


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def save_coeffs(output_file_name, coeffs):
    with open(output_file_name, "w") as f:
        for coeff in coeffs:
            f.write(str(coeff) + "\n")


def save_landmarks(output_file_name, landmarks):
    with open(output_file_name, "w") as f:
        for landmark in landmarks:
            f.write(str(landmark[0]) + " " + str(landmark[1]) + "\n")
