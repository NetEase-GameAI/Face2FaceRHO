import os.path
import torch
import numpy as np
from dataset.base_dataset import BaseDataset
from util.util import (
    make_ids,
    read_target,
    load_coeffs,
    load_landmarks
)

from util.landmark_image_generation import LandmarkImageGeneration


class VoxCelebDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot
        video_list_file = self.dataroot + "/list.txt"
        self.video_path = self.dataroot
        self.video_names = []
        with open(video_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.video_names.append(self.video_path + "/" + line)
            person_ids = set()
            for video in self.video_names:
                person_ids.add(os.path.basename(video).split('#')[0])
            self.person_ids = list(person_ids)
            self.person_ids.sort()

        self.landmark_img_generator = LandmarkImageGeneration(self.opt)

        self.total_person_id = len(self.person_ids)
        print('\tnum videos: {}, repeat {} times, total: {}'.format(self.total_person_id, opt.num_repeats,
                                                                    self.total_person_id * opt.num_repeats))

    def __getitem__(self, index):
        idx = index % self.total_person_id
        name = self.person_ids[idx]
        video_name = np.random.choice(self.choose_video_from_person_id(name))
        frame_ids = make_ids(video_name + "/img")

        frame_idx = np.sort(np.random.choice(frame_ids, replace=False, size=2))

        img_dir = video_name + "/img"
        headpose_dir = video_name + "/headpose"
        landmark_dir = video_name + "/landmark"
        mask_dir = video_name + "/mask"

        src_idx = frame_idx[0]
        drv_idx = frame_idx[1]

        src_img = read_target(img_dir + "/" + str(src_idx) + ".jpg", self.opt.output_size)
        drv_img = read_target(img_dir + "/" + str(drv_idx) + ".jpg", self.opt.output_size)

        src_headpose = torch.from_numpy(np.array(load_coeffs(headpose_dir + "/" + str(src_idx) + ".txt"))).float()
        drv_headpose = torch.from_numpy(np.array(load_coeffs(headpose_dir + "/" + str(drv_idx) + ".txt"))).float()

        src_landmark = torch.from_numpy(np.array(load_landmarks(landmark_dir + "/" + str(src_idx) + ".txt"))).float()
        drv_landmark = torch.from_numpy(np.array(load_landmarks(landmark_dir + "/" + str(drv_idx) + ".txt"))).float()

        src_landmark_img = self.draw_landmark_img(src_landmark)
        drv_landmark_img = self.draw_landmark_img(drv_landmark)

        input_data = {
            'src_img': src_img,
            'drv_img': drv_img,
            'src_headpose': src_headpose,
            'drv_headpose': drv_headpose,
            'src_landmark_img': src_landmark_img,
            'drv_landmark_img': drv_landmark_img,
        }
        if self.opt.emphasize_face_area:
            drv_face_mask = read_target(mask_dir + "/" + str(drv_idx) + ".png", self.opt.output_size)
            input_data['drv_face_mask'] = drv_face_mask.squeeze(0)
        return input_data

    def choose_video_from_person_id(self, name):
        names = []
        for video_name in self.video_names:
            if name in video_name:
                names.append(video_name.strip())
        return names

    def draw_landmark_img(self, landmarks):
        landmark_imgs = self.landmark_img_generator.generate_landmark_img(landmarks)
        return landmark_imgs

    def __len__(self):
        return self.total_person_id * self.opt.num_repeats

    def name(self):
        return 'VoxCelebDataset'


