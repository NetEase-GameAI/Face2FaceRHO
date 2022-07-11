import cv2
import torchvision.transforms as transforms
import numpy as np


class LandmarkImageGeneration:
    def __init__(self, opt):
        self.output_size = opt.output_size
        self.landmark_img_sizes = [
            int(opt.output_size / 16),
            int(opt.output_size / 8),
            int(opt.output_size / 4),
        ]

        self.facial_part_color = {
            # B G R
            'left_eyebrow': [0, 0, 255],  # red
            'right_eyebrow': [0, 255, 0],  # green
            'left_eye': [255, 0, 0],  # blue
            'right_eye': [255, 255, 0],  # cyan
            'nose': [255, 0, 255],  # purple
            'mouth': [0, 255, 255],  # yellow
            'face_contour': [125, 125, 125],  # gray
        }

        self.facial_part = [
            {
                'left_eyebrow': [],
                'right_eyebrow': [],
                'left_eye': [42],  # 1
                'right_eye': [50],  # 1
                'nose': [30],  # 1
                'mouth': [52, 57],  # 2
                'face_contour': [0, 8, 9],  # 3
            },
            {
                'left_eyebrow': [17, 21],  # 2
                'right_eyebrow': [26, 22],  # 2
                'left_eye': [36, 38, 40, 42, 36],  # 4
                'right_eye': [48, 46, 44, 50, 48],  # 4
                'nose': [[27, 33, 31], [33, 35]],  # 4
                'mouth': [[52, 62, 57, 71, 52], [63], [70]],  # 6
                'face_contour': [0, 4, 8, 13, 9],  # 5
            },
            {
                'left_eyebrow': [17, 18, 19, 20, 21],  # 5
                'right_eyebrow': [26, 25, 24, 23, 22],  # 5
                'left_eye': [36, 37, 38, 39, 40, 41, 42, 43, 36],  # 8
                'right_eye': [48, 47, 46, 45, 44, 51, 50, 48, 48],  # 8
                'nose': [[27, 28, 29, 30, 33], [31, 32, 33], [35, 34, 33]],  # 9
                'mouth': [[52, 54, 53, 62, 58, 59, 57, 65, 66, 71, 69, 68, 52], [55, 56, 63, 61, 60, 64, 70, 67, 55]],  # 20
                'face_contour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 14, 13, 12, 11, 10, 9],  # 17
            }
        ]

    def generate_landmark_img(self, landmarks):
        landmark_imgs = []
        for i in range(len(self.landmark_img_sizes)):
            cur_landmarks = landmarks.clone()
            image_size = self.landmark_img_sizes[i]

            cur_landmarks[:, 0:1] = (cur_landmarks[:, 0:1] + 1) / 2 * (image_size - 1)
            cur_landmarks[:, 1:2] = (cur_landmarks[:, 1:2] + 1) / 2 * (image_size - 1)

            cur_facial_part = self.facial_part[i]
            line_width = 1

            landmark_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['left_eyebrow'], self.facial_part_color['left_eyebrow'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['right_eyebrow'], self.facial_part_color['right_eyebrow'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['left_eye'], self.facial_part_color['left_eye'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['right_eye'], self.facial_part_color['right_eye'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['nose'], self.facial_part_color['nose'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['mouth'], self.facial_part_color['mouth'], line_width)
            self.draw_line(landmark_img, cur_landmarks, cur_facial_part['face_contour'], self.facial_part_color['face_contour'], line_width)
            landmark_img = 2.0 * transforms.ToTensor()(landmark_img.astype(np.float32)) / 255.0 - 1.0

            landmark_imgs.append(landmark_img)
        return landmark_imgs

    @staticmethod
    def draw_line(landmark_map, projected_landmarks, line_ids, line_color, line_width):
        if len(line_ids) == 1: # only single point
            center_x = int(projected_landmarks[line_ids[0], 0])
            center_y = int(projected_landmarks[line_ids[0], 1])
            cv2.circle(landmark_map, (center_x, center_y), line_width, line_color, -1, cv2.LINE_4)
        elif len(line_ids) > 1:
            if isinstance(line_ids[0], list):
                for i in range(len(line_ids)):
                    if len(line_ids[i]) == 1:
                        center_x = int(projected_landmarks[line_ids[i][0], 0])
                        center_y = int(projected_landmarks[line_ids[i][0], 1])
                        cv2.circle(landmark_map, (center_x, center_y), line_width, line_color, -1, cv2.LINE_4)
                    else:
                        for j in range(len(line_ids[i]) - 1):
                            pt1_x = int(projected_landmarks[line_ids[i][j], 0])
                            pt1_y = int(projected_landmarks[line_ids[i][j], 1])
                            pt2_x = int(projected_landmarks[line_ids[i][j + 1], 0])
                            pt2_y = int(projected_landmarks[line_ids[i][j + 1], 1])
                            pt1 = (int(pt1_x), int(pt1_y))
                            pt2 = (int(pt2_x), int(pt2_y))
                            cv2.line(landmark_map, pt1, pt2, line_color, line_width, cv2.LINE_4)
            else:
                for i in range(len(line_ids) - 1):
                    pt1_x = int(projected_landmarks[line_ids[i], 0])
                    pt1_y = int(projected_landmarks[line_ids[i], 1])
                    pt2_x = int(projected_landmarks[line_ids[i+1], 0])
                    pt2_y = int(projected_landmarks[line_ids[i + 1], 1])
                    pt1 = (int(pt1_x), int(pt1_y))
                    pt2 = (int(pt2_x), int(pt2_y))
                    cv2.line(landmark_map, pt1, pt2, line_color, line_width, cv2.LINE_4)



