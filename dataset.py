"""
data processing and loading
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle as pkl
import random
from pydub import AudioSegment
import os
import numpy as np
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import warnings
import math
from transformers import AutoTokenizer, AutoModel


warnings.filterwarnings('ignore')
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


def group_boxes_by_category(image_names, face_boxes, person_boxes, relation_classes, data_name, total_frame=50):
    video_name = {}

    if data_name == 'NoXi':
        num = 2
    elif data_name == 'UDIVA':
        num = 3

    for i, filename in enumerate(image_names):
        category_name = '_'.join(filename.split('_')[:num])

        if category_name not in video_name:
            video_name[category_name] = []

        video_name[category_name].append({
            'image_name': filename,
            'face_box': face_boxes[filename],
            'person_box': person_boxes[filename],
            'relation_class': relation_classes[filename]
        })

    final_grouped_data = {}
    group_counter = {}

    for cat, items in video_name.items():
        for idx in range(0, len(items), total_frame):
            sub_group = items[idx:idx + total_frame]
            if len(sub_group) == total_frame:
                if cat not in group_counter:
                    group_counter[cat] = 1
                else:
                    group_counter[cat] += 1

                new_key = f"{cat}_{group_counter[cat]}"
                final_grouped_data[new_key] = {
                    'image_name': [item['image_name'] for item in sub_group],
                    'face_boxes': [item['face_box'] for item in sub_group],
                    'person_boxes': [item['person_box'] for item in sub_group],
                    'relation_classes': [item['relation_class'] for item in sub_group]
                }

    return final_grouped_data, {k: len(v)//50 for k, v in video_name.items()}


def sync_dictionaries(dict1, dict2):

    def get_key_parts(key):
        parts = key.split('_')
        return (parts[0], parts[-1])

    keys1 = set(get_key_parts(key) for key in dict1.keys())
    keys2 = set(get_key_parts(key) for key in dict2.keys())

    common_keys = keys1 & keys2

    synchronized_dict1 = {key: dict1[key] for key in dict1 if get_key_parts(key) in common_keys}
    synchronized_dict2 = {key: dict2[key] for key in dict2 if get_key_parts(key) in common_keys}

    return synchronized_dict1, synchronized_dict2


def alignInfo(input_dict, data_name):
    max_segments = {}

    for key in input_dict.keys():
        if data_name == 'NoXi':
            video_name, _, segment_number = key.split('_')
        elif data_name == 'UDIVA':
            parts = key.split('_')
            video_name = parts[0] + '_' + parts[1]
            segment_number = parts[3]
        segment_number = int(segment_number)

        if video_name not in max_segments:
            max_segments[video_name] = {'max_segment': segment_number, 'key': key}
        else:
            if segment_number > max_segments[video_name]['max_segment']:
                max_segments[video_name] = {'max_segment': segment_number, 'key': key}

    for video_name in max_segments:
        del input_dict[max_segments[video_name]['key']]

    return input_dict


def select_frames(input_list, num_samples):
    interval = math.ceil(len(input_list) / num_samples)
    selected_elements = [input_list[min(i * interval, len(input_list) - 1)] for i in range(num_samples)]
    return selected_elements


def clipLevelData(input_dict, transform, image_dir, total_frames):
    image_list = input_dict['image_name']
    face_box_list = input_dict['face_boxes']
    person_box_list = input_dict['person_boxes']
    face_crop_clip, body_crop_clip = image_crop(select_frames(image_list, total_frames),
                                                image_dir, face_box_list, person_box_list, transform)

    return face_crop_clip, body_crop_clip, input_dict['relation_classes'][0]


def image_crop(image_name_list, image_dir, face_box_list, person_box_list, transform):
    cropped_face, cropped_body = [], []
    for image_name, face_box, person_box in zip(image_name_list, face_box_list, person_box_list):
        img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')

        face_tensor = transform(img.crop((face_box[0], face_box[1], face_box[2], face_box[3])))
        person_tensor = transform(img.crop((person_box[0], person_box[1], person_box[2], person_box[3])))

        cropped_face.append(face_tensor)
        cropped_body.append(person_tensor)

    return torch.stack(cropped_face, 0), torch.stack(cropped_body, 0)


class Dataset(Dataset):
    def __init__(self, image_dir, label_dir, audio_dir, text_dir, transform, total_frames):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.transform = transform
        self.total_frames = total_frames

        parts = self.image_dir.split('/')
        subfolder_name = parts[5]
        subfolder_parts = subfolder_name.split('_')
        self.data_name = subfolder_parts[0]

        with open(self.label_dir, "rb") as f:
            data = pkl.load(f, encoding='latin1')
        with open(self.audio_dir, "rb") as f:
            self.data_audio = pkl.load(f, encoding='latin1')
        with open(self.text_dir, "rb") as f:
            self.data_text = pkl.load(f, encoding='latin1')

        self.img_names_A = []
        self.img_names_B = []
        self.face_A_box = {}
        self.face_B_box = {}
        self.person_A_box = {}
        self.person_B_box = {}
        self.relation_classes_A = {}
        self.relation_classes_B = {}

        for item in data:
            frame_category = item['frame'].split('_')[1] if self.data_name == 'NoXi' else item['frame'].split('_')[2]

            if frame_category == '0':
                self.img_names_A.append(item['frame'])
                self.face_A_box[item['frame']] = item['face'][0]
                self.person_A_box[item['frame']] = item['body']
                self.relation_classes_A[item['frame']] = item['relation']
            else:
                self.img_names_B.append(item['frame'])
                self.face_B_box[item['frame']] = item['face'][0]
                self.person_B_box[item['frame']] = item['body']
                self.relation_classes_B[item['frame']] = item['relation']

        self.name_A, self.video_name_A = group_boxes_by_category(
            self.img_names_A, self.face_A_box, self.person_A_box, self.relation_classes_A, self.data_name)

        self.name_B, self.video_name_B = group_boxes_by_category(
            self.img_names_B, self.face_B_box, self.person_B_box, self.relation_classes_B, self.data_name)

        self.alignedName_A, self.alignedName_B = sync_dictionaries(self.name_A, self.name_B)

        self.alignedInfo_A = alignInfo(self.alignedName_A, self.data_name)
        self.alignedInfo_B = alignInfo(self.alignedName_B, self.data_name)

    def __len__(self):
        return len(self.alignedInfo_A)

    def __getitem__(self, idx):

        Video_name_A = list(self.alignedInfo_A.keys())[idx]
        Video_name_B = list(self.alignedInfo_B.keys())[idx]

        position = int(Video_name_A.split('_')[-1])

        if self.data_name == 'NoXi':
            num = 2
        else:
            num = 3
        category_name = '_'.join(Video_name_A.split('_')[:num])
        video_len = self.video_name_A[category_name]

        name_dict_A = self.alignedName_A[Video_name_A]
        name_dict_B = self.alignedName_B[Video_name_B]

        audio_A = self.data_audio[Video_name_A]
        audio_B = self.data_audio[Video_name_B]

        text_A = self.data_text[Video_name_A]
        text_B = self.data_text[Video_name_B]

        face_A, body_A, rel_A = clipLevelData(name_dict_A, self.transform, self.image_dir, self.total_frames)

        face_B, body_B, rel_B = clipLevelData(name_dict_B, self.transform, self.image_dir, self.total_frames)

        return face_A, body_A, audio_A, text_A, rel_A, face_B, body_B, audio_B, text_B, rel_B, position, video_len
