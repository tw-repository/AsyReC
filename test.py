"""
test.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataset import Dataset
import warnings
import pickle as pkl

warnings.filterwarnings('ignore')

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--mode', dest="mode", default="NoXi", type=str,
                    help='NoXi or UDIVA')

parser.add_argument('--k_fold', dest='k_fold', default=1, type=int, metavar='N',
                    help='K-fold cross validation.')

parser.add_argument('--person', dest='person', default='A', type=str,
                    help='NoXi training for A or B.')

parser.add_argument('--total_frames', dest='total_frames', default=16, type=int, metavar='N',
                    help='Frame numbers for training and testing.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1)')

args = parser.parse_args()

for k, v in sorted(vars(args).items()):
    print(k, ': ', v)

if args.mode == "NoXi":
    data_dir = r'/your_path/NoXi_video_5_downSample'
    audio_dir = r'/your_path/NoXi_audio_clip.pkl'
    text_dir = r"/your_path/NoXi_text_clip.pkl"
    if args.k_fold == 1:
        train_list = r"/your_path/NoXi_train_bbox1.pkl"
        test_list = r"/your_path/NoXi_test_bbox1.pkl"
    elif args.k_fold == 2:
        train_list = r"/your_path/NoXi_train_bbox2.pkl"
        test_list = r"/your_path/NoXi_test_bbox2.pkl"
    elif args.k_fold == 3:
        train_list = r"/your_path/NoXi_train_bbox3.pkl"
        test_list = r"/your_path/NoXi_test_bbox3.pkl"
    labels = ['Stranger', 'Acquaintance', 'Friend', 'Very good friend']
    num_class = len(labels)

elif args.mode == "UDIVA":
    data_dir = r"/your_path/UDIVA_video_5_downSample"
    audio_dir = r'/your_path/UDIVA_audio_clip.pkl'
    text_dir = r"/your_path/UDIVA_text_clip.pkl"
    if args.k_fold == 1:
        train_list = r"/your_path/UDIVA_train_bbox1.pkl"
        test_list = r"/your_path/UDIVA_test_bbox1.pkl"
    elif args.k_fold == 2:
        train_list = r"/your_path/UDIVA_train_bbox2.pkl"
        test_list = r"/your_path/UDIVA_test_bbox2.pkl"
    elif args.k_fold == 3:
        train_list = r"/your_path/UDIVA_train_bbox3.pkl"
        test_list = r"/your_path/UDIVA_test_bbox3.pkl"
    labels = ['Known', 'Unknown']
    num_class = len(labels)


def vg_collate(data):
    face_A = []
    face_B = []
    person_A = []
    person_B = []
    relation_A = []
    relation_B = []
    audio_A = []
    audio_B = []
    text_A = []
    text_B = []
    position = []
    video_length = []

    for d in data:
        (face_a, person_a, audio_a, text_a, rel_a,
         face_b, person_b, audio_b, text_b, rel_b, pos, length) = d
        face_A.append(face_a)
        face_B.append(face_b)
        person_A.append(person_a)
        person_B.append(person_b)
        relation_A.append(rel_a)
        relation_B.append(rel_b)
        audio_A.append(audio_a['input_values'])
        audio_B.append(audio_b['input_values'])
        text_A.append(text_a)
        text_B.append(text_b)
        position.append(pos)
        video_length.append(length)

    face_A = torch.stack(face_A, 0).transpose(1, 2)
    face_B = torch.stack(face_B, 0).transpose(1, 2)
    person_A = torch.stack(person_A, 0).transpose(1, 2)
    person_B = torch.stack(person_B, 0).transpose(1, 2)
    audio_A = torch.stack(audio_A, 0)
    audio_B = torch.stack(audio_B, 0)
    relation_A = torch.tensor(relation_A).squeeze(1)
    relation_B = torch.tensor(relation_B).squeeze(1)
    position = torch.tensor(position)
    video_length = torch.tensor(video_length)

    return (face_A, person_A, audio_A, text_A, relation_A,
            face_B, person_B, audio_B, text_B, relation_B, position, video_length)


def get_test_set(data_dir, test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    test_set = Dataset(data_dir, test_list, audio_dir, text_dir, transform, total_frames=args.total_frames)
    test_loader = DataLoader(
        dataset=test_set,
        num_workers=args.workers,
        collate_fn=vg_collate,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return test_loader


def validate(val_loader, model):
    model.eval()
    true_A, true_B, true = [], [], []
    p_A, p_B, p = [], [], []
    for batch_data in tqdm(val_loader):
        (face_A, person_A, audio_A, text_A, relation_A,
         face_B, person_B, audio_B, text_B, relation_B, position, video_length) = batch_data

        relation_A = relation_A.cuda()
        relation_B = relation_B.cuda()
        position = position.cuda()
        video_length = video_length.cuda()

        face_A = torch.autograd.Variable(face_A).cuda()
        person_A = torch.autograd.Variable(person_A).cuda()
        face_B = torch.autograd.Variable(face_B).cuda()
        person_B = torch.autograd.Variable(person_B).cuda()

        with torch.no_grad():
            output = model(face_A, person_A, audio_A, text_A, face_B, person_B, audio_B, text_B, position, video_length)

        if args.mode == 'NoXi':
            if args.person == 'A':
                target_var_A = torch.autograd.Variable(relation_A)
                output_f_A = F.softmax(output, dim=1)
                output_np_A = output_f_A.data.cpu().numpy()
                pre_A = np.argmax(output_np_A, 1).flatten()
                t_A = target_var_A.flatten().data.cpu().numpy()
                true_A.append(t_A)
                p_A.append(pre_A)

            elif args.person == 'B':
                target_var_B = torch.autograd.Variable(relation_B)
                output_f_B = F.softmax(output, dim=1)
                output_np_B = output_f_B.data.cpu().numpy()
                pre_B = np.argmax(output_np_B, 1).flatten()
                t_B = target_var_B.flatten().data.cpu().numpy()
                true_B.append(t_B)
                p_B.append(pre_B)

        elif args.mode == 'UDIVA':
            target_var = torch.autograd.Variable(relation_A)
            output_f = F.softmax(output, dim=1)
            output_np = output_f.data.cpu().numpy()
            pre = np.argmax(output_np, 1).flatten()
            t = target_var.flatten().data.cpu().numpy()
            true.append(t)
            p.append(pre)

    if args.mode == 'NoXi':
        if args.person == 'A':
            uar_A, recall_A = calculate_auc(np.concatenate(true_A, axis=0), np.concatenate(p_A, axis=0), num_class)
            return uar_A, recall_A
        elif args.person == 'B':
            uar_B, recall_B = calculate_auc(np.concatenate(true_B, axis=0), np.concatenate(p_B, axis=0), num_class)
            return uar_B, recall_B
    elif args.mode == 'UDIVA':
        uar, recall = calculate_auc(np.concatenate(true, axis=0), np.concatenate(p, axis=0), num_class)
        return uar, recall


def calculate_auc(true_labels, pred_labels, num_classes):
    classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    if len(classes) < 2 and len(pred_classes) < 2:
        micro_auc = np.nan
        auc_per_class = {i: np.nan for i in range(num_classes)}
        sensitivity_dict = {i: np.nan for i in range(num_classes)}
        uar = np.nan
    else:
        lb_true = label_binarize(true_labels, classes=np.arange(num_classes), sparse_output=False)
        lb_pred = label_binarize(pred_labels, classes=np.arange(num_classes), sparse_output=False)

        if num_classes == 2:
            lb_true = np.hstack((lb_true, 1 - lb_true))
            lb_pred = np.hstack((lb_pred, 1 - lb_pred))

        sensitivity = recall_score(true_labels, pred_labels, labels=np.arange(num_classes), average=None,
                                   zero_division=0)

        sensitivity_dict = {i: sens for i, sens in enumerate(sensitivity)}
        uar = np.nanmean([s for s in sensitivity if not np.isnan(s)])
        auc_per_class = {}
        for i in range(num_classes):
            if len(np.unique(lb_true[:, i])) > 1: 
                auc_per_class[i] = roc_auc_score(lb_true[:, i], lb_pred[:, i])
            else:
                auc_per_class[i] = np.nan

        precision_micro, recall_micro, _ = precision_recall_curve(lb_true.ravel(), lb_pred.ravel())
        micro_auc = auc(recall_micro, precision_micro)

    return uar, sensitivity_dict


def init_network(num_class, maskType, interMask, spatMask, spatMaskPos):
    from network import network
    model = network(num_class)
    return model


if __name__ == '__main__':
    path = '/your_trained_model_path'
    if args.mode == 'NoXi':
        model_name = 'your_model_name'
        model_file = os.path.join(path, model_name)
        model = init_network(num_class, args.maskType, args.interMask, args.spatMask, args.spatMaskPos)
        trained_model = torch.load(model_file)
        val_loader = get_test_set(data_dir=data_dir, test_list=test_list)
        model_dict = model.state_dict()
        trained_model = {k.replace('module.', ''): v for k, v in
                          trained_model['state_dict'].items()}
        trained_model_dict1 = {k: v for k, v in trained_model.items() if k in model_dict}
        model_dict.update(trained_model_dict1)
        model.load_state_dict(model_dict)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        uar, recall = validate(val_loader, model)
        print("UAR = {}, recall = {}".format(uar, recall))

    elif args.mode == 'UDIVA':
        model_name = 'your_model_name'
        model_file = os.path.join(path, model_name)
        val_loader = get_test_set(data_dir=data_dir, test_list=test_list)
        model = init_network(num_class, args.maskType, args.interMask, args.spatMask, args.spatMaskPos)
        trained_model = torch.load(model_file)
        model_dict = model.state_dict()
        trained_model = {k.replace('module.', ''): v for k, v in
                         trained_model['state_dict'].items()}
        trained_model_dict = {k: v for k, v in trained_model.items() if k in model_dict}
        model_dict.update(trained_model_dict)
        model.load_state_dict(model_dict)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model.to(device)

        uar, recall = validate(val_loader, model)
        print("UAR = {}, recall = {}".format(uar, recall))

