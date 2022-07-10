from importlib.resources import path
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop


class dataload():
    def __init__(self) -> None:
        pass

    def process(self):
        flag = torch.cuda.is_available()
        print(flag)

        ngpu = 0
        # Decide which device we want to run on
        device = torch.device("cuda:0" if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(device)

        torch_resize = CenterCrop((224, 224))

        path2 = "/Users/nianhua/Nutstore Files/我的坚果云/jupyter/machine learning lesson/Final/Final/new_valdataset/"
        dbtype_list = os.listdir(path2)
        dbtype_list.pop(dbtype_list.index('.DS_Store'))
        valid_set_1 = []
        valid_label = []
        valid_set_2 = []

        for i in range(len(dbtype_list)):
            dbtype_list[i] = int(dbtype_list[i])
        dbtype_list.sort()
        for i in range(len(dbtype_list)):
            dbtype_list[i] = str(dbtype_list[i])
        print(dbtype_list)

        for i in tqdm(range(len(dbtype_list))):
            pic_name = os.listdir(path2+dbtype_list[i])

            valid_set_1.append((cv2.imread(
                path2+dbtype_list[i]+"/"+pic_name[0], cv2.IMREAD_GRAYSCALE)))
            valid_set_2.append((cv2.imread(
                path2+dbtype_list[i]+"/"+pic_name[1], cv2.IMREAD_GRAYSCALE)))

            if "true" in dbtype_list[i]:
                valid_label.append(1)
            else:
                valid_label.append(0)
        valid_set_1 = np.array(valid_set_1)
        valid_set_2 = np.array(valid_set_2)

        def make_data(train_set, train_label, sets=True):

            for i in tqdm(range(len(train_set))):
                train_set[i] = cv2.equalizeHist(
                    np.array(train_set[i]).astype(np.uint8))
                #train_set[i] = scale(train_set[i])
                #x_train[i] = Gabor_blur(x_train[i], (25, 25), 2.0, 25, 3.7, 1.5, 0)
            train_set = torch.tensor(train_set, dtype=torch.float32)
            train_set = torch_resize(train_set)
            if sets:
                train_dataset = TensorDataset(
                    train_set.reshape(-1, 1, 224, 224), torch.tensor(train_label))  # 对两个列表进行压缩后作为训练集
                TrainLoader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 加载训练集
                return TrainLoader
            else:
                return train_set

        def make_data2(train_set, train_label, sets=True):
            res = []
            for i in tqdm(range(len(train_set))):
                temp = torch_resize(torch.tensor(train_set[i][150:, :]))
                temp = np.array(temp)
                res.append(temp)
                #res.append(cv2.equalizeHist(np.array(temp).astype (np.uint8)))
                #train_set[i] = scale(train_set[i])
                #x_train[i] = Gabor_blur(x_train[i], (25, 25), 2.0, 25, 3.7, 1.5, 0)
            res = np.array(res)
            train_set = torch.tensor(res, dtype=torch.float32)
            if sets:
                train_dataset = TensorDataset(
                    train_set.reshape(-1, 1, 224, 224), torch.tensor(train_label))  # 对两个列表进行压缩后作为训练集
                TrainLoader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 加载训练集
                return TrainLoader
            else:
                return train_set
        BATCH_SIZE = 32

        valid_set_1 = make_data2(valid_set_1, valid_label, False)
        valid_set_2 = make_data2(valid_set_2, valid_label, False)

        print("Data Loaded!")

        return valid_set_1, valid_set_2
