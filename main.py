from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from time import time
import torch
from MyVggModel import MyVggModel
from DataLoad import dataload
from train import train
from skimage.filters import gabor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
import pandas as pd

BATCH_SIZE = 32
EPOCHS = 20  # 设置超参

valid_set_1, valid_set_2 = dataload().process()  # 创建数据


model = MyVggModel()  # 读取模型结构
model.load_state_dict(torch.load(
    "Final/Vgg16WithAttention_real_6.24_spe.pth", map_location=torch.device('cpu')))  # 读取模型参数


model.eval()

cor = 0
wro_acc = 0
wro_ref = 0
ref = 0
real = []
predict = []
suss = []
fpr_list = []
tpr_list = []
auc_score = []
model.eval()

#theta = 0.6
real = []
predict = []
s = 0
s_list = []
s_num = 0
d = 0
d_num = 0
d_list = []
simi_list = []


for i in tqdm(range(0, len(valid_set_1))):

    simi = np.corrcoef(model(train.single_process(valid_set_1[i])).cpu().detach().numpy(
    ), model(train.single_process(valid_set_2[i])).cpu().detach().numpy())[0][-1]
    simi_list.append(simi)
    if simi > 0.977:  # 0.96
        predict.append(1)
    else:
        predict.append(0)

#fpr, tpr, thersholds = roc_curve(real, predict)
# fpr_list.append(fpr)
# tpr_list.append(tpr)
#auc_score.append(auc(fpr, tpr))
# simi_list.sort()
# print(simi_list)
pd.DataFrame(predict).to_csv("2019191138.csv", sep="\t", index=False)
simi_list = np.array(simi_list)
kmeans = KMeans(n_clusters=2, random_state=0).fit(simi_list.reshape(-1, 1))
print(simi_list)
print(predict)
print(kmeans.cluster_centers_)
