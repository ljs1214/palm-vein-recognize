import torch.nn as nn
from torchvision import models
from Attention import cbam_block


class MyVggModel(nn.Module):
    def __init__(self, train_flag=True):
        vgg16 = models.vgg16(pretrained=True)
        # 获取VGG16的特征提取层
        vgg = vgg16.features

        # 将vgg16的特征提取层参数冻结
        s = 0
        for param in vgg.parameters():
            s += 1
            param.requires_grad_(False)
            if s > 30:
                break
        super(MyVggModel, self).__init__()
        # 预训练的Vgg16的特征提取层
        self.conv = nn.Conv2d(1, 3, 1)
        self.batch = nn.BatchNorm2d(3)
        self.vgg = vgg
        self.att = cbam_block(512)
        #self.attRead = SpatialAttention()
        # 添加新的全连接层
        if train_flag:
            self.classifier = nn.Sequential(
                #nn.Linear(8192, 25088),
                # nn.Dropout(0.5),
                nn.Linear(7*7*512, 4096),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.Dropout(0.5),
                nn.Linear(4096, 1100)
            )
        else:
            self.classifier = nn.Sequential()

    # 定义网络的前向传播
    def forward(self, x):
        # x = self.attRead(x) * x # 输入空间注意力，但是效果不好
        x = self.conv(x)
        x = self.batch(x)
        x = self.vgg(x)

        x = self.att(x)

        x = x.view(x.size(0), -1)

        output = self.classifier(x)

        return output
