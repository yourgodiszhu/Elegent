import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torch.utils.tensorboard import SummaryWriter

"""1.构建神经网络"""


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 调整维度的卷积层k=1*1，，残差边使用
        # self.downsample 可以将输入张量 x 进行下采样和通道调整，以适应残差块的输出通道
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 添加倒残差结构
        self.reverse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.reverse_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 初始数据定义一个变量
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 加上残差边
        out += identity
        out = self.relu(out)
        # 应用倒残差结构
        out = self.reverse_conv(out)
        out = self.reverse_bn(out)
        out += self.downsample(x)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    # 定义第一个卷积层和池化层
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    # 定义残差块, block残差快的个数
    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        # 打包残差快添加到layer中
        layer.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layer.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layer)  # *将layer解码出来

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpooling(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

"""2.定义数据集"""


class Hehua_Zhizihua_Dataset(Dataset):
    def __init__(self, root, train=True):
        super().__init__()

        self.dataset = []
        train_or_test = "train" if train else "test"
        path = f"{root}\\{train_or_test}"
        for label in os.listdir(path):
            img_path = f"{path}\\{label}"
            for img_name in os.listdir(img_path):
                img = f"{img_path}\\{img_name}"
                self.dataset.append((img, label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = cv2.imread(data[0], 1)
        # video = video.reshape(-1)
        img = img.transpose(2, 0, 1)
        img = img / 255
        # 创建one_hot编码
        one_hot = np.zeros(5)
        one_hot[int(data[1])] = 1
        return np.float32(img), np.float32(one_hot)


"""3.训练和测试"""
class Trainer:
    def __init__(self):
        self.net = ResNet18()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.train_data = Hehua_Zhizihua_Dataset("data", train=True)
        self.train_loader = DataLoader(self.train_data, batch_size=100, shuffle=True)
        self.test_data = Hehua_Zhizihua_Dataset("data", train=False)
        self.test_loader = DataLoader(self.test_data, batch_size=100, shuffle=True)
        self.opt = torch.optim.Adam(self.net.parameters())

        # 动量（momentum），常见的取值范围在0.0到1.0之间，增加动量有助于在损失函数中跳出局部最优点,参与计算参数更新的方向和幅度，可以加速模型训练过程
        # self.opt = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        # self.criterion = nn.MSELoss()  # 均方差损失函数
        # self.criterion = nn.CrossEntropyLoss()
        self.summerWriter = SummaryWriter("logs")

    def train(self):
        for epoch in range(1, 200):
            sum_loss = 0.0
            # 分批次batch_size训练
            for i, (img, label) in enumerate(self.train_loader):
                self.net.train()
                img, label = img.to(self.device), label.to(self.device)
                # 将数据放入网络，前向计算，得到结果
                out = self.net(img)
                # 标签与计算结果之间求损失
                # 计算均方误差损失
                # loss = self.criterion(label, out)
                loss = nn.CrossEntropyLoss()(out, torch.argmax(label, dim=1))
                # 损失，优化器，反向跟新
                self.opt.zero_grad()  # 清空梯度
                loss.backward()  # 梯度跟新
                self.opt.step()
                sum_loss = sum_loss + loss.item()
            avg_loss = sum_loss / len(self.train_loader)
            # 参数：标题、Y轴的值、X轴的值
            self.summerWriter.add_scalar("训练损失", avg_loss, epoch)
            print(f"第{epoch}轮的损失是{avg_loss}", end='\t')
            # 在每个训练轮次结束后计算和输出测试精度
            test_acc = self.test()
            self.summerWriter.add_scalar("测试精度", test_acc, epoch)
            # 保存权重文件
            params = "params"
            if not os.path.exists(params):
                os.makedirs(params)
            torch.save(self.net.state_dict(), os.path.join(params, f"{epoch}.pt"))

    def test(self):
        self.net.eval()  # 将网络切换到测试模式
        sum_score = 0.0
        # 遍历测试数据集的每个批次
        for i, (img, label) in enumerate(self.test_loader):
            img, label = img.to(self.device), label.to(self.device)
            out = self.net(img)
            # out和label作比较，最大值的索引，比较eq，int()，sum()
            x = torch.argmax(out, dim=1)
            y = torch.argmax(label, dim=1)

            # 计算预测准确率
            score = torch.sum(torch.eq(x, y).float())
            sum_score += score.item()
        avg_score = sum_score / len(self.test_loader)
        print(f"测试精度: {avg_score}")
        return avg_score


# 在主函数中调用预测函数
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
