import torch
from torch import nn
# 该模块主要定义神经网络的结构

class AlexNet(nn.Module):
    def __init__(self, num_classes=1, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
# ================================================
'''
以上是ALEXNET的基本结构，其中包含了很多新的写法，例如 Sequential(...)
Sequential 用于组合多个层，这样避免在forward中重复编写大量代码，可以直接使用 Sequential 一次性代表多个层
Sequential 也存在一个严重的缺点，如果其中的某个层出错，那么控制台报错信息只能指向这个 Sequential 出错，而无法指向其中具体出错的层
这个缺点在调试时候会导致许多麻烦

同时相比下面的MyCNN，AlexNet中加入了 Dropout 层来减少过拟合现象，具体该层的原理可以查资料
同时其构造函数（init在面向对象编程中被称构造函数或者构造方法，本质是初始化函数）
同时其构造函数使用 num_classes 变量灵活控制分类类别，这个变量决定了最后一层的输出节点数量
'''
# ================================================


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第一个卷积层，输入通道3，输出通道32，核大小3，步长1，边缘扩充方式1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个池化层，最大池化，核大小2，步长2（等于每四个格子取最大）
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个最大池化层

        self.fc1 = nn.Linear(200704, 512)
        self.fc2 = nn.Linear(512, 2)
        '''
        全连接层，假设输入图片大小为224x224彩色图，则输入张量大小为224x224x3
        第一个卷积层：224x224x3 ==> 224x224x32
        第一个池化层：224x224x32 ==> 112x112x32
        第二个卷积层：112x112x32 ==> 
        第二个池化层：112x112x64 == > 56x56x64
        于是全连接层的输入维度为 56x56x64 = 200704
        '''

        '''
        forward是nn.Module的前向传播函数，也需要继承后重写写
        x是输入数据流，根据上述的构造函数中的每一个层，调用即可，这里的顺序是实际的数据传播顺序
        上述的代码只是待使用的神经网络的“组件”，顺序不重要
        '''
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 200704)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        x = nn.functional.relu(x)
        return x

if __name__ == "__main__":
    '''
    初始化一个随机模型和一组（一个batchsize）的输入张量
    测试输出张量是否正确
    此过程可以检查数据在网络的前向传播中是否一切正常
    '''
    MyCNN = AlexNet(num_classes=2)
    batchsize=8
    input_tensor = torch.rand([batchsize,3,224,224])
    print("输入张量形状：",input_tensor.shape)
    output_tensor = MyCNN(input_tensor)
    print("输出张量形状：",output_tensor.shape)







