'''
该模块主要用于设计数据集的读取
使用的变换（例如：调整图像大小、随机旋转、翻转、裁切等、此处只有调整大小和随机翻转）
transforms.ToTensor(),
最后这步转化成tensor形式的数据类型
'''

'''
关于这部分的代码，完全可以使用GPT生成
使用方法：
告诉GPT你的数据组织方式，例如对话如下：

“我的数据集根目录下包含train和val两个子文件夹，
每个文件夹下都包含cat和dog两个子文件夹，均存放jpg格式的图片，
你可以帮我使用pytorch编写Mydatasets吗，要求继承自torch自带的Dataset，
重写__init__、__len__、__getitem__这三个函数，
要求返回一个三通道大小为224的图片张量和onehot编码标签”

'''

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

# 定义图像的变换
# MyTransform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

MyTransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
torch中自带一个数据集读取的类,也就是开头这句代码中的Dataset，如下
from torch.utils.data import Dataset
但是为了适应不同的数据集，这个类并不完善，我们需要重写编写该类的：
初始化函数__init_
返回张量和标签的函数__getitem__
以及返回数据集长度的函数__len__
这样我们才可以后续使用
'''
# 自定义数据集类
class Mydatasets(Dataset):
    def __init__(self, root_dir, transform=MyTransform):
        """
        Args:
            root_dir (string): 数据集的根目录路径。
            transform (callable, optional): 可选的转换操作（例如：数据增强）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_dict = {'Cat': 0, 'Dog': 1}

        # 遍历目录以获取图像路径和标签
        for label in ['Cat', 'Dog']:
            folder_path = os.path.join(root_dir, label)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.data.append(img_path)
                self.labels.append(self.label_dict[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # 读取图像并转换为RGB
        image = Image.open(img_path).convert('RGB')

        # 图像变换
        if self.transform:
            image = self.transform(image)

        # One-hot编码
        one_hot_label = torch.zeros(2)
        one_hot_label[label] = 1

        return image, one_hot_label

# 定义用于显示图像的函数
def show_image(tensor, title=None):
    """显示一个图像张量"""
    # 逆归一化处理，将图像数据从[-1, 1]恢复到[0, 1]范围
    image = tensor * 0.5 + 0.5

    # 转换张量格式以便显示
    image = image.permute(1, 2, 0).numpy()  # 从 CxHxW 转换为 HxWxC

    # 显示图像
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# 测试主函数
def main():
    # 实例化数据集
    train_dataset = Mydatasets(root_dir=r"D:\pythonItem\catdog\data_set\catsdogs\val")

    # 测试数据集读取
    print(f"数据集中样本数量: {len(train_dataset)}")

    # 初始化类别计数器
    cat_count = 0
    dog_count = 0

    # 遍历整个数据集并统计每个类别的数量
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        # 假设label[0] == 1表示猫，label[0] == 0表示狗
        if label[0] == 1:
            cat_count += 1
        else:
            dog_count += 1

    # 输出每个类别的数量
    print(f"猫的数量: {cat_count}")
    print(f"狗的数量: {dog_count}")

    # 显示一些图像和它们的标签
    for i in range(200,399):
        image, label = train_dataset[i]
        print("数据张量和标签的形状：", image.shape, label.shape)
        show_image(image, title=f"Label: {'Cat' if label[0] == 1 else 'Dog'}")
        print(label)

# 确保主函数只在被调用时执行，而避免该模块被其他模块import时也执行
if __name__ == "__main__":
    main()


