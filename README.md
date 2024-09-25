
猫狗分类项目

该项作为模板目用于让初学者进行参考，代码中均给出了关键部分的注释。

项目结构

├── mymodel.py        # 模型定义（AlexNet）
├── mydatasets.py     # 数据集加载类
├── train.py          # 模型训练脚本
├── val.py            # 模型验证脚本
├── demo.py           # 模型推理脚本
└── README.md         # 项目说明文件

依赖项

在运行该项目之前，请确保已安装以下依赖项：

- Python 3.x
- PyTorch
- torchvision
- tqdm
- scikit-learn
- Pillow
- seaborn
- matplotlib

你可以使用 requirements.txt 或手动安装依赖项：

pip install torch torchvision tqdm scikit-learn pillow seaborn matplotlib

数据集准备

项目使用的是二分类数据集（猫和狗）。请将数据集按照以下结构组织：

data_set/
    └── catsdogs/
        ├── train/
        │   ├── Cat/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── Dog/
        │       ├── img1.jpg
        │       ├── img2.jpg
        │       └── ...
        └── val/
            ├── Cat/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            └── Dog/
                ├── img1.jpg
                ├── img2.jpg
                └── ...

使用说明

1. 模型训练

train.py 用于训练模型。如果存在之前训练好的模型权重，脚本会自动加载权重并继续训练。否则，将从头开始训练。
训练过程中的模型权重会定期保存为 AlexNet.pth。

2. 模型验证

val.py 用于加载训练好的模型，并在验证集上测试模型的准确率，同时绘制混淆矩阵。

python val.py

验证脚本会输出模型的准确率，并使用混淆矩阵评估分类效果。

3. 图像分类推理

demo.py 用于加载训练好的模型并对单张图像进行分类。用户需提供 JPG 图像路径，模型将输出该图像属于 "Cat" 还是 "Dog"。
运行后输入图像路径，即可得到分类结果：

请输入要分类的图像路径 (jpg格式): path_to_image.jpg
分类结果: Cat

自定义组件说明

1. mymodel.py
该文件定义了 AlexNet 模型架构以及一个简易的CNN便于初学者理解网络搭建过程。

2. mydatasets.py
该文件定义了 Mydatasets 类，用于加载自定义的猫狗图像数据集。图像经过数据增强和归一化处理，便于输入到模型中进行训练或推理。

3. train.py
该脚本用于训练模型，支持从已有权重继续训练，避免每次重新训练。模型权重保存在 AlexNet.pth 文件中。

4. val.py
该脚本用于在验证集上评估模型的表现，输出模型的准确率，并绘制混淆矩阵。

5. demo.py
该脚本用于加载训练好的模型，针对单张图像进行推理，返回图像是猫还是狗的分类结果。


