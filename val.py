import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from MyDatasets import Mydatasets
from MyModels import AlexNet

'''
此为新版的val，不再用于输出指定样本的类别，而是绘制混淆矩阵，计算性能参数等
对于输出指定样本的类别的需求，一半会编写一个单独的demo.py
'''

def evaluate_model(data_dir, model_path='AlexNet.pth', batch_size=32):
    # 设备配置：使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化验证数据集
    val_dataset = Mydatasets(root_dir=os.path.join(data_dir, "val"))

    # 创建 DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型并加载权重
    model = AlexNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model weights from {model_path}")

    # 设置为评估模式
    model.eval()

    # 初始化评价指标
    all_labels = []
    all_preds = []

    # 不计算梯度
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.argmax(labels, dim=1).to(device)  # 将one-hot标签转为类别索引

            # 前向传播
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            # 保存真实标签和预测标签
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, ['Cat', 'Dog'])

def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵的函数"""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    data_dir = './data_set/catsdogs'
    evaluate_model(data_dir=data_dir, model_path='AlexNet.pth')
