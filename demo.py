import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from MyDatasets import Mydatasets
from MyModels import AlexNet,MyCNN

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 分类标签
class_names = ['Cat', 'Dog']


def load_model(weight_path, device):
    """加载模型及权重"""
    model = AlexNet(num_classes=2).to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()  # 设置为评估模式
        print(f"Model loaded from {weight_path}")
    else:
        print(f"Weight file '{weight_path}' not found.")
        exit(1)

    return model


def predict_image(image_path, model, device):
    """对指定的图像进行分类"""
    # 打开图像
    img = Image.open(image_path).convert('RGB')

    # 应用预处理
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度

    # 前向传播
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

    # 返回预测结果
    return class_names[preds.item()]


if __name__ == "__main__":
    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型权重路径
    weight_path = "AlexNet.pth"

    # 加载模型
    model = load_model(weight_path, device)

    # 指定要分类的图像路径
    # image_path = input("请输入要分类的图像路径 (jpg格式): ")
    image_path = r"data_set\catsdogs\val\Cat\4020.jpg"

    if not os.path.exists(image_path):
        print(f"图像文件 '{image_path}' 不存在。")
    else:
        # 输出分类结果
        prediction = predict_image(image_path, model, device)
        print(f"分类结果: {prediction}")
