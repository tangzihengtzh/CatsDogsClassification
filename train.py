import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from MyDatasets import Mydatasets
from MyModels import AlexNet,MyCNN

'''
此为新版train
'''

def train_model(data_dir, num_epochs=100, batch_size=32, learning_rate=0.0002, save_path='AlexNet.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化训练数据集
    train_dataset = Mydatasets(root_dir=os.path.join(data_dir, "train"))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = AlexNet(num_classes=2).to(device)
    weight_path = save_path

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"权重文件 '{weight_path}' 已加载，继续训练。")
    else:
        print(f"未找到权重文件，开始全新训练。")

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        model.train()  # 切换模型为训练模式
        running_loss = 0.0

        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 梯度清零

            # 前向传播
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            # print("(outputs, labels.to(device))",outputs.shape, labels.shape)
            # exit(2)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新进度条描述
            train_bar.desc = f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / (step + 1):.4f}"

        # 每隔10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1}")

    # 保存最终模型
    torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    data_dir = './data_set/catsdogs'
    train_model(data_dir=data_dir,save_path="AlexNet.pth")
