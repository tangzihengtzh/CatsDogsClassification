import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from MyDatasets import Mydatasets
from MyModels import AlexNet

# 此份train为旧版卡loss = 0.693的重写版本

# 前置过程部分
def main():
    # 设备配置：使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径
    data_dir = './data_set/catsdogs'

    # 实例化训练数据集
    train_dataset = Mydatasets(root_dir=os.path.join(data_dir, "train"))

    # 设置超参数
    batch_size = 32
    learning_rate = 0.0002
    num_epochs = 1000
    save_path = 'AlexNet.pth'

    # 创建 DataLoader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = AlexNet(num_classes=2).to(device)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"权重文件 '{save_path}' 已加载，继续训练。")
    else:
        print(f"未找到权重文件，开始全新训练。")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(num_epochs):
        model.train()  # 切换模型为训练模式
        total_loss = 0.0

        # 进度条显示每个 epoch 的训练进展
        with tqdm(total=len(dataloader), desc=f"Epoch [{epoch + 1}/{num_epochs}]", ncols=100) as pbar:
            for batch_idx, (imgs, lbls) in enumerate(dataloader):
                # 数据转移到 GPU（如果可用）
                imgs, lbls = imgs.to(device), lbls.to(device)

                # 前向传播
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累积损失
                total_loss += loss.item()

                # 每个batch结束，更新进度条
                pbar.set_postfix({'Batch Loss': loss.item(), 'Avg Loss': total_loss / (batch_idx + 1)})
                pbar.update(1)

        # epoch结束后的打印
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

        # 每隔10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1}")

    # 保存最终模型
    # torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
