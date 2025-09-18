import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset_build import MinihackDataset
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from PIL import Image
from loguru import logger

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model_path = "./src/rec_manager/cnn_rec/cnn_model.pth"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(num_classes):
    if not os.path.exists(model_path):
        logger.info("没有检测到cnn模型, 为您训练一个捏")
        model = train_model(num_classes)
        model.eval()
        model.to(device)
        return model

    else:
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        return model


def train_model(num_classes, epochs=100, batch_size=32):
    dataset = MinihackDataset()
    model = SimpleCNN(num_classes)
    model.to(device)
    if dataset.length == 0:
        logger.info("tile数据为空，停止模型训练")
        return model
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in tqdm(range(epochs)):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            logger.info(
                f"Epoch {i + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%"
            )

    torch.save(model.state_dict(), f"{model_path}")

    logger.info("训练完成, 开始测试")
    test_data_path = os.listdir("./src/rec_manager/cnn_rec/tiles")
    for i in range(len(test_data_path)):
        img_path = os.path.join("./src/rec_manager/cnn_rec/tiles", test_data_path[i])
        img = Image.open(img_path)
        transform = ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)
        pred = model(img.to(device))
        pred = pred.squeeze(0)
        pred = pred.argmax(dim=0)
        logger.info(f"{test_data_path[i]}预测结果为: {pred.item()}")

    return model
