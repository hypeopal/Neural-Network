import time
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# 数据预处理与加载
batch_size = 128
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = tv.datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = tv.datasets.MNIST(root='./data', train=False, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义LeNet模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block1 = nn.Sequential(  # 第一层卷积
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(  # 第二层卷积
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense_block = nn.Sequential(  # 全连接层
            nn.Linear(32 * 4 * 4, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.dense_block(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyNet().to(device)
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)  # 优化函数

losses = []
accuracies = []
start_time = time.time()

# 训练
for i in tqdm(range(10)):
    for j, (batch_data, batch_label) in enumerate(train_data_loader):
        batch_data, batch_label = batch_data.to(device), batch_label.to(device)
        optimizer.zero_grad()
        prediction = model(batch_data)
        loss = criterion(prediction, batch_label)
        loss.backward()
        optimizer.step()
        predicted = torch.max(prediction, 1)[1]
        correct = (predicted == batch_label).sum().item()
        accuracy = correct / batch_size * 100
        losses.append(loss.item())  
        accuracies.append(accuracy)  # 保存准确率

end_time = time.time()
print('训练花了： %d s' % int((end_time - start_time)))

# 训练结果可视化
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color=color)
ax1.plot(losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 共享同一个x轴
color = 'tab:green'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracies, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training Loss and Accuracy')
plt.show()

# 测试
correct = 0
for batch_data, batch_label in tqdm(test_data_loader):
    batch_data, batch_label = batch_data.to(device), batch_label.to(device)
    prediction = model(batch_data)
    predicted = torch.max(prediction.data, dim=1)[1]
    correct += torch.sum(predicted == batch_label)
print('准确率：%.2f %%' % (correct / 100))
