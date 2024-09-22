import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn

batch_size=4
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
dataiter=iter(train_loader)
images,labels=next(dataiter)

class lenet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.con1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.con2=nn.Conv2d(6,16,5)
        self.linear1=nn.Linear(16*5*5,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(torch.nn.functional.relu(self.con1(x)))
        x=self.pool(torch.nn.functional.relu(self.con2(x)))
        x=torch.flatten(x,1)
        x=torch.nn.functional.relu(self.linear1(x))
        x=torch.nn.functional.relu(self.linear2(x))
        x=self.linear3(x)
        return x
    

net=lenet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
def train(model,train_loader,criterion,optmizer):
    
    for epoch in range(10):
        running_loss=0.0
        for i,data in enumerate(train_loader,0):
            inputs,labels=data
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%2000==1999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

def test(model,test_loader,criterion):
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.daSta,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')











import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# 超参数
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个小批次打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

# 测试模型
def test(model, test_loader, criterion):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# 训练和测试
train(model, train_loader, criterion, optimizer, num_epochs)
test(model, test_loader, criterion)