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
epcho=10
lr=0.01
flag=0
for i in range(epcho):
    running_loss=0.0
    for image, label in zip(images, labels):
        output=net(image)
        loss=criterion(output,label)
        net.zero_grad()
        loss.backward()
        running_loss+=loss.item()
        if flag==0:
            for param in net.parameters():
                momentum=torch.zeros_like(param.grad.data)
                flag=1
        for param in net.parameters():
            momentum=0.9*momentum+param.grad.data
            param.data-=lr*momentum
    print(running_loss/len(train_loader))
    