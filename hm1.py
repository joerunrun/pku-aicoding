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
#train_dataset 有五万张图片，test_dataset有一万张图片
#train_dataset[0]是一个元组，第一个元素是图片，第二个元素是标签
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
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
import matplotlib.pyplot as plt

net=lenet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
def train(model,train_loader,criterion,optmizer):
    loss_list=[]
    for epoch in range(10):
        running_loss=0.0
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        #每个batch有4张图片，所以每个epoch有50000/4=12500个batch
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
                loss_list.append(running_loss/2000)
                running_loss = 0.0
    print('Finished Training')
    return loss_list
def test(model,test_loader,criterion):
    correct=0
    respected_corrext=[0]*10
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            for i in range(4):
                respected_corrext[labels[i]]+=(predicted[i]==labels[i]).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    return respected_corrext
losslist=train(net,train_loader,criterion,optimizer)

respect=test(net,test_loader,criterion)

plt.plot(losslist)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()



