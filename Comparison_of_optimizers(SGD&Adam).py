import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

transform=transforms.ToTensor()
train_data=datasets.MNIST(root="./data",train=True,transform=transform,download=True)
train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)


def train_model(optimizer_type):
    losses = []
    model=MLP()
    criterion=nn.CrossEntropyLoss()
    if optimizer_type=="SGD":
        optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(5):
        for images,labels in train_loader:
            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return losses

start=time.time()
sgd_losses=train_model("SGD")
print("SGD losses:", sgd_losses, "length:", len(sgd_losses))
sgd_time=time.time()-start

start=time.time()
adam_losses=train_model("Adam")
adam_time=time.time()-start

plt.plot(sgd_losses,label=f"SGD ({sgd_time:.2f})s",marker="o")
plt.plot(adam_losses,label=f"Adam ({adam_time:.2f})s",marker="o")
plt.xlabel("batch")
plt.ylabel("loss")
plt.legend()
plt.show()
