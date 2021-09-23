import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Dataloader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ExampleNet import ExampleNet
from datasets import Datasets_writ

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 20# 迭代次数
    batch_size =512

    net = ExampleNet().to(device)
    # loss_fun = nn.MSELoss() #
    loss_fun=nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    transform_train = transforms.Compose([
         transforms.Resize(32),
         transforms.CenterCrop(32),
         transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
     ])
    print("debug 1")
    datasets=Datasets_writ('E:\\cloud',
                      train = True,
                     transform = transform_train )
    print("debug 2")
    transform_test = transforms.Compose([
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

    testdataset = Datasets_writ('E:\\cloud',
                      train = False,
                     transform = transform_test )
    dataloader = DataLoader(datasets,batch_size=batch_size,shuffle=True)
    testdataloader = DataLoader(testdataset,batch_size=batch_size,shuffle=False)

    for i in range(epochs):
        net.train()
        print("epochs:{}".format(i))
        for j,(input,target) in enumerate(dataloader):
            input = input.to(device)
            output = net(input)
            # target = torch.zeros(target.size(0),10).scatter_(1,target.view(-1,1).to(device))
            loss = loss_fun(output,target)
            print("loss",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 ==0:
                print("[epochs - {0} - {1}/{2}] loss:{3}".format(i,j,len(dataloader),loss.float()))
        torch.save(net,"E:\\pycharmprojectImageProcessing\\test\\"+str(i)+".pth")
    #torch.save(net,"models/net.pth")
    torch.save(net,"E:\\pycharmprojectImageProcessing\\test\\"+str(i)+".pth")
train()
