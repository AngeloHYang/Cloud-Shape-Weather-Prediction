import torch
import torchvision.transforms as transforms
from  PIL import Image

def core(net, img_path):
    device = torch.device('cpu')

    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    img = transform(img).unsqueeze(0)
    output = net(img.to(device))
    index = torch.argmax(output).item()
    list = ['sunny', 'rain', 'fog', 'storm']
    print(list[index])

#if _name_ =='_main_':

net = torch.load("E:\\pycharmprojectImageProcessing\\test\\19.pth")
# test
#for i in range(1,10):
 # core(net,"E:\\cloud\\test\\sunny\\sunny ("+"%s"% (i) +").jpg")
 # print(i)

core(net,"E:\\cloud\\test\\rain\\rain (1).jpg")
