import json
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Datasets_writ(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        # super(CIFAR10_IMG, self).__init__()
        # super().__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_annotation = root
            img_folder = root + '\\train\\'
        else:
            file_annotation = root
            img_folder = root + '\\test\\'
        # fp = open(file_annotation, 'r')
        # data_dict = json.load(fp)
        class_map=["sunny","rain","fog","storm"]
        label_map={"sunny":0,"rain":1,"fog":2,"storm":3}
        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        # assert len(data_dict['images']) == len(data_dict['categories'])
        # num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in class_map:
            path=os.listdir(img_folder+"\\"+i)
            for j in path:
                self.filenames.append(img_folder+"\\"+i+"\\"+j)
                self.labels.append(label_map[i])

    def __getitem__(self, index):
        # img_name = self.img_folder + self.filenames[index]
        img_name=self.filenames[index]
        label = self.labels[index]
        # img = plt.imread(img_name)
        img=Image.open(img_name).convert('RGB')

        print(img_name)
        # img=np.array(img)
        img = self.transform(img)  # 可以根据指定的转化形式对数据集进行转换

        return img, label

    def __len__(self):
        print("len",len(self.filenames))
        return len(self.filenames)