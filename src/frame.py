from torch.utils.data import Dataset
from PIL import Image
import csv
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


IMG_INPUT_SIZE = [12,12]

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize(IMG_INPUT_SIZE[0]),
    transforms.CenterCrop(IMG_INPUT_SIZE[0]),
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor。
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])


def label_transform(label, img_size):
    # 目标尺寸
    nh, nw = IMG_INPUT_SIZE[1], IMG_INPUT_SIZE[0]
    # 原始尺寸
    h, w = img_size
    # 计算缩放比例
    x_scale = nw / w
    y_scale = nh / h
    
    # 处理标签中的每个坐标
    transformed_label = []
    for i, value in enumerate(label):
        if i % 2 == 0:  # 偶数索引位置，x坐标
            transformed_label.append(value * x_scale)
        else:  # 奇数索引位置，y坐标
            transformed_label.append(value * y_scale)
            
    return transformed_label



class FLCDataset(Dataset):
    
    def __init__(self, csv_dir, img_dir):
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        self.lable_transform = label_transform
        self.datalines = self.read_csv_file(csv_dir)

        self.sample_type = 0
        # 0 positive, 1 mixed, 2 negative, 3 landmark


    def __len__(self):
        return len(self.datalines)


    def __getitem__(self, idx):

        data_info = self.datalines[idx]

        img = self.read_img(data_info[0])

        labbel = [int (i) for i in data_info[1].split()]
        labbel = self.lable_transform(labbel, img.size)
        labbel += [0 for i in range(10-len(labbel))]
        labbel = torch.tensor(labbel)

        if self.transform:
            img = self.transform(img)
        
        #labbel = " ".join([str(i) for i in labbel])

        return img, labbel, data_info[2]
        
    def read_img(self, i):
        img = Image.open(self.to_path(i))
        return img
    
    def read_csv_file(self, file_path):
        """
        read the csv file
        :param file_path: the path of the csv file
        :return: the list of the csv file
        """
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        return lines

    def to_path(self, i):
        path = self.img_dir + "\\" + str(i) + ".jpg"
        return path
    
        
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化PyTorch tensor图像"""
    for t, m, s in zip(tensor, mean, std):  # 对每个通道进行操作
        t.mul_(s).add_(m)  # 对应于 (x * std) + mean
    return tensor

def visualize_transformed_image(tensor_image, bbox, landmark, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反标准化并可视化PyTorch tensor图像。
    
    Parameters:
    - tensor_image: PyTorch tensor，代表经过transform变换的图像。
    - mean: 标准化时使用的均值，应与transform操作中的均值相对应。
    - std: 标准化时使用的标准差，应与transform操作中的标准差相对应。
    """
    # 克隆图像tensor以避免修改原始数据，并进行反标准化处理
    unnormalized_image = unnormalize(tensor_image.clone().detach(), mean, std)
    
    # 将tensor图像转换为NumPy数组，并调整形状为HxWxC以适应matplotlib
    img = unnormalized_image.numpy().transpose((1, 2, 0))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    b = bbox
    # 在图片上画矩形框，参数分别是：图片、左上角坐标、右下角坐标、颜色（BGR格式）、线条厚度
    img = cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 1)

    if landmark != None:
        for x, y in [(landmark[i],landmark[i+1]) for i in range(0,len(landmark),2)]:
            img = cv2.rectangle(img, (x,y), (x,y), (0, 0, 255), 1)
            
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # 不显示坐标轴
    plt.show()



