import os
import csv
import random
import cv2
import math
import scipy.io
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


IMG_INPUT_SIZE = [12,12]
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
# 定义转换操作
transform = transforms.Compose([
    transforms.Resize(IMG_INPUT_SIZE[0]),
    transforms.CenterCrop(IMG_INPUT_SIZE[0]),
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor。
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(3, 10, 3)  #12 -> 10 -> maxp -> 5
        self.conv2 = nn.Conv2d(10, 16, 3) #5 -> 3
        self.conv3 = nn.Conv2d(16, 32, 3) #3 -> 1

        self.face_det = nn.Conv2d(32, 2, 1) #1 -> 1
        self.bbox = nn.Conv2d(32, 4, 1) #1 -> 1
        self.landmark = nn.Conv2d(32, 10, 1) #1 -> 1

    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.conv1(x)) #10
        x = F.max_pool2d(x, 2) #5
        x = F.relu(self.conv2(x)) #3
        x = F.relu(self.conv3(x)) #1

        facedet = self.face_det(x)
        bbox = self.bbox(x)
        landmark = self.landmark(x)

        facedet = torch.flatten(facedet, 1)
        bbox = torch.flatten(bbox, 1)
        landmark = torch.flatten(landmark, 1)

        return facedet, bbox, landmark


model_trained = torch.load(r"C:\Users\lucyc\Desktop\AI_GOGOGO\Pytorch\face_loc\v1\face_loc_p_3.pth")
model_trained.eval()  # 设置模型为评估/测试模式

def random_color_shift(img):

    img = img.astype(np.int32) # 转换为整数类型
    # 生成色偏值，例如在-50到50的范围内随机选择
    bias = np.random.randint(-50, 50, 3) # 对于RGB三个通道
    # 应用色偏
    for i in range(3): # 对于每个颜色通道
        img[:, :, i] = np.clip(img[:, :, i] + bias[i], 0, 255)
    
    return img.astype(np.uint8)


def cal_iou_wh(boxA, boxB):
        # 计算两个边界框的坐标
        boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
        boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        # 计算交集的面积
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # 计算两个边界框的面积
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        # 计算并集的面积
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        # 返回计算出的IoU值
        return iou

def nms(imgs_list, threshold):

    if imgs_list == []:
        return []

    imgs_list = sorted(imgs_list, key=lambda x: x[4], reverse=True)
    max_img = imgs_list[0]

    next_recu_imgs = []

    for i in range(1, len(imgs_list)):
        if cal_iou_wh(max_img[:4], imgs_list[i][:4]) > threshold:
            pass
        else:
            next_recu_imgs.append(imgs_list[i])
    
    return [max_img] + nms(next_recu_imgs, threshold)


class CELEBADriver:
    
    #['image_id', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y','leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
    #['000001.jpg', '69', '109', '106', '113', '77', '142', '73', '152', '108', '154']
    #['image_id', 'x_1', 'y_1', 'width', 'height']
    #['000001.jpg', '95', '71', '226', '313']

    def __init__(self, bbox_path, landmarks_path, imgs_path):
        self.bbox_path = bbox_path
        self.landmarks_path = landmarks_path
        self.imgs_path = imgs_path
        self.data = {}
        self.dataset_index = []  # 样本汇总
        self.init_dataset()

    def init_dataset(self):
        with open(self.bbox_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count == 0:
                    count += 1
                    continue
                self.data[count] = {"name":row[0]}
                self.data[count]["bbox"] = [int(x) for x in row[1:]]
                count += 1
                
        with open(self.landmarks_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count == 0:
                    count += 1
                    continue
                self.data[count]["ldmk"] = [int(x) for x in row[1:]]
                count += 1

        self.dataset_index = [x for x in range(count)]

    def get_file_path(self, index):
        # index 获取一个图片文件路径
        return self.imgs_path + "\\" + self.data[index]["name"]

    def get_face_bbx_list(self, index):
        return [self.data[index]["bbox"]]

    def get_face_ldmk_list(self, index):
        return [self.data[index]["ldmk"]]

    def random_init(self):
        random.shuffle(self.dataset_index)

    def get_data(self, i):
        i = self.dataset_index[i]
        return (self.get_file_path(i),self.get_face_bbx_list(i),self.get_face_ldmk_list(i))

class WFDriver:

    def __init__(self, mat_path, clas_root_path):
        self.data = scipy.io.loadmat(mat_path) # 读取 mat 文件到 内存中
        self.r_path = clas_root_path # 类别文件 系统根目录
        self.clas_i_map = {} # 我们利用文件命名中的数字编号 作为索引
        # 建立clas_i_map 将文件命名中的编号 和 mat数据中索引进行关联
        self.clas_map = {} # 将 文件命名中的编号 和 类别文件夹的全名
        self.dataset_index = [] # 样本汇总

        self.init_clas_map() 
        
    def init_clas_map(self):
        # 初始化 类别标签
        # 我们利用文件命名中的数组编号 作为索引
        # 建立clas_i_map 将文件命名中的编号 和 数据中随机索引进行关联
        for i in range(len(self.data["file_list"])):
            j = int(self.data["file_list"][i][0][0][0][0].split("_")[0])
            self.clas_i_map[j] = i

        # 生成clas_map，将类别编号 和其 名称绑定
        for _, dirs, files in os.walk(self.r_path):
            for dir in dirs:

                clas_i = int(dir.split("--")[0]) # 根据目录名字 生成类别索引
                self.clas_map[clas_i] = dir      # 将类别索引 与 目录名字 建立连接

                num_smp = len(self.data["file_list"][self.clas_i_map[clas_i]][0]) # 求出一个类别有多少个样本
                self.dataset_index += [(clas_i, x) for x in range(num_smp)] # 并入 样本数组

            
    def get_file_path(self, clas, index):
        # 通过 clas 和 index 获取一个图片文件信息
        return self.r_path + "\\" + self.clas_map[clas]\
                + "\\" + self.data["file_list"][self.clas_i_map[clas]][0][index][0][0] + ".jpg"

    def get_face_bbx_list(self, clas, index):
        return self.data["face_bbx_list"][self.clas_i_map[clas]][0][index][0]

    def random_init(self):
        random.shuffle(self.dataset_index)

    def get_data(self, i):
        assert i < len(self.dataset_index), "Index out of range WF {} {}".format(i, len(self.dataset_index))
        i,j = self.dataset_index[i]
        return (self.get_file_path(i,j),self.get_face_bbx_list(i,j),None)
            

class ImgTransform_R:
    """
    Transforms for images for R Net
    We use cv2 for image processing
    1. Generate face bounding box via landmarks
    2. Random crop face bounding box 
        (iou>0.6 for positive, iou<0.2 for negative, 0.2<iou<0.55 for mixed)
    3. Modify landmarks and bounding box according to the crop

    sample type:
        l-landmark, p-positive, m-mixed, n-negative
    """

    def __init__(self, img_path, bbox=None, landmark=None, p_net=None):
        self.p_net = p_net
        self.img_path = img_path
        self.img = cv2.imread(img_path,1)
        self.bbox = bbox        # [[x,y,w,h]]
        self.landmark = landmark # [[x1,y1,x2,y2,...]]
        self.hight, self.width = self.img.shape[:2]

        if self.landmark != None:
            self.update_bbox_via_landmark()
            self.datatype = "Full Sample" # landmark
        else:
            self.datatype = "Bbox Sample" # only bbox
    

    def update_bbox_via_landmark(self):
        ceb_landmark = self.landmark[0]
        # 定位左右眼
        leye = ceb_landmark[:2]
        reye = ceb_landmark[2:4]

        # 求解双目距离
        dis = math.sqrt((abs(leye[0] - reye[0])**2) + (abs(leye[1] - reye[1])**2))
        dis = dis//1

        # 求解 嘴部图像最低点
        lower = min(ceb_landmark[-1], ceb_landmark[-3])

        # 硬编码 面部关系
        x = leye[0] - dis//1.5
        y = leye[1] - dis
        w = dis*2.5
        h = lower - y + dis
        
        bbox = [int(x) for x in [x,y,w,h]]
        self.bbox = [bbox]
        #show_img(ceb_img, [bbox], None)


    def generate_image_pyramid(self, scale_factor=1.2, min_size=(24, 24)):
        """
        生成图像的金字塔。
        
        :param img: 原始图像
        :param scale_factor: 缩放因子
        :param min_size: 图像在金字塔中的最小尺寸
        :return: 金字塔图像列表
        """
        img = self.img
        pyramid_images = []
        scale_factor_base = 5

        while True:
            new_width = int(img.shape[1] / scale_factor_base)
            new_height = int(img.shape[0] / scale_factor_base)

            if new_width < min_size[0] or new_height < min_size[1]:
                break

            img2 = cv2.resize(img, (new_width, new_height))
            pyramid_images.append([img2, scale_factor_base])
            scale_factor_base *= scale_factor # 可以调整以控制金字塔的级别间隔

        return pyramid_images


    def sliding_window(self, image, step_size, window_size, model_trained):
        """
        对图像应用滑动窗口，并使用提供的模型检测人脸。
        
        :param image: 输入的原始图像
        :param step_size: 每次滑动的像素数
        :param window_size: 窗口大小 (宽度, 高度)
        :param model_trained: 训练好的人脸检测模型
        """
        # 图像尺寸
        (h, w) = image.shape[:2]

        result = []
        
        # 逐步移动窗口
        for y in range(0, h - window_size[1], step_size):
            for x in range(0, w - window_size[0], step_size):
                # 提取当前窗口的图像片段
                window = image[y:y + window_size[1], x:x + window_size[0]]

                image_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                # 将NumPy数组转换为PIL.Image对象
                image_pil = Image.fromarray(image_rgb)
                

                window_tensor = transform(image_pil).unsqueeze(0).to(device)
                x_scale = w / 12
                y_scale = h / 12

                with torch.no_grad():
                    face_det, bbox, _ = model_trained(window_tensor)

                #result.append((x, y, window_size[0], window_size[1]))
                
                if face_det[0][0] > face_det[0][1]:
                    result.append((x, y, window_size[0], window_size[1], face_det[0][0] - face_det[0][1]))
                    # nx = bbox[0][0].item() * x_scale + x
                    # ny = bbox[0][1].item() * y_scale + y
                    # nw = bbox[0][2].item() * x_scale
                    # nh = bbox[0][3].item() * y_scale
                    # result.append((nx, ny, nw, nh, face_det[0][0] - face_det[0][1]))
                
        return result


    def cal_iou(self, boxA, boxB):
        # 计算两个边界框的坐标
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        # 计算交集的面积
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # 计算两个边界框的面积
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        # 计算并集的面积
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        # 返回计算出的IoU值
        return iou
    
    def int_values(self, values):
        return [int(x) for x in values]
    

    def gen_r_sample(self):
        # 生成R Net的样本
        pyramid = self.generate_image_pyramid(scale_factor=1.1, min_size=(24, 24))

        result = []
        for img, scal in pyramid:
            res = self.sliding_window(img, step_size=13, window_size=(24, 24), model_trained=model_trained)
            res = [[x*scal for x in y] for y in res]
            result += res

        result = nms(result, 0.3)

        p_samples = []
        n_samples = []
        m_samples = []

        for box in self.bbox:
            for x, y, w, h, score in result:
                iou = self.cal_iou([box[0],box[1],box[0]+box[2],box[1]+box[3]], [x,y,x+w,y+h])
                x, y, w, h = self.int_values([x, y, w, h])
                if iou > 0.6:
                    p_samples.append(([x,y,w,h],box))
                elif iou < 0.2:
                    n_samples.append(([x,y,w,h],box))
                elif iou > 0.2 and iou < 0.55:
                    m_samples.append(([x,y,w,h],box))
        
        minv = min([len(p_samples), len(n_samples), len(m_samples)])
        print("p_samples: {}, m_samples: {}, n_samples: {}".format(len(p_samples), len(m_samples), len(n_samples)))
        p_samples = random.choices(p_samples, k=minv)
        n_samples = random.choices(n_samples, k=minv)
        m_samples = random.choices(m_samples, k=minv)

        return p_samples, m_samples, n_samples


    def redefine_bbox(self, crop, bbox):
        # 通过裁剪后的图片和原始bbox，重新定义bbox
        x = bbox[0] - crop[0]
        y = bbox[1] - crop[1]
        w = bbox[2]
        h = bbox[3]
        return [x,y,w,h]
    
    def img_size_check(self, bbox):
        # 检查裁剪后的边框是否符合要求
        if bbox[2] < 24 or bbox[3] < 24:
            return False
        return True

    def generate_wfd_sample(self):

        sample_list = [] # 生成的样本列表
        """
        [
            [(img,[nbx,nby,nbw,nbh]), xxx, xxx],
            [positive, mixed, negative, negative, negative],
            ...
        ]
        """ # 将 负样本 生成多个，增加样本数量，平衡正负样本


        sample_trip = [] # 生成的样本列表

        # 生成三种样本
        # p-positive 正样本，用于人脸识别 和 边框检测训练
        # m-mixed 混合样本，用于人脸识别 和 边框检测训练
        # n-negative 负样本，用于人脸识别

        p_samples, m_samples, n_samples = self.gen_r_sample()

        for i in range(len(p_samples)):
            
            # p_samples[i][0] 是裁剪后的边框, p_samples[i][1] 是高度相关边框
            nbbox = self.redefine_bbox(p_samples[i][0], p_samples[i][1])
            nimg = self.img[p_samples[i][0][1]:p_samples[i][0][1]+p_samples[i][0][3],
                            p_samples[i][0][0]:p_samples[i][0][0]+p_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, nbbox))

            nbbox = self.redefine_bbox(m_samples[i][0], m_samples[i][1])
            nimg = self.img[m_samples[i][0][1]:m_samples[i][0][1]+m_samples[i][0][3],
                            m_samples[i][0][0]:m_samples[i][0][0]+m_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, nbbox))

            nbbox = self.redefine_bbox(n_samples[i][0], n_samples[i][1])
            nimg = self.img[n_samples[i][0][1]:n_samples[i][0][1]+n_samples[i][0][3],
                            n_samples[i][0][0]:n_samples[i][0][0]+n_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, nbbox))

        sample_list.append(sample_trip)
        
        return sample_list # allow empty list
    

    def redefine_landmark(self, crop, landmark):
        # 通过裁剪后的图片和原始landmark，重新定义landmark
        nlandmark = landmark[:]
        for i in range(1,len(landmark),2):
            nlandmark[i-1] -= crop[0]
            nlandmark[i] -= crop[1]
        return nlandmark
    

    def generate_ceb_sample(self):
        
        sample_trip = []
        p_samples, m_samples, n_samples = self.gen_r_sample()
        
        for i in range(len(p_samples)):

            # only one landmark here for ceb, so landmark[0]
            plandmark = self.redefine_landmark(p_samples[i][0], self.landmark[0])
            nimg = self.img[p_samples[i][0][1]:p_samples[i][0][1]+p_samples[i][0][3],
                            p_samples[i][0][0]:p_samples[i][0][0]+p_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, p_samples[i][1], plandmark))
            
            mlandmark = self.redefine_landmark(m_samples[i][0], self.landmark[0])
            nimg = self.img[m_samples[i][0][1]:m_samples[i][0][1]+m_samples[i][0][3],
                            m_samples[i][0][0]:m_samples[i][0][0]+m_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, m_samples[i][1], mlandmark))

            nlandmark = self.redefine_landmark(n_samples[i][0], self.landmark[0])
            nimg = self.img[n_samples[i][0][1]:n_samples[i][0][1]+n_samples[i][0][3],
                            n_samples[i][0][0]:n_samples[i][0][0]+n_samples[i][0][2], :].copy()
            nimg = random_color_shift(nimg)
            sample_trip.append((nimg, n_samples[i][1], nlandmark))

        return sample_trip # allow empty list
    

class BuildDataset:
    
    def __init__(self, wfdd, ceba, imgs_path, csv_path):
        self.wfdd = wfdd
        self.ceba = ceba
        self.wfd_index = wfdd.dataset_index
        self.ceb_index = ceba.dataset_index

        self.imgs_path = imgs_path
        self.csv_path = csv_path

        self.sample_index = 0


    def write_csv(self, line_elements):

        for i in range(len(line_elements)):
            if type(line_elements[i]) == list:
                line_elements[i] = " ".join([str(x) for x in line_elements[i]])
            else:
                line_elements[i] = str(line_elements[i])

        with open(self.csv_path, mode='a', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(line_elements)

    def save_img(self, img):
        path = self.imgs_path + "\\" + str(self.sample_index) + ".jpg"
        cv2.imwrite(path, img)


    def generate_dataset(self):
        
        faile_count = 0 # 记录失败次数
        ceb_count = 0 # 记录使用的ceb样本数量

        j = 0
        for i in range(len(self.wfd_index)):

            alread_failed = False
            ceb_count += 1

            print("Processing {}/{}".format(i, len(self.wfd_index)))

            wfd_imgp, wfd_bbox, _ = self.wfdd.get_data(i)

            samples = ImgTransform_R(wfd_imgp,
                                    wfd_bbox,
                                    p_net=model_trained)\
                                    .generate_wfd_sample()
            
            if len(samples) == 0:
                print("Failed to generate sample for {}".format(wfd_imgp))
                alread_failed = True
            else:
                for sample_group in samples:
                    for i in range(len(sample_group)):
                        csv_line = [self.sample_index, sample_group[i][1], i]
                        self.save_img(sample_group[i][0])
                        self.write_csv(csv_line)
                        self.sample_index += 1


            ceb_imgp, _, ceb_landmark = self.ceba.get_data(j)

            samples = ImgTransform_R(ceb_imgp,
                                    None,
                                    ceb_landmark,
                                    p_net=model_trained)\
                                    .generate_ceb_sample()
            
            if len(samples) == 0:
                print("Failed to generate sample for {}".format(ceb_imgp))
                alread_failed = True
            else:
                for sample_group in samples:
                    for i in range(len(sample_group)):
                        csv_line = [self.sample_index, sample_group[i][1], i]
                        self.save_img(sample_group[i][0])
                        self.write_csv(csv_line)
                        self.sample_index += 1
            
            if alread_failed:
                faile_count += 1

        print("\n\n--------Final Report-----------")
        print("Total failed samples: {}".format(faile_count))
        print("Total ceb samples used: {}".format(ceb_count))
        print("Total wfb samples used: {}".format(len(self.wfd_index)))
        print("Total samples generated: {}".format(self.sample_index // 4))
        print("number of ceb samples: {}".format(len(self.ceba.dataset_index)))
        print("-------------------------------\n\n")



bbox_path = r"C:\Users\lucyc\Desktop\celebA\list_bbox_celeba.csv"
ldmk_path = r"C:\Users\lucyc\Desktop\celebA\list_landmarks_align_celeba.csv"
basic_path = r"C:\Users\lucyc\Desktop\celebA\img_align_celeba\img_align_celeba"

# mkdir face_loc_d
os.makedirs(r"C:\Users\lucyc\Desktop\face_loc_dataset", exist_ok=True)

cead = CELEBADriver(bbox_path, ldmk_path, basic_path)

mat_path = r"C:\Users\lucyc\Desktop\faces\WIDER_train\WIDER_train\images"
clas_root_path = r"C:\Users\lucyc\Desktop\faces\wider_face_split\wider_face_split\wider_face_train.mat"

wfd = WFDriver(clas_root_path, mat_path)

cead.random_init()
wfd.random_init()

bds = BuildDataset(wfd, cead, r"C:\Users\lucyc\Desktop\face_loc_dataset", r"C:\Users\lucyc\Desktop\face_loc_dataset.csv")
bds.generate_dataset()

# belabelabelabela
# belabelabelabela