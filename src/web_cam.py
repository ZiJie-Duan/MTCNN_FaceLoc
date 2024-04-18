
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import time

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

transform_r = transforms.Compose([
    transforms.Resize(24),
    transforms.CenterCrop(24),
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor。
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

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

class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(3, 28, 3)  #24 -> 22 -> maxp -> 11
        self.conv2 = nn.Conv2d(28, 48, 3) #11 -> 9 -> maxp -> 4
        self.conv3 = nn.Conv2d(48, 64, 2) #4 -> 3

        self.linear1 = nn.Linear(64*3*3, 128)

        self.face_det = nn.Linear(128, 2)
        self.bbox = nn.Linear(128, 4)
        self.landmark = nn.Linear(128, 10)

    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.conv1(x)) #22
        x = F.max_pool2d(x, 2) #11
        x = F.relu(self.conv2(x)) #9
        x = F.max_pool2d(x, 2) #4
        x = F.relu(self.conv3(x)) #3

        x = x.view(-1, 64*3*3)
        x = F.relu(self.linear1(x))

        facedet = self.face_det(x)
        bbox = self.bbox(x)
        landmark = self.landmark(x)

        return facedet, bbox, landmark

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


def generate_image_pyramid(img, scale_factor=1.2, min_size=(24, 24)):
    """
    生成图像的金字塔。
    
    :param img: 原始图像
    :param scale_factor: 缩放因子
    :param min_size: 图像在金字塔中的最小尺寸
    :return: 金字塔图像列表
    """
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


def sliding_window(image, step_size, window_size, model_trained):
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
            x_scale = w / 24
            y_scale = h / 24

            with torch.no_grad():
                face_det, bbox, _ = model_trained(window_tensor)

            #result.append((x, y, window_size[0], window_size[1]))
            probabilities = F.softmax(face_det, dim=1)
            
            if probabilities[0][0] > 0.95:
                result.append((x, y, window_size[0], window_size[1], probabilities[0][0]))

                # nx = bbox[0][0].item() * x_scale + x
                # ny = bbox[0][1].item() * y_scale + y
                # nw = bbox[0][2].item() * x_scale
                # nh = bbox[0][3].item() * y_scale
                # result.append((nx, ny, nw, nh, face_det[0][0] - face_det[0][1]))
            
        
    return result


def verify_face(image, model_trained):
    """
    使用训练好的模型检测图像中的人脸。
    
    :param image: 输入的原始图像
    :param model_trained: 训练好的人脸检测模型

    :return: 布尔值
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将NumPy数组转换为PIL.Image对象
    image_pil = Image.fromarray(image_rgb)

    image_tensor = transform_r(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        face_det, _,_ = model_trained(image_tensor)
    probabilities = F.softmax(face_det, dim=1)

    if probabilities[0][0] > 0.95:
        return True, probabilities[0][0]
    else:
        return False, probabilities[0][0]


# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0代表计算机的默认摄像头


net1 = torch.load(r"C:\Users\lucyc\Desktop\MTCNN_FaceLoc\pnet_temp\PNet_45.pth")

p_net = PNet()
p_net.load_state_dict(net1.state_dict())
p_net.eval()
p_net.to(device)

net2 = torch.load(r"C:\Users\lucyc\Desktop\MTCNN_FaceLoc\src\face_loc_r_48.pth")

r_net = RNet()
r_net.load_state_dict(net2.state_dict())
r_net.eval()
r_net.to(device)



# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# # 打开视频文件
# file = r"C:\Users\lucyc\Desktop\4月17日.mp4"
# cap = cv2.VideoCapture(file)

#img = cv2.imread(r"C:\Users\lucyc\Desktop\IMG_20150528_145916.jpg")

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()

    # frame = img
    # ret = True

    frame = frame[60:-60, :]

    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    pyramid = generate_image_pyramid(frame, scale_factor=1.3, min_size=(24, 24))

    result = []
    for img, scal in pyramid:
        res = sliding_window(img, step_size=13, window_size=(24, 24), model_trained=p_net)
        res = [[x*scal for x in y] for y in res]
        result += res

    result = nms(result, 0.3)

    result2 = []
    for x, y, w, h, _ in result:
        x, y, w, h = int(x), int(y), int(w), int(h)

        is_face, score = verify_face(frame[y:y+h, x:x+w], r_net)
        if is_face:
            result2.append((x, y, w, h, score))
    
    result = nms(result2, 0.3)

    for x, y, w, h, score in result:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face: {:.2f}".format(score), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # #显示结果帧

    # for x, y, w, h, score in result:
    #     x, y, w, h = int(x), int(y), int(w), int(h)
    #     random = np.random.randint(0, 255, 3)
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (int(random[0]), int(random[1]), int(random[2])), 2)

    # 打印边框数量
    cv2.putText(frame, "Number of faces: {}".format(len(result)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame with Border', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()



# import cv2

# # 读取图像
# img_path = r"C:\Users\lucyc\Desktop\IMG_20150528_145916.jpg"
# img = cv2.imread(img_path)

# # 检查图像是否正确加载
# if img is None:
#     print("Can't load image. Exiting ...")
# else:
#     # 调整图像，去除上下各60像素
#     img_cropped = img[60:-60, :]

#     # 假设generate_image_pyramid是一个已定义的函数，用于创建图像金字塔
#     # 需要提供这个函数的实现
#     pyramid = generate_image_pyramid(img_cropped, scale_factor=1.5, min_size=(24, 24))

#     results = []
#     # 处理图像金字塔中的每一层
#     for img_level, scale in pyramid:
#         # 这里可以加入图像处理的代码
#         # 例如：检测图像中的某些特征
#         results.append((img_level, scale))


#         # 检查图像是否正确加载
#     if img is None:
#         print("Can't load image. Exiting ...")
#     else:
#         # 显示图像
#         cv2.imshow('Loaded Image', img)
#         cv2.waitKey(0)  # 等待直到有键盘输入
#         cv2.destroyAllWindows()  # 关闭显示窗口
#     # 输出、显示或保存处理结果
#     print("Processing complete. Number of pyramid levels:", len(results))

# input("Press Enter to continue ...")
