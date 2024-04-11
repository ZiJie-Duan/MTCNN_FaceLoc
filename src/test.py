import os
import csv
import random
import cv2
import math
import scipy.io
import numpy as np


class ImgDataEnhance:
    def __init__(self, cut_bbox, img, bbox, landmark) -> None:
        # deepcopy the image, so that the original image will not be changed
        self.img = img[cut_bbox[1]:cut_bbox[1]+cut_bbox[3],
                       cut_bbox[0]:cut_bbox[0]+cut_bbox[2], :].copy()
        self.bbox = bbox
        self.landmark = landmark

    def random_color_shift(self):
        img = self.img
        img = img.astype(np.int32) # 转换为整数类型
        # 生成色偏值，例如在-50到50的范围内随机选择
        bias = np.random.randint(-50, 50, 3) # 对于RGB三个通道
        # 应用色偏
        for i in range(3): # 对于每个颜色通道
            img[:, :, i] = np.clip(img[:, :, i] + bias[i], 0, 255)
        
        self.img = img.astype(np.uint8)
    
    def rotate_bbox(self, M):
        # 旋转bbox
        x, y, w, h = self.bbox
        print(x, y, w, h)
        # 生成四个顶点
        pts = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
        # 应用旋转矩阵
        npts = cv2.transform(np.array([pts]), M)[0]
        # 生成新的bbox
        nx, ny, nw, nh = cv2.boundingRect(npts)
        return [nx, ny, nw, nh]

    def rotate_landmark(self, M):
        # 旋转landmark
        nldmk = np.array(self.landmark, dtype=np.float32).reshape(-1, 2)
        nldmk = cv2.transform(np.array([nldmk]), M)[0]
        return nldmk.flatten()
    
    def random_rotate(self):
        img = self.img
        bbox = self.bbox
        landmark = self.landmark
        # 生成随机角度
        angle = np.random.randint(-30, 30)
        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        # 应用旋转矩阵
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        bbox = self.rotate_bbox(M)
        landmark = self.rotate_landmark(M)

        self.img = img
        self.bbox = bbox
        self.landmark = landmark

    def random_flip(self):
        img = self.img
        bbox = self.bbox
        landmark = self.landmark
        # 随机翻转
        if np.random.randint(2) == 0:
            # 沿着y轴翻转
            img = cv2.flip(img, 1)
            bbox[0] = img.shape[1] - bbox[0] - bbox[2]
            landmark[0::2] = img.shape[1] - landmark[0::2]

        self.img = img
        self.bbox = bbox
        self.landmark = landmark

    def random_zoom(self):
        img = self.img
        bbox = self.bbox
        landmark = self.landmark
        # 随机缩放
        scale = np.random.uniform(0.8, 1.2)
        # 缩放图像
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 缩放bbox

        bbox[0] = int(bbox[0] * scale)
        bbox[1] = int(bbox[1] * scale)
        bbox[2] = int(bbox[2] * scale)
        bbox[3] = int(bbox[3] * scale)
        # 缩放landmark

        landmark[0::2] = [int(x * scale) for x in landmark[0::2]]
        landmark[1::2] = [int(y * scale) for y in landmark[1::2]]
        
        self.img = img
        self.bbox = bbox
        self.landmark = landmark

    def random_generate(self, number=1):
        # for i in range(number):
        #     for j in range(np.random.randint(1, 4)):
        #         choice = np.random.randint(4)
        #         if choice == 0:
        #             print("color shift")
        #             self.random_color_shift()
        #         elif choice == 1:
        #             print("rotate")
        #             self.random_rotate()
        #         elif choice == 2:
        #             print("flip")
        #             self.random_flip()
        #         else:
        #             print("zoom")
        #             self.random_zoom()
        for i in range(number):

            print("color shift")
            self.random_color_shift()

            print("rotate")
            self.random_rotate()

            print("flip")
            self.random_flip()

            print("zoom")
            self.random_zoom()



def main():

    img = cv2.imread(r"C:\Users\lucyc\Desktop\29bairdWEB-master1050.jpg")
    bbox = [100, 100, 200, 200]
    landmark = [150, 150, 160, 150, 150, 160, 160, 160]
    cut_bbox = [10, 10, 500, 500]

    img_data = ImgDataEnhance(cut_bbox, img, bbox, landmark)
    img_data.random_generate(1)
    cv2.imshow('img', img_data.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()