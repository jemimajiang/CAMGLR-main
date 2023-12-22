import cupy as cp
import numpy as np
import sys
import cv2
import chainer.functions as F



class State():
    def __init__(self, size): # 没有使用move_range
        self.image = np.zeros(size,dtype=np.float32)
    
    def reset(self, x, n):
        self.image = x + n
    def contrast_reset(self, x):
        self.image = x

    def getEdge(self,im_ch,act,kernel_size,radius):
        filter = cp.ones((1, 1, kernel_size, kernel_size))
        L, R, U, D = [filter.copy() for _ in range(4)]

        U[:, :, radius + 1:, :] = 0
        D[:, :, 0: radius, :] = 0
        NW, NE, SW, SE = U.copy(), U.copy(), D.copy(), D.copy()

        NW[:, :, :, radius + 1:] = 0
        NE[:, :, :, 0: radius] = 0
        SE[:, :, :, 0: radius] = 0

        NW, NE, SW, SE = NW / ((radius + 1) ** 2), NE / ((radius + 1) ** 2), \
                         SW / ((radius + 1) ** 2), SE / ((radius + 1) ** 2)

        # 等式右侧输出形状为{b,c,h,w}
        res_4 = cp.asnumpy(F.convolution_2d(x=im_ch, W=NW, pad=(radius, radius)).array)
        res_5 = cp.asnumpy(F.convolution_2d(x=im_ch, W=NE, pad=(radius, radius)).array)
        res_7 = cp.asnumpy(F.convolution_2d(x=im_ch, W=SE, pad=(radius, radius)).array)

        if radius==1:
            self.image = np.where((act == 0), res_4, self.image)
            self.image = np.where((act == 1), res_5, self.image)
            self.image = np.where((act == 2), res_7, self.image)
        elif radius==2:
            self.image = np.where((act == 3), res_4, self.image)
            self.image = np.where((act == 4), res_5, self.image)
            self.image = np.where((act == 5), res_7, self.image)
        elif radius==3:
            self.image = np.where((act == 6), res_4, self.image)
            self.image = np.where((act == 7), res_5, self.image)
            self.image = np.where((act == 8), res_7, self.image)

    def step(self, act):
        # print(act[0][0])
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        # 调整act形状为 {b,c,h,w}
        act = act[:,np.newaxis,:,:]
        # 使用cupy进行加速卷积计算
        im = cp.asarray(self.image)
        im_ch = im.copy()
        b, c, h, w = im.shape
        for i in range(0,b):
            if np.sum(act[i]==21) > 0:
               bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=3, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(act[i]==22) > 0:
               bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(act[i]==23) > 0:
               bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=7, sigmaColor=0.1, sigmaSpace=5)

            if np.sum(act[i]==24) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=3, sigmaColor=1.0,sigmaSpace=5)
            if np.sum(act[i]==25) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=1.0,sigmaSpace=5)
            if np.sum(act[i]==26) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=7, sigmaColor=1.0,sigmaSpace=5)

            if np.sum(act == 12) > 0:
                median[i, 0] = cv2.medianBlur(self.image[i, 0], 3)
            if np.sum(act == 13) > 0:
                median[i, 0] = cv2.medianBlur(self.image[i, 0], 5)
            if np.sum(act == 14) > 0:
                median[i, 0] = cv2.blur(self.image[i, 0], (7,7))

            if np.sum(act[i]==15) > 0:
                gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(3,3), sigmaX=0.5)
            if np.sum(act[i]==16) > 0:
                gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
            if np.sum(act[i]==17) > 0:
                gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(7,7), sigmaX=0.5)
            if np.sum(act[i]==18) > 0:
               gaussian2[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(3,3), sigmaX=1.5)
            if np.sum(act[i]==19) > 0:
               gaussian2[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5,5), sigmaX=1.5)
            if np.sum(act[i]==20) > 0:
               gaussian2[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(7,7), sigmaX=1.5)

        # act =0,1,2
        self.getEdge(im_ch, act,3,1)
        # act =4,5,6
        self.getEdge(im_ch, act, 5, 2)
        # act =7,8,9
        self.getEdge(im_ch, act, 7, 3)
        move = ((act-10)/255).astype(np.float32)
        self.image = np.where((act == 9) | (act == 10) | (act == 11), self.image + move, self.image)
        self.image = np.where((act==12)|(act==13)|(act==14),median,self.image)
        self.image = np.where((act==15)|(act==16)|(act==17),gaussian,self.image)
        self.image = np.where((act == 18) | (act == 19) | (act == 20), gaussian2, self.image)
        self.image = np.where((act==21)|(act==22)|(act==23),bilateral,self.image)
        self.image = np.where((act==24)|(act==25)|(act==26), bilateral2, self.image)



