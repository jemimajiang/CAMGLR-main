import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor, weight, bias):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D( in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False, initialW=weight, initial_bias=bias),
            #bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h

# class CA_Block(chainer.Chain):
#     # TODO
#     def __init__(self, crop_size):
#         super(CA_Block, self).__init__(
#             h=crop_size,
#             w=crop_size,
#
#         )
#
#
#         self.train = True
#
#     def __call__(self, x):
#         # 输入尺寸{64，C，70，70}
#
#         return x
import chainer
import chainer.functions as F
import chainer.links as L

class CA_Block(chainer.Chain):
    def __init__(self, h,w, reduction=64):
        super(CA_Block, self).__init__()
        with self.init_scope():
            self.h = h
            self.w = w
            self.reduction = reduction

            self.avg_pool_x = lambda x: F.average_pooling_2d(x, (self.h, 1))
            self.avg_pool_y = lambda x: F.average_pooling_2d(x, (1, self.w))

            self.conv_1x1 = L.Convolution2D(64, 64 // reduction,
                                            ksize=1,stride=1, nobias=True)

            self.F_h = L.Convolution2D(64 // reduction, 64, ksize=1, nobias=True)
            self.F_w = L.Convolution2D(64 // reduction, 64, ksize=1, nobias=True)
        self.train=True

    def __call__(self, x):
        x_h = self.avg_pool_x(x)
        x_w = self.avg_pool_y(x).transpose((0, 1, 3, 2))

        x_cat_conv_relu = F.relu(self.conv_1x1(F.concat((x_h, x_w), axis=3)))
        # x_cat_conv_relu:{64,1,1,140}
        x_cat_conv_split_h, x_cat_conv_split_w = F.split_axis(x_cat_conv_relu, 2, axis=3)
        # x_cat_conv_split_h::{64,1,1,70}
        # x_cat_conv_split_w::{64,1,1,70}
        s_h = F.sigmoid(self.F_h(x_cat_conv_split_h.transpose((0, 1, 3, 2))))
        s_w = F.sigmoid(self.F_w(x_cat_conv_split_w))
        # s_h.expand_as(x)
        out = x * F.broadcast_to(s_h,x.shape) * F.broadcast_to(s_w,x.shape)

        return out

class MyFcn(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions,crop_size):
        w = chainer.initializers.HeNormal()
        net = CaffeFunction('../initial_weight/zhang_cvpr17_denoise_50_gray.caffemodel')
        super(MyFcn, self).__init__(
            conv1=L.Convolution2D(1, 64, 3, stride=1, pad=1, nobias=False, initialW=net.layer1.W.data,
                                  initial_bias=net.layer1.b.data),
            diconv2=DilatedConvBlock(2, net.layer3.W.data, net.layer3.b.data),
            diconv3=DilatedConvBlock(3, net.layer6.W.data, net.layer6.b.data),
            diconv4=DilatedConvBlock(4, net.layer9.W.data, net.layer9.b.data),
            diconv5_pi=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_pi=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_mlp=L.Convolution2D( 64, n_actions, 3, stride=1, pad=1, nobias=False, initialW=w),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(
                L.Convolution2D(n_actions, n_actions * 3, 3, stride=1, pad=1, nobias=False, initialW=w)),

            diconv5_V=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_V=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_V=L.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False, initialW=net.layer18.W.data,
                                    initial_bias=net.layer18.b.data),
            # TODO
            CAM = CA_Block(crop_size,crop_size)
        )
        self.train = True
 
    def pi_and_v(self, x):
         
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)

        #CAM
        # 代码思路来源：论文：https://arxiv.org/abs/2103.02907代码链接（刚刚开源）：https://github.com/Andrew-Qibin/CoordAttention
        # https://blog.csdn.net/amusi1994/article/details/114559459?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-114559459-blog-117328186.pc_relevant_multi_platform_whitelistv2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
        h = self.CAM(h)

        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        hout = self.conv7_mlp(h_pi)
        pout = self.conv7_pi(hout)

        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)
       
        return pout, vout, h_pi, hout
