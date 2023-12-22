import os

import chainer
import chainer.functions as F
import cv2
import numpy as np
import cupy as cp
import scipy.sparse as ss
import cupyx
import cupy

def show_image(arr,name = './mg.jpg'):
    import cv2
    import numpy as np
    arr = arr[:,:,np.newaxis]
    # arr = np.clip(arr, 0, 1)
    arr = (arr).astype(np.uint8)

    cv2.imwrite(name,arr)

def get_target_imarr(image_arr,size=10):
    # image_arr = chainer.cuda.to_cpu(image_arr)
    c,h,w = image_arr.shape
    res = []
    assert h==w
    num = h//size
    for i in range(c):
       for j in range(num):
           for k in range(num):
               res.append(image_arr[i, j*size:j*size+size, k*size:k*size+size])

    res = cupy.array(res)
    return res
# 基于像素点周围的八邻域 以构造图，其中0表示没有连通，1表示有连通
def connected_adjacency(image, connect=8, patch_size=(1, 1)):
    """
    Construct 8-connected pixels base graph (0 for not connected, 1 for connected)
    """
    r, c = image.shape[:2]
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == "4":
        # constructed from 2 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.ones(c * (r - 1))
        upper_diags = ss.diags([d1, d2], [1, c])
        return upper_diags + upper_diags.T

    elif connect == "8":
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.append([0], d1[: c * (r - 1)])
        d3 = np.ones(c * (r - 1))
        d4 = d2[1:-1]
        upper_diags = ss.diags([d1, d2, d3, d4], [1, c - 1, c, c + 1])
        return upper_diags + upper_diags.T


def supporting_matrix(opt):
    width = opt.width

    pixel_indices = cp.asarray([i for i in range(width * width)])
    pixel_indices = cp.reshape(pixel_indices, (width, width))
    A = cupyx.scipy.sparse.dia_matrix(connected_adjacency(pixel_indices, connect=opt.connectivity))
    A_pair = cp.asarray(cp.where(A.toarray() == 1)).T

    opt.edges = A_pair.shape[0]
    I = cp.eye(width ** 2, width ** 2) #单位矩阵
    A = cp.zeros((width ** 2, width ** 2))
    H = cp.zeros((opt.edges, width ** 2))
    for e, p in enumerate(A_pair):
        H[e, p[0]] = 1
        H[e, p[1]] = -1
        A[p[0], p[1]] = 1

    opt.I = I  # .type(dtype).requires_grad_(True)
    opt.pairs = A_pair
    opt.H = H  # .type(dtype).requires_grad_(True)
    opt.connectivity_full = chainer.Variable(A, requires_grad=True)
    opt.connectivity_idx = cp.where(A > 0)

    for e, p in enumerate(A_pair):
        with chainer.using_config('autograd', False):
            A[p[1], p[0]] = 1

def getL(opt,feature,sigma=1):
    support_L = cp.ones((opt.width**2,1))
    supporting_matrix(opt)
    W = cp.zeros((opt.batch_size, opt.width ** 2, opt.width ** 2))

    # 计算分子
    Fs = (cp.matmul(opt.H, feature.reshape(feature.shape[0], 1, opt.width ** 2, 1)) ** 2)
    w = cp.exp(-(Fs.sum(axis=1)) / (2 * (sigma ** 2)))
    # print("w.shape: {}".format(w.shape))
    W[:,  opt.connectivity_idx[0], opt.connectivity_idx[1]] = w.reshape(opt.batch_size,  -1)
    W[:,  opt.connectivity_idx[1], opt.connectivity_idx[0]] = w.reshape(opt.batch_size,  -1)
    L1 = W @ support_L  #程度矩阵D
    a = (L1.reshape(opt.batch_size, -1))  # (16,1296)
    b = opt.I               # (1296,1296)
    c = F.expand_dims(a, axis=1)
    D = c * b
    # 将array{16,1,1296,1}转化为对角矩阵array{16,1,1296,1296}

    L = chainer.cuda.to_gpu(D.data) - W
    # L输出为三对角矩阵  array{16,1,1296,1296}
    return L, W

def getglr(opt,feature,image,epesoln):
    # print("feature.shape",feature.shape)
    L, W = getL(opt, feature, epesoln)
    # tmp_W = cupy.asnumpy(W).astype('float32')
    # print("tmp_W : {}".format(tmp_W))
    xT = image.reshape(image.shape[0], 1, -1)
    xf = image.reshape(image.shape[0], -1, 1)
    regulizer = (xT @ L @ xf).reshape(-1)

    return regulizer


def save_accu(accu_fout,action,t,t_len=5,a_len=9):
    accu = np.zeros((t_len, a_len * 3))
    for j in range(a_len * 3):
        accu[t][j] = np.sum(action == j)
        # print("t:{}, j:{}, accu_pixel_number: {}".format(t, j, accu[t][j]))
        accu_fout.write("t:{}, j:{}, accu_pixel_number: {}\n".format(t, j, accu[t][j]))
    # print("################################################")
    accu_fout.write("################################################\n")

# 定义颜色映射
color_map = {
    (1, 1, 1): (0, 255, 0),   # 绿色
    (2, 2, 2): (0, 0, 255),   # 蓝色
    (3, 3, 3): (255, 255, 0), # 黄色
    (4, 4, 4): (255, 0, 255), # 品红色
    (5, 5, 5): (0, 255, 255), # 青色
    (6, 6, 6): (128, 0, 0),   # 深红色
    (7, 7, 7): (0, 128, 0),   # 深绿色
    (8, 8, 8): (0, 0, 128),   # 深蓝色
    (9, 9, 9): (255, 0, 0),   # 红色
}
def show_actplt(action,i,t,img_res_path):
    _, h, w = action.shape
    arr = np.ones_like(action)
    arr[np.isin(action, [0, 1, 2])] = 1
    arr[np.isin(action, [3, 4, 5])] = 2
    arr[np.isin(action, [6, 7, 8])] = 3
    arr[np.isin(action, [9, 10, 11])] = 4
    arr[np.isin(action, [12, 13, 14])] = 5

    arr[np.isin(action, [15, 16, 17])] = 6
    arr[np.isin(action, [18, 19, 20])] = 7
    arr[np.isin(action, [21, 22, 23])] = 8
    arr[np.isin(action, [24, 25, 26])] = 9

    arr = np.squeeze(arr).astype(np.uint8)
    bgr_arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    process_img = np.zeros((h, w, 3), dtype=np.uint8)
    # 使用颜色映射为每个像素点着色
    for hid in range(h):
        for wid in range(w):
            process_img[hid, wid] = color_map[tuple(bgr_arr[hid, wid])]

    cv2.imwrite(os.path.join(img_res_path,'[{}th]-act_step_{}.png'.format(i,t)),process_img)

def save_img(image,i,t,img_res_path):

    p = np.maximum(0, image)
    p = np.minimum(1, p)
    p = (p[0] * 255 + 0.5).astype(np.uint8)
    p = np.transpose(p, (1, 2, 0))
    cv2.imwrite(os.path.join(img_res_path,'[{}th]-img_step_{}.png'.format(i,t)), p)