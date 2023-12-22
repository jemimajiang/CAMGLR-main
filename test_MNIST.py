import numpy as np
#TODO:使用的是不一样的图像预处理操作
from v0_batch_loader import *
# from chainer import serializers
from MyFCN_combine import *
import sys
import time
import State_combine as State
import os
import utils
from pixelwise_a3c_combine import *

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training.txt"
TESTING_DATA_PATH           = "../MNIST/test_MNIST.txt"
IMAGE_DIR_PATH              = "../"
MODEL_PATH            = "./my_model"
MODEL_NAME             ='/model.npz'
IMG_RES_PATH = 'result_P/'

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 50000
# N_EPISODES           = 1
EPISODE_LEN = 5
SNAPSHOT_EPISODES  = 3000
TEST_EPISODES = 30
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 50

N_ACTIONS = 9 # 动作数
CROP_SIZE = 70

GPU_ID = 0
EPESOLN = 10

def test(loader, agent, fout):
    accu_fout = open(IMG_RES_PATH + 'accu.txt', 'w')
    sum_psnr = 0
    sum_reward = 0

    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        np.random.seed(1)
        raw_n = np.random.normal(MEAN, SIGMA, raw_x.shape).astype(raw_x.dtype) / 255
        current_state.reset(raw_x, raw_n)
        # zero_r = np.zeros(raw_x.shape, raw_x.dtype)

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action = agent.act(current_state.image)
            utils.save_accu(accu_fout,action,t,EPISODE_LEN,N_ACTIONS)
            current_state.step(action)
            utils.show_actplt(action,i,t,IMG_RES_PATH)
            utils.save_img(current_state.image,i,t,IMG_RES_PATH)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            # zero_r += (reward == 0)

            # todo:统计reward中正样本和负样本的比例
            # positive_r = np.sum(reward > 0)
            # negtive_r = np.sum(reward < 0)
            # zero_r = np.sum(reward == 0)
            # print("t:{}, 正样本：{}, 负样本：{}, 零：{}".format(t, positive_r, negtive_r, zero_r))
            # fout.write("t:{}, 正样本：{}, 负样本：{}, 零：{}\n".format(t, positive_r, negtive_r, zero_r))
            # sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agent.stop_episode()
        # fout.write("sparse_reward [2],[3],[4] number: {}, {}, {}\n".
        #            format(np.sum(zero_r >= 2), np.sum(zero_r >= 3), np.sum(zero_r >= 4)))
        # print("sparse_reward [2],[3],[4] number: {}, {}, {}".
        #       format(np.sum(zero_r >= 2), np.sum(zero_r >= 3), np.sum(zero_r >= 4)))

        I = np.maximum(0, raw_x)
        I = np.minimum(1, I)
        N = np.maximum(0, raw_x + raw_n)
        N = np.minimum(1, N)
        p = np.maximum(0, current_state.image)
        p = np.minimum(1, p)
        I = (I[0] * 255 + 0.5).astype(np.uint8)
        N = (N[0] * 255 + 0.5).astype(np.uint8)
        p = (p[0] * 255 + 0.5).astype(np.uint8)
        p = np.transpose(p, (1, 2, 0))
        I = np.transpose(I, (1, 2, 0))
        N = np.transpose(N, (1, 2, 0))
        cv2.imwrite(IMG_RES_PATH + str(i) + '_output.png', p)
        # cv2.imwrite(IMG_RES_PATH + str(i) + '_clean.png', I)
        # cv2.imwrite(IMG_RES_PATH + str(i) + '_input.png', N)
        cv2.imwrite(IMG_RES_PATH + str(i) + '_p-I.png', p - I)
        psnr = cv2.PSNR(p, I)
        opsnr = cv2.PSNR(N, I)
        print("{ith}-th image PSNR is {res} \t origin is {o}".format(ith=i, res=psnr, o=opsnr))
        fout.write("{ith}-th image PSNR is {res} \t origin is {o}\n".format(ith=i, res=psnr, o=opsnr))
        sum_psnr += psnr

    print("test total reward {a}, PSNR {b}".format(a=sum_reward * 255 / test_data_size, b=sum_psnr / test_data_size))
    fout.write(
        "test total reward {a}, PSNR {b}\n".format(a=sum_reward * 255 / test_data_size, b=sum_psnr / test_data_size))
    sys.stdout.flush()


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE))

    # load myfcn model
    model = MyFcn(N_ACTIONS,CROP_SIZE)
    #TODO: count params numbers
    cout_parm = count_parameters(model)
    unitM = calculate_model_size(cout_parm)
    fout.write("Total number of parameters:{}\n".format(cout_parm))
    fout.write("Total parameters:{} MiB\n".format(unitM))
    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz(MODEL_PATH + MODEL_NAME, agent.model)
    # loaded = np.load('./model/entire_dataset_15.npz',allow_pickle=True)['data']
    # agent.model = loaded.tolist()

    agent.act_deterministically = True
    agent.model.to_gpu()

    # _/_/_/ testing _/_/_/
    # test(mini_batch_loader, agent, fout)

def count_parameters(model):
    count = 0
    for param in model.params():
        count += param.data.size
    return count

def calculate_model_size(parameters, bytes_per_parameter=4):
    total_bytes = parameters * bytes_per_parameter
    total_megabytes = total_bytes / (1024 * 1024)
    return total_megabytes
if __name__ == '__main__':
    try:
        if not os.path.exists(IMG_RES_PATH):
            os.makedirs(IMG_RES_PATH)
        fout = open(IMG_RES_PATH + 'param_count.txt', "w")
        fout.write("model pth:{}\n".format(MODEL_PATH + MODEL_NAME))
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(error.message)
