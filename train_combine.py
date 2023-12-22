from mini_batch_loader import *
from chainer import serializers
from MyFCN_combine import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State_combine as State
import os
from pixelwise_a3c_combine import *

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training.txt"
TESTING_DATA_PATH           = "../MNIST/test_MNIST.txt"
IMAGE_DIR_PATH              = "../datasets/"
SAVE_PATH            = "./len5_model/"
 
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
accu_fout = open('test_accu.txt','w')
def test(loader, agent, fout,flag_accu=False):
    sum_psnr     = 0
    sum_reward = 0
    sum_noise_psnr = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        np.random.seed(1)
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
        current_state.reset(raw_x,raw_n)
        # reward = np.zeros(raw_x.shape, raw_x.dtype)*255

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action = agent.act(current_state.image)
            save_accu(accu_fout,action[0],t,t_len=EPISODE_LEN,a_len=N_ACTIONS) if flag_accu==True else None
            current_state.step(action)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        # print("acc: {}".format(accu))
        agent.stop_episode()

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

        psnr = cv2.PSNR(p, I)
        Npsnr = cv2.PSNR(N, I)

        fout.write("{ith}-th image PSNR is {res}, noise PSNR is {b},\n sub = {c}\n".format(ith=i, res=psnr, b=Npsnr,
                                                                                           c=(psnr - Npsnr)))
        sum_psnr += psnr
        sum_noise_psnr += Npsnr

    avg_psnr = sum_psnr / test_data_size
    print("test total reward {a}, PSNR {b}, noise PSNR {c}, sub = {d}".format(a=sum_reward * 255 / test_data_size,
                                                                              b=avg_psnr,
                                                                              c=sum_noise_psnr / test_data_size,
                                                                              d=(sum_psnr - sum_noise_psnr) / test_data_size))
    fout.write(
        "test total reward {a}, PSNR {b}, noise PSNR {c}, sub = {d}\n".format(a=sum_reward * 255 / test_data_size,
                                                                              b=avg_psnr,
                                                                              c=sum_noise_psnr / test_data_size,
                                                                              d=(sum_psnr - sum_noise_psnr) / test_data_size))
    print('======================== finished =============================')
    fout.write('======================== finished ============================\n')
    sys.stdout.flush()
    return avg_psnr


def main(fout):
    #_/_/_/ load dataset 载入数据集 _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use() # 获取gpu运行设备

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
 
    # load myfcn model
    model = MyFcn(N_ACTIONS,CROP_SIZE)
 
    #_/_/_/ setup _/_/_/
 
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    max_psnr = 0
    for episode in range(1, N_EPISODES+1):
        # display current episode
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        # generate noise
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
        # initialize the current state and reward
        current_state.reset(raw_x,raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action = agent.act_and_train(current_state.image, reward, EPESOLN)
            current_state.step(action)
            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode_and_train(current_state.image, reward, True)
        print("train total reward {a}".format(a=sum_reward*255))
        fout.write("train total reward {a}\n".format(a=sum_reward*255))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0 or episode==3 or episode==N_EPISODES or episode==15:
            # _/_/_/ testing _/_/_/

            if episode % TEST_EPISODES == 0:
                avg_psnr = test(mini_batch_loader, agent, fout,flag_accu=True)
            else:
                avg_psnr = test(mini_batch_loader, agent, fout)
            if avg_psnr > max_psnr:
                max_psnr = avg_psnr
                agent.save(SAVE_PATH + str(episode) + "_P_{}".format(avg_psnr))

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+"snap_"+str(episode)+"_P_{}".format(avg_psnr))
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
     
 
if __name__ == '__main__':
    try:
        start = time.time()
        local_time = time.localtime(start)
        standard_time = time.strftime("%Y%m%d_%H%M%S", local_time)
        SAVE_PATH += standard_time + '/'
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        fout = open(SAVE_PATH + 'log_' + standard_time + '_.txt', "w")  # 设置日志文件

        main(fout) #训练
        end = time.time()

        #得到训练的总时间
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
