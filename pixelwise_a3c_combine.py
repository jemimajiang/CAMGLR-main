#下面的导入将帮助那些python3特有的代码也能直接在python2中运行
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
import logging
from logging import getLogger

import chainer
from chainer import functions as F
import numpy as np
from utils import *

from chainerrl import agent
from chainerrl.misc import async_
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept # 这是一个上下文管理器，它在进入上下文之前保存保存链接的当前状态，然后在转义上下文之后恢复保存的状态。

from chainerrl.agents.a3c import A3CModel
import chainerrl
from cached_property import cached_property

logger = getLogger('')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('R_V.txt')
# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class OPT:
    def __init__(
        self,
        batch_size=64*49,width=10,connectivity="4",admm_iter=1,prox_iter=1,delta=1.,channels=1,
        eta=0.1,u=1,u_max=100,u_min=10,lr=1e-4,momentum=0.99,ver=None,train="gauss_batch"):
        self.batch_size = batch_size
        self.width = width
        self.edges = 0
        self.nodes = width ** 2
        self.I = None
        self.pairs = None
        self.H = None
        self.connectivity = connectivity
        self.admm_iter = admm_iter
        self.prox_iter = prox_iter
        self.channels = channels
        self.eta = eta
        self.u = u
        self.lr = lr
        self.delta = delta
        self.momentum = momentum
        self.u_max = u_max
        self.u_min = u_min
        self.ver = ver
        self.D = None
        self.train = train
        self.pg_zero = None

    def _print(self):
        print(
            "batch_size =",self.batch_size,", width =",self.width,", admm_iter =",self.admm_iter,
            ", prox_iter =",self.prox_iter,", delta =",self.delta,", channels =",self.channels,
            ", eta =",self.eta,", u_min =",self.u_min,", u_max =",self.u_max,", lr =",self.lr,
            ", momentum =",self.momentum,
        )

#########重写myentropy，区别在于再将结果在第一维上进行叠加##############
@cached_property
def myentropy(self):
    with chainer.force_backprop_mode():
        return F.stack([- F.sum(self.all_prob * self.all_log_prob, axis=1)], axis=1)
#######################

###########################
def mylog_prob(self, x):
    n_batch, n_actions, h, w = self.all_log_prob.shape
    p_trans = F.transpose(self.all_log_prob, axes=(0,2,3,1))
    p_trans = F.reshape(p_trans,(-1,n_actions))
    x_reshape = F.reshape(x,(1,-1))[0]
    selected_p = F.select_item(p_trans,x_reshape)
    return F.reshape(selected_p, (n_batch,1,h,w))
##########################



class PixelWiseA3C(agent.AttributeSavingMixin, agent.AsyncAgent):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783

    Args:
        model (A3CModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        process_idx (int): Index of the process.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,keep_loss_scale_same=False,normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2,act_deterministically=False,
                 average_entropy_decay=0.999,average_value_decay=0.999,batch_states=batch_states):

        self.process_idx = process_idx
        assert isinstance(model, A3CModel)
        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        # async_.assert_params_not_shared(self.shared_model, self.model)

        self.optimizer = optimizer
        # self.opt = OPT(admm_iter=4, prox_iter=3, delta=.1, channels=1, eta=.05, u=50, lr=8e-6,
        #                momentum=0.9, u_max=65, u_min=50)

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        # self.batch_states = batch_states

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_sec_rewards = {}
        self.average_reward = 0
        # A3C won't use a explorer, but this arrtibute is referenced by run_dqn
        # self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

        #######################
        self.shared_model.to_gpu()
        chainerrl.distribution.CategoricalDistribution.mylog_prob = mylog_prob
        chainerrl.distribution.CategoricalDistribution.myentropy = myentropy
        #######################

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    @property
    def shared_attributes(self):
        return ('shared_model', 'optimizer')

    def update(self, statevar):
        assert self.t_start < self.t

        E_g = 0
        if statevar is None:
            R = 0
        else:
            with state_kept(self.model):
                _, vout,_,_ = self.model.pi_and_v(statevar)
        ######## 转换数据类型 ###############
            R = F.cast(vout.data, 'float32') # 该状态下总回报的期望
        #######################

        pi_loss = 0
        v_loss = 0
        f_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            """E_g的计算"""
            # E_g *= self.gamma
            if i-2 >= self.t_start : #t_start=0,那么当i=2，3，4是E_g才有值
                E_g = self.past_sec_rewards[i-1]-self.past_sec_rewards[i]
                # print("i: {} , E_g = {}".format(i, E_g))
                logger.debug("i: {} , E_g = {}\n".format(i, E_g))

            R += self.past_rewards[i]
            if self.use_average_reward: # false
                R -= self.average_reward
            ###########################
            v = self.past_values[i] # ti时刻的状态值
            advantage = R - v #ti时刻的优势函数
            if self.use_average_reward: # false
                self.average_reward += self.average_reward_tau * float(advantage.data)
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            # Log probability is increased proportionally to advantage
        ##############################
            pi_loss -= log_prob * F.cast(advantage.data, 'float32')
        ##############################
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function
            v_loss += (v - R) ** 2 / 2
            f_loss += E_g

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and \
                self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss *= factor
            v_loss *= factor

        if self.normalize_grad_by_t_max:
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start

        if self.process_idx == 0 and self.t % 30 == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data[0][0], v_loss.data[0][0])

        ##########################
        #total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)
        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)
        total_loss = F.mean(total_loss)
        f_loss /= (self.t_max-1)
        f_loss = F.mean(F.cast(f_loss, 'float32'))
        total_loss -= f_loss * 0.3
        print("total_loss = {}, f_loss = {}\n".format(total_loss, f_loss))
        logger.debug("total_loss = {}, f_loss = {}\n".format(total_loss, f_loss))
        ##########################

        # Compute gradients using thread-specific model
        self.model.zerograds()
        total_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_model.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_model, source_link=self.model)
        # Update the globally shared model
        # if self.process_idx == 0 and self.t % 30 == 0:
        #     norm = sum(np.sum(np.square(param.grad))
        #                for param in self.optimizer.target.params())
        #     logger.debug('grad norm:%s', norm)
        self.optimizer.update()
        # if self.process_idx == 0 and self.t % 30 == 0:
        #     logger.debug('update')

        self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_sec_rewards = {}
        self.past_values = {}

        self.t_start = self.t

    def act_and_train(self, state, reward, epesolun):
    #########################
        statevar = chainer.cuda.to_gpu(state)

        self.past_rewards[self.t - 1] = chainer.cuda.to_gpu(reward)
    ##########################

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        pout, vout,inner_state,hout = self.model.pi_and_v(statevar)
        action = pout.sample().data  # Do not backprop through sampled actions
        self.past_action_log_prob[self.t] = pout.mylog_prob(action)
        self.past_action_entropy[self.t] = pout.myentropy

        self.past_values[self.t] = vout

        #  计算正则项 通过共享特征层最后一层的输出计算
        if self.t % self.t_max > 0:
            feature = F.sum(hout, axis=1)
            # innr = F.sum(inner_state, axis=1)
            # show_image(chainer.cuda.to_cpu (innr.data[0]),'innr_step{}.jpg'.format(self.t))
            # show_image(chainer.cuda.to_cpu (feature.data[0]),'feature_step{}.jpg'.format(self.t))

            target_arrays = get_target_imarr(feature.data)
            image_arr = statevar[:, 0, :]  # {batch,70,70}
            target_imarr = get_target_imarr(image_arr)
            regulizer_arr = getglr(self.opt, target_arrays, target_imarr, epesolun)  # {3136，}
            regulizer = F.reshape(regulizer_arr, (feature.shape[0], -1))
            regulizer = F.sum(regulizer, axis=1)  # 聚合七个分片的正则项
            self.past_sec_rewards[self.t] = regulizer
        else:
            self.past_sec_rewards[self.t] = [0]

        self.t += 1

        # Update stats
        self.average_value += (
                (1 - self.average_value_decay) *
                (F.cast(vout.data, 'float32') - self.average_value))
        ############################
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (F.cast(pout.entropy.data, 'float32') - self.average_entropy))
    #################################
        if self.process_idx == 0 and self.t%30==0:
            logger.debug('t:%s r:%s a:%s ',self.t, reward, action[0][0])
            logger.debug('avg_value:%s avg_entropy:%s ',self.average_value,self.average_entropy)

        return chainer.cuda.to_cpu(action)

        #############################
            #(float(pout.entropy.data[0]) - self.average_entropy))
        #return action
        #############################

    # 测试阶段，根据状态选择动作
    def act(self, obs):
        # Use the process-local model for acting
        with chainer.no_backprop_mode():

            #statevar = self.batch_states([obs], np, self.phi)
            statevar = chainer.cuda.to_gpu(obs)
            pout, _,_,__ = self.model.pi_and_v(statevar)
            if self.act_deterministically: # 返回概率最大的动作
                #return pout.most_probable.data[0]
                return chainer.cuda.to_cpu(pout.most_probable.data)
            else:
                #return pout.sample().data[0]
                return chainer.cuda.to_cpu(pout.sample().data)

    # 训练时，回合结束，更新网络
    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = chainer.cuda.to_gpu(reward)
        if done:
            self.update(None)
        else: # 该分支就没有进去过
            statevar = chainer.cuda.to_gpu(state)
            self.update(statevar)

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    # 在测试阶段使用，表示回合结束，刷新环境
    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()
    # # 没有调用过
    # def load(self, dirname):
    #     super().load(dirname)
    #     copy_param.copy_param(target_link=self.shared_model,
    #                           source_link=self.model)

    def get_statistics(self): # 实际上没有使用，但是不能删除
        return [
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]

