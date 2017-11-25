import tensorflow as tf 
import numpy as np
import collections
import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray


from Netural_network  import NN



def make_copy_params_op(v1_list, v2_list):
        """
        Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
        The ordering of the variables in the lists must be identical.
        """
        v1_list = list(sorted(v1_list, key=lambda v: v.name))
        v2_list = list(sorted(v2_list, key=lambda v: v.name))
        update_ops = []
        for v1, v2 in zip(v1_list, v2_list):
                op = v2.assign(v1)
                update_ops.append(op)
        return update_ops

def global_update(local_estimator,global_estimator):
        local_grads, _ = zip(*local_estimator.grads_and_vars)
        # Clip gradients
        local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
        _, global_vars = zip(*global_estimator.grads_and_vars)
        local_global_grads_and_vars = list(zip(local_grads, global_vars))
        return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,global_step=tf.contrib.framework.get_global_step())


class Worker(object):
        def __init__(self, name, global_Net,global_cnt, max_global_steps=None):

                self.name=name
                self.global_step = tf.contrib.framework.get_global_step()
                self.global_NN=global_Net
                self.max_global_steps=max_global_steps
                self.global_counter=global_cnt
                with tf.variable_scope(self.name):
                        self.NN=NN()

                self.global_transfer=global_update(self.NN,self.global_NN)
                self.state = None
                self.copy_params_op = make_copy_params_op(tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        def run(self,sess, coord):

                self.env=gym.make('Breakout-v0')
                self.state=self.atari_make_initial_state(self.env)



                while not coord.should_stop():
                        sess.run(self.copy_params_op)
                        transisition,global_t=self.Q_N_steps(sess,self.env)

                        if global_t > self.max_global_steps:
                                coord.request_stop()
                        self.update_w(sess,transisition)



        def Q_N_steps(self,sess,env):
                trans=collections.namedtuple('trans',['state','action','next_state','reward','done'])
                transisition=[]
                i=0
                Pre_info=5
                for _ in range(32):
                        action=self.NN.Action_probality(sess,self.state)
                        next_state,reward,done,info=env.step(action)
                        info= info['ale.lives']
                        if info < Pre_info:
                        	reward=-5
                        Pre_info=info
                        next_state=self.atari_make_next_state(self.state,next_state)
                        reward = max(min(reward, 1), -1)
                        transisition.append(trans(state=self.state,action=action,next_state=next_state,reward=reward,done=done))
                        if done:
                                self.next_state=self.atari_make_initial_state(env)
                                Pre_info=5
                        self.state=next_state


                global_t = next(self.global_counter)
                return transisition,global_t


        def atari_make_initial_state(self,env):
                state=env.reset()
                state=resize(rgb2gray(state),(84,84))
                return np.stack([state] * 4, axis=2)



        def atari_make_next_state(self,state, next_state):
                next_state=resize(rgb2gray(next_state),(84,84))
                return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

        def update_w(self,sess,transitions):
                reward=0
                if not transitions[-1].done:
                        reward=self.NN.Value_esitimation(sess,transitions[-1].next_state)

                states=[]
                actions=[]
                advantages=[]
                rewards=[]

                for transition in transitions[::-1]:
                        reward = transition.reward + 0.99 * reward

                        advantage = reward - self.NN.Value_esitimation(sess,transition.state)
                        states.append(transition.state)
                        actions.append(transition.action)
                        advantages.append(advantage)
                        rewards.append(reward)


                feed_dic={
                self.NN.states:np.array(states),
                self.NN.advantage:advantages,
                self.NN.actions:actions,
                self.NN.reward:rewards
                }
                loss,_=sess.run([self.NN.loss,self.global_transfer],feed_dic)
