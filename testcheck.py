import tensorflow as tf 
import numpy as np
import collections
import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from Netural_network  import NN



def make_copy_params_op(v1_list, v2_list):

	v1_list = list(sorted(v1_list, key=lambda v: v.name))
	v2_list = list(sorted(v2_list, key=lambda v: v.name))

	update_ops = []
	for v1, v2 in zip(v1_list, v2_list):
		op = v2.assign(v1)
		update_ops.append(op)

	return update_ops

class reward_test():
		
	def __init__(self):
		#with tf.variable_scope("test"):
        #       	self.test_check=NN()
		self.copy_params_op = make_copy_params_op(tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),tf.contrib.slim.get_variables(scope="test", collection=tf.GraphKeys.TRAINABLE_VARIABLES))
		
		sess=tf.Session()
		new_saver = tf.train.import_meta_graph("/Users/shashank/Desktop/finisher/breakout.meta")
		new_saver.restore(sess,"/Users/shashank/Desktop/finisher/breakout75")
		print(sess.run(tf.report_uninitialized_variables()))
		#tvars = tf.trainable_variables()
		#tvars_vals = sess.run(tvars)

		#for var, val in zip(tvars, tvars_vals):
		#	print("restore :{} {}".format(var.name, val))
		#self.check(sess)


		sess.close()

	def check(self,sess):


		sess.run(self.copy_params_op)


		self.env=gym.make('Breakout-v0')
		self.state=self.atari_make_initial_state(self.env)




		i=0
		rewards=0


			#self.saver.save(sess,"/home/shashankjain1994/A3C/checkpoint/breakout",write_meta_graph=False)	

		for _ in range(10):
			done= False
			while not done :
				self.env.render()
				action=self.test_check.Action_probality(sess,self.state)
				next_state,reward,done,_=self.env.step(action)
				next_state=self.atari_make_next_state(self.state,next_state)
				self.state=next_state
				rewards=reward+rewards
				if done:
					print("reward:{}".format(rewards))
					rewards=0
					self.state=self.atari_make_initial_state(self.env)
		


	def atari_make_initial_state(self,env):
		self.state=env.reset()
		self.state=resize(rgb2gray(self.state),(84,84))
		return np.stack([self.state] * 4, axis=2)



	def atari_make_next_state(self,state, next_state):
		next_state=resize(rgb2gray(next_state),(84,84))
		return np.append(self.state[:,:,1:], np.expand_dims(next_state, 2), axis=2)	


reward_test()

