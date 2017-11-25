import numpy as np
import tensorflow as tf



class NN():


        def Action_probality(self,sess,X):
                state=np.reshape(X, [-1, 84, 84, 4])
                prob_weights = sess.run(self.policy, feed_dict={self.states: state})
                print("Prob_weights:{}".format(prob_weights))
                action=np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
                return action

        def Value_esitimation(self,sess,X):
                state=np.reshape(X, [-1, 84, 84, 4])
                value=sess.run(self.value, feed_dict={self.states: state})
                return value[0][0]



        def build_NN(self):
                input_layer = tf.reshape(self.states, [-1, 84, 84, 4])
                conv1 = tf.contrib.layers.conv2d(input_layer, 16, 8, 4, activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),scope="conv1")
                conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(),scope="conv2")
                fc1 = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv2),num_outputs=256,weights_initializer=tf.contrib.layers.xavier_initializer(),scope="fc1")
                self.policy=tf.contrib.layers.fully_connected(fc1, 4, activation_fn=tf.nn.softmax,weights_initializer=tf.contrib.layers.xavier_initializer(),scope="policy")
                self.value=tf.contrib.layers.fully_connected(inputs=fc1,num_outputs=1,weights_initializer=tf.contrib.layers.xavier_initializer(),scope="value")

                return self.policy,self.value

        def __init__(self):


                self.states = tf.placeholder(shape=[None,84, 84, 4], dtype=tf.float32)


                self.policy,self.value=self.build_NN()


                self.advantage = tf.placeholder(shape=[None], dtype=tf.float32,name="advatage")
                self.actions=tf.placeholder(shape=[None], dtype=tf.uint8,name="actions")
                self.reward=tf.placeholder(shape=[None], dtype=tf.float32,name="rewards")
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy), 1)
                self.policy_red=tf.log(tf.reduce_sum(self.policy*tf.one_hot(self.actions, 4, dtype=tf.float32),1))
                self.actor_loss= tf.reduce_sum(-(self.policy_red*self.advantage)-self.entropy*0.01)
                self.critic_loss=tf.reduce_sum(tf.square(self.reward - tf.reshape(self.value,[-1])))
                self.loss=0.25*self.critic_loss+self.actor_loss

                self.optimizer = tf.train.RMSPropOptimizer(0.0007, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]