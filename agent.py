
import tensorflow as tf
from time import sleep
import threading
import itertools
import Netural_network
from worker import Worker
from Netural_network import *
from testcheck import reward_test
num_workers=32
with tf.device("/cpu:0"): 
        global_step = tf.Variable(0, name="global_step", trainable=False)
        saver = tf.train.Saver()
        with tf.variable_scope("global"):
                global_NN=NN()
        global_counter = itertools.count()
        check_test=reward_test(global_NN,global_counter,saver)
        workers=[]
        for i in range(num_workers):
                work=Worker(name="agent_{}".format(i),global_Net=global_NN,global_cnt=global_counter,max_global_steps=80000000)
                workers.append(work)
        worker_threads = []
        sess=tf.Session()
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        #latest_checkpoint = tf.train.latest_checkpoint("/home/shashankjain1994/A3C/checkpoint")
        
        saver.restore(sess,"/Users/shashank/Desktop/finisher/breakout")
        
        """for worker in workers:
                worker_work = lambda worker=worker: worker.run(sess, coord)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)"""

        check_thread = threading.Thread(target=lambda: check_test.check(sess, coord))
        check_thread.start()
        coord.join(worker_threads)
        sess.close()
