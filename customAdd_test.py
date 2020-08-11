import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.framework import ops
import numpy as np


customAdd_module = tf.load_op_library('./customAdd.so')

customAddition = customAdd_module.custom_addition

shape = (3,512,512,64)

a_tensor = np.random.random(shape)
b_tensor = np.random.random(shape)

a = tf.placeholder(tf.int32,shape=shape)
b = tf.placeholder(tf.int32,shape=shape)

c = customAddition(a,b)
c_expected = a+b

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config)as sess:
    feed = {a: a_tensor, b: b_tensor}
    result = sess.run(c, feed_dict = feed)
    expected = sess.run(c_expected,feed_dict=feed)
    if (np.array_equal(result,expected)):
        print("Operation Successful!")
    
