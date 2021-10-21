# coding=utf-8

import random
import pandas as pd
import tensorflow as tf
tf = tf.compat.v1
tf.disable_eager_execution()

# no need 4
a = tf.constant(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 1, 1, 1, 4, 1, 1, 1, 1],
        [1, 1, 1, 1, 4, 1, 1, 4, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=tf.float32
)
b = tf.equal(a, 4)
c = tf.argmax(b, 1)
d = tf.ones_like(a, dtype=tf.int64) * tf.expand_dims(c, 1)
l = tf.range(tf.shape(a)[1],dtype=tf.int64)
cond = d > l
aim = tf.where(
    cond, tf.ones_like(a),tf.zeros_like(a)
)
with tf.Session() as session:
    print(session.run(b))
    print(session.run(c))
    print(session.run(d))
    print(session.run(l))
    print(session.run(cond))
    print(session.run(aim))
    aim = [
        [1,1,1,0,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
    ]


# need 4
a = tf.constant(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 1, 1, 1, 4, 1, 1, 1, 1],
        [1, 1, 1, 1, 4, 1, 1, 4, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=tf.float32
)
b = tf.equal(a, 4)
c = tf.argmax(b, 1)
c = tf.where(c<=0,tf.ones_like(c)*-1,c)
d = tf.ones_like(a, dtype=tf.int64) * tf.expand_dims(c, 1)
l = tf.range(tf.shape(a)[1],dtype=tf.int64)
cond = d >= l
aim = tf.where(
    cond, tf.ones_like(a),tf.zeros_like(a)
)
with tf.Session() as session:
    print(session.run(b))
    print(session.run(c))
    print(session.run(d))
    print(session.run(l))
    print(session.run(cond))
    print(session.run(aim))
    aim = [
        [1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
    ]


exit(1)
