#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
def attention(inputs,attention_size,topic_vector,time_major=False):
    if time_major:
        #强行將维度信息改为（B，T，D）
        inputs=tf.array_ops.transpose(inputs,[1,0,2])
    batch_size=inputs.shape[0].value
    hidden_size=inputs.shape[2].value
    #获取hiddensize的维度，假设为D
    #这里的attentionsize的维度为训练LDA主题向量的维度，因为要用主题向量来attention encoder的隐藏向量
    with tf.variable_scope("attention_W"):
        w_omega=tf.get_variable("w_omega",[hidden_size,attention_size],initializer=tf.glorot_normal_initializer())
        #b_omega=tf.get_variable("b_omega",[attention_size],initializer=tf.glorot_normal_initializer())
    # topic_vector=tf.reshape(topic_vector,[batch_size,attention_size])
    # with tf.name_scope("topic_fullconnect"):
    #     u_omega=tf.layers.dense(inputs=topic_vector,units=attention_size,activation=None,kernel_initializer=tf.glorot_normal_initializer())
    u_omega=topic_vector

    #u_omega维度为bacth*attention_size
    temp=tf.tensordot(inputs,w_omega,axes=1)

    temp=tf.reshape(temp,[batch_size,-1,attention_size])
    #將input的维度用全连接层转为（B，T，attention_size），以便可以与topic_vactor做内乘
    #v=tf.tanh(temp+b_omega)
    vu=tf.matmul(temp,tf.expand_dims(u_omega,-1))
    #vu=tf.matmul(v,tf.reshape(u_omega,[batch_size,attention_size,1]))
    vu=tf.reshape(tf.squeeze(vu),[batch_size,-1])
    alphas=tf.nn.softmax(vu,name='alphas')
    #B*T的形状
    output=tf.reduce_sum(inputs*tf.expand_dims(alphas,-1),1)
    return output