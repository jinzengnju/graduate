#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
from attention import attention

class Model(object):
    def rnn_cell(self,FLAGS,dropout):
        single_cell=tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden_units,forget_bias=1.0,initializer=tf.glorot_normal_initializer())
        single_cell=tf.nn.rnn_cell.DropoutWrapper(single_cell,output_keep_prob=dropout)
        return single_cell

    def __init__(self,FLAGS,embedding_matrix):
        self.inputs_X=tf.placeholder(tf.int32,shape=[None,None],name='inputs_X')
        self.targets_y=tf.placeholder(tf.float32,shape=[None,None],name='targets_y')
        self.seq_lens=tf.placeholder(tf.float32,shape=[None,],name='seq_lens')
        self.dropout=tf.placeholder(tf.float32)
        self.topic_vector = tf.placeholder(tf.float32, shape=[None, None], name='topic_vector')
        self.global_step = tf.Variable(0, trainable=False)
        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            self.embedding=tf.Variable(initial_value=embedding_matrix,dtype=tf.float32,name="Embedding",trainable=True)
            inputs=tf.nn.embedding_lookup(self.embedding,self.inputs_X)

        stacked_cell=tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell(FLAGS,self.dropout) for _ in range(FLAGS.num_layers)])

        initial_state=stacked_cell.zero_state(FLAGS.batch_size,tf.float32)
        all_outputs,state=tf.nn.dynamic_rnn(initial_state=initial_state,cell=stacked_cell,inputs=inputs,sequence_length=self.seq_lens,dtype=tf.float32)

        out_before_attention=tf.summary.histogram("out_before_attention",all_outputs)

        with tf.name_scope('attention_layer'):
            outputs_attention=attention(all_outputs,256,self.topic_vector,time_major=False)

        #outputs_attention=outputs_attention/self.seq_lens[:,None]
        #outputs_attention=tf.reduce_sum(all_outputs,1)/self.seq_lens[:,None]
        out_after_attention = tf.summary.histogram("out_after_attention", outputs_attention)
        logits = tf.layers.dense(inputs=outputs_attention, units=FLAGS.num_classes,activation=None,kernel_initializer=tf.glorot_normal_initializer())  # 默认不用激活函数激活
        #self.probablities=tf.nn.sigmoid(logits)

        self.predict=tf.nn.top_k(logits,6,sorted=True)
        #predict是从0开始的label

        def get_accuracy(logits,targets_y):
            correct_prediction=tf.equal(tf.argmax(targets_y,1),tf.argmax(logits,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            return accuracy

        self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets_y,logits,FLAGS.pos_weight))
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.targets_y))
        loss_summary=tf.summary.scalar('loss', self.loss)
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,trainable_vars),FLAGS.max_gradient_norm)
        grad_summaries=[]
        for g,v in zip(grads,trainable_vars):
            if g is not None:
                grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name),g)
                grad_summaries.append(grad_hist_summary)
                sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged=tf.summary.merge(grad_summaries)
        self.summary=tf.summary.merge([loss_summary,out_before_attention,out_after_attention,grad_summaries_merged])
        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_optimizer=optimizer.apply_gradients(zip(grads,trainable_vars),global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)

    def step(self,sess,batch_X,batch_seq_lens,batch_y,topic_vector,dropout=1.0,forward_only=True):
        input_feed={self.inputs_X:batch_X,
                    self.targets_y:batch_y,
                    self.seq_lens:batch_seq_lens,
                    self.dropout:dropout,
                    self.topic_vector:topic_vector}
        if forward_only:
            output_feed=[self.summary,self.loss,self.predict,self.lr]
        else:
            output_feed=[self.summary,self.train_optimizer,self.loss,self.predict,self.lr]
        outputs=sess.run(output_feed,input_feed)
        if forward_only:
            return outputs[0],outputs[1],outputs[2],outputs[3]
        else:
            return outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]
