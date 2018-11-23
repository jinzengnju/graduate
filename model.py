#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf

def rnn_cell(FLAGS,dropout):
    if FLAGS.rnn_unit=='rnn':
        rnn_cell_type=tf.nn.rnn_cell.BasicRNNCell
    elif FLAGS.rnn_unit=='gru':
        rnn_cell_type=tf.nn.rnn_cell.GRUCell
    elif FLAGS.rnn_unit=='lstm':
        rnn_cell_type=tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("choose a valid RNN unit type")

    single_cell=rnn_cell_type(FLAGS.num_hidden_units)
    single_cell=tf.nn.rnn_cell.DropoutWrapper(single_cell,output_keep_prob=1-dropout)

    stacked_cell=tf.nn.rnn_cell.MultiRNNCell([single_cell]*FLAGS.num_layers)

    return stacked_cell

def rnn_inputs(FLAGS,input_data):
    with tf.variable_scope('rnn_inputs',reuse=True):
        W_input=tf.get_variable("W_input",[FLAGS.en_vocab_size,FLAGS.num_hidden_units])
    embeddings=tf.nn.embedding_lookup(W_input,input_data)
    return embeddings

def rnn_softmax(FLAGS,outputs):
    with tf.variable_scope('rnn_softmax',reuse=True):
        W_softmax=tf.get_variable("W_softmax",[FLAGS.num_hidden_units,FLAGS.num_classes])
        b_softmax=tf.get_variable("b_softmax",[FLAGS.num_classes])
    logits=tf.matmul(outputs,W_softmax)+b_softmax
    return logits



class Model(object):
    def __init__(self,FLAGS):
        self.inputs_X=tf.placeholder(tf.int32,shape=[None,None],name='inputs_X')
        self.targets_y=tf.placeholder(tf.float32,shape=[None,None],name='targets_y')
        self.seq_lens=tf.placeholder(tf.int32,shape=[None,],name='seq_lens')
        self.dropout=tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

        stacked_cell=rnn_cell(FLAGS,self.dropout)

        with tf.variable_scope('rnn_inputs'):
            W_input=tf.get_variable("W_input",[FLAGS.en_vocab_size,FLAGS.num_hidden_units])
        inputs=rnn_inputs(FLAGS,self.inputs_X)

        all_outputs,state=tf.nn.dynamic_rnn(cell=stacked_cell,inputs=inputs,sequence_length=self.seq_lens,dtype=tf.float32)
        outputs=state[-1][1]


        with tf.variable_scope('rnn_softmax'):
            print(FLAGS.num_classes)
            W_softmax=tf.get_variable("W_softmax",[FLAGS.num_hidden_units,FLAGS.num_classes])
            b_softmax=tf.get_variable("b_softmax",[FLAGS.num_classes])
            tf.summary.histogram('rnn_softmax',W_softmax)

        #outputs，使用最后一层的h_state作为向量进行softmax然后文本分类
        logits=rnn_softmax(FLAGS,outputs)
        probabilities=tf.nn.softmax(logits)

        def get_accuracy(logits,targets_y):
            correct_prediction=tf.equal(tf.argmax(targets_y,1),tf.argmax(logits,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            return accuracy
        #self.accuracy=get_accuracy(self.targets_y,logits)

        self.predict=tf.nn.top_k(logits,5)
        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.targets_y))

        tf.summary.scalar('loss',self.loss)

        self.lr=tf.Variable(0.0,trainable=False)
        trainable_vars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,trainable_vars),FLAGS.max_gradient_norm)
        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_optimizer=optimizer.apply_gradients(zip(grads,trainable_vars),global_step=self.global_step)
        self.merged=tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)

    def step(self,sess,batch_X,batch_seq_lens,batch_y=None,dropout=0.0,forward_only=True):
        input_feed={self.inputs_X:batch_X,
                    self.targets_y:batch_y,
                    self.seq_lens:batch_seq_lens,
                    self.dropout:dropout}
        if forward_only:
            output_feed=[self.merged,self.loss,self.predict]
        else:
            output_feed=[self.merged,self.train_optimizer,self.loss,self.predict]
        outputs=sess.run(output_feed,input_feed)
        if forward_only:
            return outputs[0],outputs[1],outputs[2]
        else:
            return outputs[0],outputs[1],outputs[2],outputs[3]