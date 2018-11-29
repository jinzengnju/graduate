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
        initial_state=stacked_cell.zero_state(FLAGS.batch_size,tf.float32)


        all_outputs,state=tf.nn.dynamic_rnn(initial_state=initial_state,cell=stacked_cell,inputs=inputs,sequence_length=self.seq_lens,dtype=tf.float32)
        outputs=tf.reduce_sum(all_outputs,1)/self.seq_lens


        with tf.variable_scope('rnn_softmax'):
            print(FLAGS.num_classes)
            W_softmax=tf.get_variable("W_softmax",[FLAGS.num_hidden_units,FLAGS.num_classes])
            b_softmax=tf.get_variable("b_softmax",[FLAGS.num_classes])

        #outputs，使用最后一层的h_state作为向量进行softmax然后文本分类
        logits=rnn_softmax(FLAGS,outputs)
        probabilities=tf.nn.softmax(logits)

        def get_accuracy(logits,targets_y):
            correct_prediction=tf.equal(tf.argmax(targets_y,1),tf.argmax(logits,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            return accuracy
        #self.accuracy=get_accuracy(self.targets_y,logits)
        self.predict = tf.nn.top_k(logits, 5)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.targets_y))
        loss_summary=tf.summary.scalar('loss', self.loss)
        self.lr = tf.Variable(0.00001, trainable=False)

        trainable_vars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,trainable_vars),FLAGS.max_gradient_norm)


        grad_and_vars=zip(grads,trainable_vars)
        grad_summaries=[]
        for g,v in grad_and_vars:
            if g is not None:
                grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name),g)
                grad_summaries.append(grad_hist_summary)
                sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged=tf.summary.merge(grad_summaries)
        self.summary=tf.summary.merge([loss_summary,grad_summaries_merged])



        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_optimizer=optimizer.apply_gradients(grad_and_vars,global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)

    def step(self,sess,batch_X,batch_seq_lens,batch_y=None,dropout=0.0,forward_only=True):
        input_feed={self.inputs_X:batch_X,
                    self.targets_y:batch_y,
                    self.seq_lens:batch_seq_lens,
                    self.dropout:dropout}
        if forward_only:
            output_feed=[self.summary,self.loss,self.predict,self.lr]
        else:
            output_feed=[self.summary,self.train_optimizer,self.loss,self.predict,self.lr]
        outputs=sess.run(output_feed,input_feed)
        if forward_only:
            return outputs[0],outputs[1],outputs[2],outputs[3]
        else:
            return outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]
