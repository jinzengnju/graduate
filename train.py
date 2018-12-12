#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import json
from read_data import *
import time
from model import Model
from data_utils import *
from MacroWithPython import *
import yaml
from gensim.models import Word2Vec

#Configs
tf.app.flags.DEFINE_string("rnn_unit",'lstm',"Type of RNN unit:rnn|gru|lstm.")
tf.app.flags.DEFINE_float("learning_rate",1e-5,"Learning Rate.")
tf.app.flags.DEFINE_float("max_gradient_norm",5.0,"Clip gradients to this norm")
tf.app.flags.DEFINE_integer("batch_size",64,"Batch size to use during training")
tf.app.flags.DEFINE_integer("num_hidden_units",300,"Number of hidden units in each RNN unit")
tf.app.flags.DEFINE_integer("num_layers",2,"NUmber of layers in the model")
tf.app.flags.DEFINE_float("dropout",0.5,"Amount to drop during training")
tf.app.flags.DEFINE_integer("en_vocab_size",10000,"English vocabulary size")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoints","Directory to save the model checkpoints")
tf.app.flags.DEFINE_integer("num_classes","2"," ")
tf.app.flags.DEFINE_string("input_traindata","/home/jin/data/cail_0518/temp/TFrecords/train.tfrecords","训练数据路径")
tf.app.flags.DEFINE_string("input_validdata","/home/jin/data/cail_0518/temp/TFrecords/test.tfrecords","验证数据路径")
tf.app.flags.DEFINE_integer("valid_step",100,'训练多少步后执行一次验证并保存模型')
tf.app.flags.DEFINE_integer("valid_num_batch",10,'执行valid时跑多少个batch的数据')
tf.app.flags.DEFINE_string("log_dir","/home/jin/data/log",'')
tf.app.flags.DEFINE_string("vocab_dict","/home/jin/data/vocab_dict",'')
tf.app.flags.DEFINE_string("config","/home/jin/data/vocab_dict",'')
tf.app.flags.DEFINE_integer("max_time_step_size",600,'')
tf.app.flags.DEFINE_string("w2v_model","/home/jin/data",'')
tf.app.flags.DEFINE_integer("pos_weight",90,'')
tf.app.flags.DEFINE_integer("embedding_size","150",'')
tf.app.flags.DEFINE_string("valid_logdir","/home/jin/log",'')
tf.app.flags.DEFINE_float("sig_value",0.5,'')
FLAGS=tf.app.flags.FLAGS

def create_model(sess,FLAGS,embedding_matrix):
    text_model=Model(FLAGS,embedding_matrix)
    ckpt=tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring old model parameters from %s"%ckpt.model_checkpoint_path)
        text_model.saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        print("Create new Model")
        sess.run(tf.global_variables_initializer())
    return text_model

def save_model(model,sess,step_index):
    if not os.path.isdir(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)
    checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
    print("Saving the model and global_step is:",step_index)
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def get_EmbeddingMatrix(sorted_word):
    model=Word2Vec.load(FLAGS.w2v_model)
    embedding_matrix=np.random.randn(FLAGS.en_vocab_size,FLAGS.embedding_size)
    for word,index in sorted_word.items():
        try:
            embedding_vector=model[word]
            embedding_matrix[index]=embedding_vector
        except KeyError:
            embedding_vector=embedding_matrix[3]
            embedding_matrix[index]=embedding_vector
    return embedding_matrix

def train(vocab_dict):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    gpuConfig=tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth=True
    judge=Judger()
    embedding_matrix = get_EmbeddingMatrix(vocab_dict)
    f_write=open(FLAGS.valid_logdir,'w')
    with tf.Graph().as_default(), tf.Session(config=gpuConfig) as sess:
        train_fact, train_laws = inputs(FLAGS.input_traindata, FLAGS.batch_size,FLAGS.num_classes)
        valid_fact,valid_laws=inputs(FLAGS.input_validdata,FLAGS.batch_size,FLAGS.num_classes)
        model =create_model(sess,FLAGS,embedding_matrix)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        valid_writer=tf.summary.FileWriter(FLAGS.log_dir+'/valid')
        try:
            step=0
            start_time = time.time()
            sess.run(tf.assign(model.lr,FLAGS.learning_rate))
            while not coord.should_stop():#这里是永远不会停止的，因为epoch设置的是NOne
                train_fact_v,train_law_v=sess.run([train_fact, train_laws])
                #print([bytes.decode(e) for e in train_fact_v])
                train_fact_val,train_seq_lens=get_X_with_word_index(train_fact_v,vocab_dict,FLAGS.max_time_step_size)
                summary_train,_, loss, predict_result,lr= model.step(sess,train_fact_val,train_seq_lens,train_law_v,dropout=FLAGS.dropout,
                                               forward_only=False)
                #predict_result是batch内每个样本的预测类标记0-182
                #train_law_v是经过one—hot编码的label向量0-182
                #上面两个均为np.array类型
                if step%5==0:
                    step_index = sess.run(model.global_step)
                    print('Step %d:train loss=%.6f' % (step_index, loss))

                if step%(FLAGS.valid_step)==0:
                    predict=[np.where(e >= FLAGS.sig_value)[0] for e in predict_result]
                    true_label=[np.where(e == 1)[0] for e in train_law_v]
                    for num_i in range(len(predict)):
                        print("预测*******************************************")
                        print(predict[num_i])
                        print("真实*******************************************")
                        print(true_label[num_i])
                    time_use = time.time() - start_time
                    print("***********************************************")
                    step_index=sess.run(model.global_step)
                    save_model(model,sess,step_index)
                    print('Step %d:train loss=%.6f(%.3sec)'%(step_index,loss,time_use))
                    train_writer.add_summary(summary_train,step_index)
                    valid_loss = 0
                    accracy=np.zeros(7)
                    for _ in range(FLAGS.valid_num_batch):
                        #print("验证一下")
                        #print(lr)
                        valid_fact_v, valid_law_v = sess.run([valid_fact,valid_laws])
                        valid_fact_val, valid_seq_lens = get_X_with_word_index(valid_fact_v, vocab_dict,FLAGS.max_time_step_size)
                        summary_valid,loss, valid_predict,lr= model.step(sess, valid_fact_val, valid_seq_lens, valid_law_v, dropout=1.0,
                                                       forward_only=True)

                        valid_loss+=loss
                        temp=judge.getAccuracy(predict=valid_predict,truth=valid_law_v,sig_value=FLAGS.sig_value)
                        accracy+=np.array(temp)
                    accracy=accracy/FLAGS.valid_num_batch*1.0
                    json.dump(accracy.tolist(),f_write)
                    f_write.write('\n')
                    valid_loss_res=valid_loss/FLAGS.valid_num_batch
                    valid_accu_res=accracy[6]
                    valid_loss_summary=tf.Summary(value=[tf.Summary.Value(tag="valid_loss",simple_value=valid_loss_res)])
                    valid_accu_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_accu", simple_value=valid_accu_res)])
                    valid_writer.add_summary(valid_loss_summary,step_index)
                    valid_writer.add_summary(valid_accu_summary, step_index)
                    print("valid loss=%.6f and accuracy=%.5f"%(valid_loss_res,valid_accu_res))
                    start_time=time.time()
                step+=1
        except tf.errors.OutOfRangeError:
            print('Done training for %d steps'%step)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        f_write.close()

class TrainConfig:
    def __init__(self,path):
        f = open(path, encoding='utf-8')
        self.configs_dict = yaml.load(f)
    def get(self,name):
        return self.configs_dict.get(name)

class PreProcess:
    def __init__(self,trainconfig):
        FLAGS.rnn_unit=trainconfig.get("rnn_unit")
        FLAGS.learning_rate=trainconfig.get("learning_rate")
        FLAGS.max_gradient_norm=trainconfig.get("max_gradient_norm")
        FLAGS.batch_size=trainconfig.get("batch_size")
        FLAGS.num_hidden_units=trainconfig.get("num_hidden_units")
        FLAGS.num_layers=trainconfig.get("num_layers")
        FLAGS.dropout=trainconfig.get("dropout")
        FLAGS.en_vocab_size=trainconfig.get("en_vocab_size")
        FLAGS.ckpt_dir=trainconfig.get("ckpt_dir")
        FLAGS.num_classes=trainconfig.get("num_classes")
        FLAGS.input_traindata=trainconfig.get("input_traindata")
        FLAGS.input_validdata=trainconfig.get("input_validdata")
        FLAGS.valid_step=trainconfig.get("valid_step")
        FLAGS.valid_num_batch=trainconfig.get("valid_num_batch")
        FLAGS.log_dir=trainconfig.get("log_dir")
        FLAGS.vocab_dict=trainconfig.get("vocab_dict")
        FLAGS.max_time_step_size=trainconfig.get("max_time_step_size")
        FLAGS.w2v_model=trainconfig.get("w2v_model")
        FLAGS.pos_weight=trainconfig.get("pos_weight")
        FLAGS.embedding_size=trainconfig.get("embedding_size")
        FLAGS.valid_logdir=trainconfig.get("valid_logdir")
        FLAGS.sig_value=trainconfig.get("sig_value")
    def before_train(self):
        law_num = getClassNum("law")
        FLAGS.num_classes=law_num
        f_read1 = open(FLAGS.vocab_dict, 'r')
        vocab_dict = json.load(f_read1)
        f_read1.close()
        return vocab_dict



def main(unuse_args):
    traincofig=TrainConfig(FLAGS.config)
    preprocess=PreProcess(traincofig)
    vocab_dict=preprocess.before_train()
    train(vocab_dict)

if __name__=="__main__":
    tf.app.run()
