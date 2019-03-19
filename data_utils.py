#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
import jieba
from train import stopwords
import re
def init():
	f = open('law.txt', 'r', encoding = 'utf8')
	law = {}
	lawname = {}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
        #lawname的键是law文件行号索引，对应值是law的名称即第几条
		law[line.strip()] = len(law)
        #law的键是第几条，值是对应的类标记
		line = f.readline()
	f.close()
	return law,lawname

law, lawname= init()


def getClassNum(kind):
    global law
    global accu
    if kind == 'law':
        return len(law)
#
# def getName(index, kind):
#     global lawname
#     global accuname
#     if kind == 'law':
#         return lawname[index]
#
#     if kind == 'accu':
#         return accuname[index]
# def gettime(time):
#     # 将刑期用分类模型来做
#     v = int(time['imprisonment'])
#     if time['death_penalty']:
#         return 0
#     if time['life_imprisonment']:
#         return 1
#     elif v > 10 * 12:
#         return 2
#     elif v > 7 * 12:
#         return 3
#     elif v > 5 * 12:
#         return 4
#     elif v > 3 * 12:
#         return 5
#     elif v > 2 * 12:
#         return 6
#     elif v > 1 * 12:
#         return 7
#     else:
#         return 8
#
# def getlabel(d, kind):
#     global law
#     global accu
#     # 做单标签
#     if kind == 'law':
#         # 返回多个类的第一个
#         return law[str(d['meta']['relevant_articles'])]
#     if kind == 'accu':
#         return accu[d['meta']['accusation'][0]]
#
#     if kind == 'time':
#         return gettime(d['meta']['term_of_imprisonment'])
_PAD="_PAD"
_GO="_GO"
_EOS="EOS"
_UNK="_UNK"
_START_VOCAB=[_PAD,_GO,_EOS,_UNK]
PAD_ID=0
GO_ID=1
EOS_ID=2
UNK_ID=3

def cut_text(alltext,maxsize):
    train_text = []
    for text in alltext:
        text = re.sub('[^(\\u4e00-\\u9fa5)]', '', text)
        text = re.sub('(?i)[^a-zA-Z0-9\u4E00-\u9FA5]', '', text)
        one_text_res = []
        one_text=[word for word in jieba.cut(text) if len(word)>1]
        for e in one_text:
            if e in stopwords:
                continue
            one_text_res.append(e)
        if (len(one_text_res)>maxsize):
            one_text_res=one_text_res[:maxsize]
        train_text.append(one_text_res)
    return train_text


def sentence_to_token_ids(sentence,vocab_dict):
    return [vocab_dict.get(word,UNK_ID) for word in sentence]

#这里传入的X是分词后的结果
def data_to_token_ids(X,vocab_dict):
    max_len=max(len(sentence) for sentence in X)
    seq_lens=[]
    data_as_tokens=[]
    for line in X:
        token_ids=sentence_to_token_ids(line,vocab_dict)
        #Padding
        data_as_tokens.append(token_ids+[PAD_ID]*(max_len-len(token_ids)))
        seq_lens.append(len(token_ids))
    return data_as_tokens,seq_lens

def get_X_with_word_index(allfact,vocab_dict,max_time_step_size):
    temp=cut_text(allfact,max_time_step_size)
    alltext,seq_lens=data_to_token_ids(temp,vocab_dict)

    alltext=np.array(alltext)
    seq_lens=np.array(seq_lens)
    return alltext,seq_lens
