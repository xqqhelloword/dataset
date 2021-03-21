"""
transform txt to vector
"""
import re

import gensim
import jieba
import pandas as pd

from special_string import *
from static_num import *


def stopwordslist():
    stopwords = [line.strip() for line in open(STOP_LIST_PATH, encoding='UTF-8').readlines()]
    return stopwords


def trans_to_sentence_mode(train_frame,test_frame,stop_list):
    """

    :param train_frame: 训练集数据框,Id,Text（文本,未分词）,Label
    :param test_frame: 测试集数据框，Id,Text,Label
    :param stop_list: 停用词,list
    :return:训练集，测试集数据框，Id,Text（分好词，用空格连接的字符串）,Label
    """
    print('feature_engineer:trans_to_sentence_mode')
    jieba.load_userdict(USER_DIC_PATH)
    # 1.分词 2.转为词向量
    train_frame['Text'] = train_frame['Text'].apply(lambda x: re.sub(r'[^\u4E00-\u9FD5\d\w\]\[://@.]+', '', x))
    test_frame['Text'] = test_frame['Text'].apply(lambda x: re.sub(r'[^\u4E00-\u9FD5\d\w\]\[://@.]+', '', x))
    for i in range(len(train_frame['Text'])):
        try:
            train_frame['Text'][i] = jieba.lcut(train_frame['Text'][i])
        except:
            print('error occur')
        word_list = []
        for item in train_frame['Text'][i]:
            if item not in stop_list and len(item) <= 12:
                word_list.append(item)
        train_frame['Text'][i] = ' '.join(word_list)
    for i in range(len(test_frame['Text'])):
        try:
            test_frame['Text'][i] = jieba.lcut(test_frame['Text'][i])
        except:
            print('test error occur')
        word_list = []
        for item in test_frame['Text'][i]:
            if item not in stop_list and len(item) <= 12:
                word_list.append(item)
        test_frame['Text'][i] = ' '.join(word_list)
    return train_frame, test_frame


def trans_to_sentence_mode_from_file(path, stop_list):
    """
    将从网络手动搜集的语料文本文件中的内容进行分词，去停用词，并用空格连接成字符串
    :param path:文本文件路径
    :param stop_list:停用词
    :return:空格连接的字符串
    """
    print('feature_engineer.trans_to_sentence_mode_from_file')
    f = open(path, 'r', encoding='utf-8', errors='ignore')
    txt = f.read()
    txt = re.sub('宋体|[^\u4E00-\u9FD5]+', '', txt)
    jieba.load_userdict(USER_DIC_PATH)
    txt_list = jieba.lcut(txt)
    tmp_list = []
    for item in txt_list:
        if item not in stop_list:
            tmp_list.append(item)
    txt = ' '.join(tmp_list)
    return txt


def save_trans_to_setence_mode_from_file_for_rnn_data_word2vec(file_path_list, save_path, stop_list):
    """
    将RNN模型对应数据集里的语料转换为用空格连接的字符串形式，每一行代表简历中一个模块
    :param file_path_list:原语料所在路径列表（多个语料的路径集合），每一个语料文件形式为 每一个模块文本间用******************隔开
    :param stop_list:停用词列表
    :param save_path:保存路径
    :return:None
    """
    jieba.load_userdict(USER_DIC_PATH)
    with open(save_path, 'w', encoding='utf-8') as wf:
        for file_path in file_path_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                txt = f.read()
                module_list = txt.split('********************************************************************')
                for module in module_list:
                    word_list = jieba.cut(re.sub(DEL_SPECIAL_PAT, '', module))
                    word_list = [word for word in word_list if word not in stop_list]
                    word_str = ' '.join(word_list)
                    wf.write(word_str+'\n')


def change_to_word2vec(train_frame, test_frame, feature_size, word2vec_model, type='full', max_len=MAX_LEN):
    """
    type=full, full connect
    type = cnn, text cnn
    """
    print('feature_engineer:change_to_word2vec')
    stop_list = stopwordslist()  # 创建停用词表
    train_frame, test_frame = trans_to_sentence_mode(train_frame, test_frame, stop_list)
    train_arr = trans_to_wordvec_by_word2vec(train_frame['Text'], feature_size, type=type, max_len=max_len, word2vec_model=word2vec_model)
    test_arr = trans_to_wordvec_by_word2vec(test_frame['Text'], feature_size, type=type, max_len=max_len, word2vec_model=word2vec_model)
    if type == 'full':
        for i in range(train_arr.shape[0]):
            train_frame['Text'][i] = list(train_arr[i])
        for i in range(test_arr.shape[0]):
            test_frame['Text'][i] = list(test_arr[i])
    else:  # train_arr :(sample_num, word_num, feature_size)
        for i in range(train_arr.shape[0]):
            train_frame['Text'][i] = train_arr[i]
        for i in range(test_arr.shape[0]):
            test_frame['Text'][i] = test_arr[i]
    return train_frame, test_frame


def trans_to_wordvec_by_word2vec(text_list, feature_size, word2vec_model=None, type='full', max_len=MAX_LEN, time_step=24, begin=0):
    """
    :param:function：将未分词的文本list转为词向量list
    :param:text_list 是未分词的文本列表
    :param:feature_size: 词向量维数
    :param:type: full表示构建全连接神经网络需要的输入，此时返回的是多个样本的输入
    type: cnn表示构建text cnn 需要的输入，此时返回的是多个样本的输入
    type: rnn表示构建rnn需要的输入，此时text_list表示的是N份简历中所有模块文本内容组成的列表，形如：[ [text1, text2,..], [text1, text2,..], ...]
    ，返回的是N个样本的所有模块组成的向量序列 [ [vector1, vector2,..], [vector1, vector2,..], ...],3维
    type:attention:[sample_size, time_step, word_num, feature_size]
    :param:max_len：句子最大长度
    :param:time_step:rnn所需最大时间步长，超过该步长的截断，不足的补0，默认为24
    :param:begin:需要截取的起始位置
    """
    print('feature_engineer:trans_to_wordvec_by_word2vec')
    import numpy as np
    np.set_printoptions(suppress=True)  # 取消科学计数法输出
    np.seterr(divide='ignore', invalid='ignore')
    jieba.load_userdict(USER_DIC_PATH)
    stop_list = stopwordslist()
    if isinstance(word2vec_model, str):
        model = gensim.models.Word2Vec.load(word2vec_model)
    else:
        model = word2vec_model
    if isinstance(text_list, list):  # 是list
        text_list_len = len(text_list)
    else:  # 是series
        text_list_len = text_list.shape[0]
    if type == 'full' or type == 'cnn':
        text_list = pd.Series(text_list).apply(lambda x: re.sub(DEL_SPECIAL_PAT, '', x))
        # print(text_list)
        for i in range(text_list_len):
            # print(text_list[i])
            text_list[i] = jieba.lcut(text_list[i])
            word_list = []
            for item in text_list[i]:
                if item not in stop_list:
                    word_list.append(item)
            text_list[i] = word_list
        if type == 'full':
            print('full')
            vec_arr = np.zeros((text_list_len, feature_size))
            # print('feature_engineer:trans_to_wordvec_by_word2vec', vec_arr.shape)
            for i, word_list in enumerate(text_list):
                length = len(word_list)
                for elem in word_list:
                    try:
                        vec_arr[i, :] += model[elem]
                    except KeyError:
                        length -= 1
                        continue
                if length != 0:
                    vec_arr[i, :] /= length
        else:
            print('cnn')
            # 返回 textCNN所需数据集
            vec_arr = np.zeros((text_list_len, max_len, feature_size))
            # print('feature_engineer:trans_to_wordvec_by_word2vec', vec_arr.shape)
            for i, word_list in enumerate(text_list):
                for j, elem in enumerate(word_list):
                    if j < max_len:
                        try:
                            vec_arr[i, j] = model[elem]
                        except KeyError:
                            continue
                    else:
                        print('exceed')

    elif type == 'rnn':
        print('rnn')
        # 返回rnn模型所需数据集
        vec_arr = np.zeros((text_list_len, time_step, feature_size))
        for i, resume_modules in enumerate(text_list):  # 遍历每一个简历样本
            resume_modules = pd.Series(resume_modules).apply(lambda x: re.sub(DEL_SPECIAL_PAT, '', x))
            # print(text_list)
            for j in range(resume_modules.shape[0] - begin):
                # print(text_list[i])
                resume_modules[begin + j] = jieba.lcut(resume_modules[begin + j])
                word_list = []
                for item in resume_modules[j + begin]:
                    if item not in stop_list:
                        word_list.append(item)
                resume_modules[begin + j] = word_list
            for j in range(begin, resume_modules.shape[0]):
                if j == begin + time_step:
                    break
                # 生成第i份简历中第j个模块的词向量
                word_list = resume_modules[j]
                length = len(word_list)
                for elem in word_list:
                    try:
                        vec_arr[i, j-begin, :] += model[elem]
                    except KeyError:
                        print('keyError:', elem)
                        length -= 1
                        continue
                if length != 0:
                    vec_arr[i, j-begin, :] /= length

    elif type == 'attention':
        print('attention rnn')
        # print('attention,text_list:', text_list)
        # 返回加入attention的rnn模型所需数据集
        vec_arr = np.zeros((text_list_len, time_step, MAX_LEN, feature_size))
        for i, resume_modules in enumerate(text_list):  # 遍历每一个简历样本
            resume_modules = pd.Series(resume_modules).apply(lambda x: re.sub(DEL_SPECIAL_PAT, '', x))
            # print(text_list)
            for j in range(resume_modules.shape[0] - begin):
                # print(text_list[i])
                resume_modules[begin + j] = jieba.lcut(resume_modules[begin + j])
                word_list = []
                for item in resume_modules[j + begin]:
                    if item not in stop_list:
                        word_list.append(item)
                resume_modules[begin + j] = word_list
            for j in range(begin, resume_modules.shape[0]):  # 遍历同一份简历的每一个模块(按时间步遍历)
                if j == begin + time_step:
                    break
                # 生成第i份简历中第j个模块的词向量
                word_list = resume_modules[j]  # 一个模块下的所有单词
                word_array = np.zeros((MAX_LEN, feature_size))  # 一个模块下的所有单词组成的向量矩阵
                for k, elem in enumerate(word_list):
                    if k >= MAX_LEN:  # 截断
                        break
                    try:
                        word_array[k] = model[elem]  # 不足补零
                    except KeyError:
                        continue
                # print(word_array.shape)
                # print(vec_arr[i, j - begin, :, :].shape)
                vec_arr[i, j - begin, :, :] = word_array

    return vec_arr


def process_rnn_label_list(label_list, time_step=24, one_hot_num=10, begin=0):
    """
    统一标签序列时间步长，超出的丢弃，不足的补0
    :param label_list:[  [[0,0,0,1,0],[0,0,0,1,0],...],  [[0,0,0,1,0],[0,0,0,1,0],...]  ]
    :param time_step:最大时间步长，默认为24
    :param one_hot_num:独热编码长度，默认为10
    :param begin:截取开始位置，默认从头开始
    :return:统一维度的标签列表
    """
    for i, one_resume_labels in enumerate(label_list):
        for j in range(len(label_list[i])):
            label_list[i][j].append(0)  # 加上非空白标志
        if len(one_resume_labels) >= begin + time_step:
            label_list[i] = one_resume_labels[begin:begin + time_step]
        else:
            for m in range(begin + time_step - len(one_resume_labels)):
                ones = [0 for k in range(one_hot_num+1)]
                ones[-1] = 1  # 最后一位为空白标志位
                label_list[i].append(ones)
            label_list[i] = label_list[i][begin:begin + time_step]


if __name__ == '__main__':
    def test_process_rnn_label_list():
        label_list = [
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0]],

            [[0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1]]
        ]
        process_rnn_label_list(label_list, time_step=4, one_hot_num=5, begin=8)
        for label in label_list:
            print(label)







