# -*- coding: utf-8 -*-
import os
import re
import random
from feature_engineer import trans_to_wordvec_by_word2vec

# ROOT_PATH =os.path.abspath(os.path.dirname(__file__))  # current directory path
from special_string import *


def load_data_for_rnn(data_dir=RNN_DATA_DATA_PATH1, label_dir=RNN_DATA_LABEL_PATH1):
    """

    :param data_dir: the path of corpus
    :param label_dir: directory path of resume label
    :return:
    """
    resume_list = []
    label_list = []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    for resume_file_name in resume_file_names:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        resume_list.append(one_resume)
    for label_file_name in label_file_names:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        label_list.append(one_resume_labels)
    return resume_list, label_list


def load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names, label_file_names, noise_percent=0, noise_type='shuffle'):
    x, y = [], []
    indexes = {}  # used to decide whether producing abnormal resume sample
    for i, resume_file_name in enumerate(resume_file_names):
        is_abnormal = random.randint(0, 9)
        if is_abnormal >= 10 - noise_percent:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
                if len(one_resume) > 2:
                    new_one_resume = []
                    if noise_type == 'skip':  # random skip
                        num = random.randint(0, len(one_resume) - 1)
                        skips = random.sample([k for k in range(len(one_resume))], num)
                        skips = set(skips)
                        for j in range(len(one_resume)):
                            if j in skips:
                                # print('continue')
                                continue
                            new_one_resume.append(one_resume[j])
                        indexes[i] = skips
                    elif noise_type == 'shuffle':  # random shuffle
                        shuffles = random.sample([k for k in range(0, len(one_resume))], len(one_resume))  # 从1开始打乱
                        for index in shuffles:
                            new_one_resume.append(one_resume[index])
                        indexes[i] = shuffles
                    else:  # random swap
                        num = random.randint(2, 4) if len(one_resume) >= 4 else random.randint(2, len(one_resume))
                        swaps = random.sample([k for k in range(len(one_resume))], num)  # 随机选取的原始需要打乱的索引子集
                        print(swaps)
                        swaps_dic = {}
                        for j in range(len(swaps)):
                            swaps_dic[swaps[j]] = j

                        shuffles = random.sample(swaps, len(swaps))  # 打乱后的索引
                        for j in range(len(one_resume)):
                            if j in swaps:
                                index_swap = swaps_dic[j]
                                new_one_resume.append(one_resume[shuffles[index_swap]])  # 如果当前索引是需要打乱的，则用打乱后的值填充
                            else:
                                new_one_resume.append(one_resume[j])
                        indexes[i] = (swaps, swaps_dic, shuffles)
                    one_resume = new_one_resume
        else:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
        x.append(one_resume)
    for i, label_file_name in enumerate(label_file_names):
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(k) for k in one_label]
                one_resume_labels.append(one_label)
            if i in indexes.keys():
                skips = indexes[i]
                new_labels = []
                if noise_type == 'shuffle':  # 表明是顺序需要打乱
                    for index in skips:
                        new_labels.append(one_resume_labels[index])
                elif noise_type == 'skip':
                    for j in range(len(one_resume_labels)):
                        if j in skips:
                            continue
                        new_labels.append(one_resume_labels[j])
                else:
                    for j in range(len(one_resume_labels)):
                        if j in indexes[i][0]:
                            index_swap = indexes[i][1][j]
                            new_labels.append(one_resume_labels[indexes[i][2][index_swap]])  # 如果当前索引是需要打乱的，则用打乱后的值填充
                        else:
                            new_labels.append(one_resume_labels[j])
                one_resume_labels = new_labels
        y.append(one_resume_labels)

    return x, y


def load_data_for_rnn_new_add_noise(data_set=1, train_num=25, test_num=400, val_num=None, load_train=True,
                                    train_noise_percent=0, noise_percent=0, noise_type='shuffle'):
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(resume_file_names)
    if load_train:
        train_x, train_y = load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names[:train_num],
                                                                label_file_names[:train_num],
                                                                noise_percent=train_noise_percent, noise_type=noise_type)
    test_x, test_y = load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names[length - test_num:length],
                                                                label_file_names[length - test_num:length],
                                                                noise_percent=noise_percent, noise_type=noise_type)
    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
            val_x.append(one_resume)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                one_resume_labels = []  # 一份简历中N个标题对应的label
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_label = one_title_label.split(' ')
                    one_label = [int(i) for i in one_label]
                    one_resume_labels.append(one_label)
            val_y.append(one_resume_labels)
    if val_num is None:
        if load_train:
            return train_x, train_y, test_x, test_y
        else:
            return test_x, test_y
    else:
        if load_train:
            return train_x, train_y, test_x, test_y, val_x, val_y
        else:
            return test_x, test_y, val_x, val_y


def load_data_for_rnn_new(data_set=1, train_num=25, test_num=400, val_num=None):
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(resume_file_names)
    for resume_file_name in resume_file_names[:train_num]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        train_x.append(one_resume)
    for label_file_name in label_file_names[:train_num]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        train_y.append(one_resume_labels)

    for resume_file_name in resume_file_names[length - test_num:length]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        test_x.append(one_resume)
    for label_file_name in label_file_names[length - test_num:length]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        test_y.append(one_resume_labels)

    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
            val_x.append(one_resume)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                one_resume_labels = []  # 一份简历中N个标题对应的label
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_label = one_title_label.split(' ')
                    one_label = [int(i) for i in one_label]
                    one_resume_labels.append(one_label)
            val_y.append(one_resume_labels)
    if val_num is None:
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y, test_x, test_y, val_x, val_y


def load_data_for_single_muti_classification(data_set=1, train_num=50, test_num=150, val_num=None):
    """
    load dataset for models that only classify single resume block.
    :param data_dir: directory path of corpus
    :param label_dir: directory path of resume label
    :param train_num:the train set size
    :param test_num:the test size
    :return:
    """
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(label_file_names)
    for resume_file_name in resume_file_names[:train_num]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
            for one_module in one_resume:
                train_x.append(one_module)
    for label_file_name in label_file_names[:train_num]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_title_label = re.sub('\n', '', one_title_label)
                one_label = one_title_label.split(' ')
                no_one_flag = True
                for i, num in enumerate(one_label):
                    if num == '1':
                        train_y.append(i)
                        no_one_flag = False
                        break
                if no_one_flag:
                    print(one_label)
    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
                for one_module in one_resume:
                    val_x.append(one_module)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_title_label = re.sub('\n', '', one_title_label)
                    one_label = one_title_label.split(' ')
                    no_one_flag = True
                    for i, num in enumerate(one_label):
                        if num == '1':
                            val_y.append(i)
                            no_one_flag = False
                            break
                    if no_one_flag:
                        print(one_label)
    for resume_file_name in resume_file_names[length - test_num:length]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
            for one_module in one_resume:
                test_x.append(one_module)
    for label_file_name in label_file_names[length - test_num:length]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_title_label = re.sub(r'\n', '', one_title_label)
                one_label = one_title_label.split(' ')
                for i, num in enumerate(one_label):
                    if num == '1':
                        test_y.append(i)
                        break
    if val_num is None:
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y, test_x, test_y, val_x, val_y


def minEditDist(sm, sn):
    m, n = len(sm) + 1, len(sn) + 1

    # create a matrix (m*n)

    matrix = [[0] * n for i in range(m)]

    matrix[0][0] = 0
    for i in range(1, m):
        matrix[i][0] = matrix[i - 1][0] + 1

    for j in range(1, n):
        matrix[0][j] = matrix[0][j - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if sm[i - 1] == sn[j - 1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

    return matrix[m - 1][n - 1]


if __name__ == '__main__':

    def test_load_data_for_rnn():
        resume_list, label_list = load_data_for_rnn()
        resume_list = trans_to_wordvec_by_word2vec(resume_list, feature_size=100, word2vec_model=word2vec_model_path_2021_2_5, type='rnn')
        print(resume_list)


    def test_load_data_for_single_muti_classification():
        train_x, train_y, test_x, test_y = load_data_for_single_muti_classification(data_set=1, train_num=50, test_num=150)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                                                   word2vec_model=word2vec_model_path_2021_2_5, type='full')
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                                               word2vec_model=word2vec_model_path_2021_2_5, type='full')
        # for txt in resume_list:
        #     print(txt)
        print(len(train_x), len(test_x))


    # ----------------------------------------test-------------------------------------- #
    train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=25, test_num=400, val_num=None)
    for one_resume in test_y:
        print('------------------------------')
        for module in one_resume:
            print(module)
