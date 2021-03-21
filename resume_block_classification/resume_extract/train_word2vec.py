# encoding:utf-8
import gensim
from special_string import *
from load_resume_data import load_data, load_word2vec_wiki_corpus
from feature_engineer import trans_to_sentence_mode, stopwordslist, trans_to_sentence_mode_from_file
import os
import numpy as np
import jieba
from static_num import *
import re
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_corpus():
    """
    :return: [str,str,str,...], str:按空格连接的字符串
    """
    train_frame, test_frame = load_data()  # 得到多分类原始语料
    stop_list = stopwordslist()  # 加载停用词
    train_frame, test_frame = trans_to_sentence_mode(train_frame, test_frame, stop_list)  # 得到符合训练词嵌入模型格式的语料
    train_list, test_list = train_frame['Text'], test_frame['Text']
    train_list, test_list = list(train_list), list(test_list)
    train_list.extend(test_list)
    return train_list


def adjust_it_corpus(path, corpus_dir_name, corpus_name):
    """
    原来的语料是多个文件，将其整合为一个文件
    :param path: 语料所在文件夹路径
    :param corpus_dir_name: 整理后存入的文件夹路径
    :return: [str,str,str,...], str:按空格连接的字符串
    """
    res = []  # 最终返回的list
    file_names = os.listdir(path)
    stop_list = stopwordslist()
    for name in file_names:
        file_path1 = path + '/' + name
        f = open(file_path1, 'r', encoding='utf-8', errors="ignore")
        one_list = f.readlines()
        temp = ''
        cnt = 0
        for item in one_list:
            if cnt < 10:
                temp += item
                cnt += 1
            else:
                temp = re.sub('span|博文|p7|p8|p9|style|div|[/<>_]+', '', temp)
                temp = re.sub('[a-zA-Z]{11,}', '', temp)
                temp = re.sub('[\d]{2,}', '', temp)
                if len(temp) > 60:
                    temp = re.sub('\n', ' ', temp)
                    temp_list = temp.split(' ')
                    temp_list = [item for item in temp_list if item not in stop_list]
                    temp = ' '.join(temp_list)
                    res.append(temp)
                else:
                    print('<30:', temp)
                temp = item
                cnt = 1
        if temp != '':
            if cnt > 5:
                temp = re.sub('span|博文|p7|p8|p9|style|div|[/<>_]+', '', temp)
                temp = re.sub('[a-zA-Z]{11,}', '', temp)
                temp = re.sub('[\d]{2,}', '', temp)
                if len(temp) > 30:
                    temp = re.sub('\n', ' ', temp)
                    temp_list = temp.split(' ')
                    temp_list = [item for item in temp_list if item not in stop_list]
                    temp = ' '.join(temp_list)
                    res.append(temp)
                else:
                    print('<30:', temp)
            else:
                res[-1] += temp
    f.close()
    if not os.path.exists(corpus_dir_name):
        os.mkdir(corpus_dir_name)
    file_name = os.path.join(corpus_dir_name, corpus_name)
    f = open(file_name, 'w', encoding='utf-8')
    for item in res:
        if len(item) >= 50:
            f.write(item)
            f.write('\n')
    f.close()
    return res


def save_corpus(train_list, sentence_path):
    f = open(sentence_path, 'w', encoding='utf-8')
    for item in train_list:
        f.write(item)
        f.write('\n')
    f.close()


class Word2vecFunction(object):
    def __init__(self, model_path):
        # 对.sava保存的模型的加载：
        self.model = gensim.models.Word2Vec.load(model_path)
        # 对..wv.save_word2vec_format保存的模型的加载：
        # model = gensim.model.wv.load_word2vec_format('模型文件名')

    def cos_distance_by_word2vec(self, sentence1, sentence2):
        """
        返回两个句子的word2vec余弦相似度
        :param sentence1:str
        :param sentence2:str
        :return:
        """
        if len(sentence1) == 0 or len(sentence2) == 0:
            return 0
        list1, list2 = jieba.lcut(sentence1), jieba.lcut(sentence2)
        arr1, arr2 = np.zeros((len(list1), WORD2VEC_FEATURE_NUM)), np.zeros((len(list2), WORD2VEC_FEATURE_NUM))
        for i in range(len(list1)):
            for j in range(WORD2VEC_FEATURE_NUM):
                arr1[i, j] = self.model[list1[i]][j]

        for i in range(len(list2)):
            for j in range(WORD2VEC_FEATURE_NUM):
                arr2[i, j] = self.model[list2[i]][j]
        vector1, vector2 = np.array([np.mean(arr1[:, j]) for j in range(WORD2VEC_FEATURE_NUM)]), np.array([np.mean(arr2[:, j]) for j in range(WORD2VEC_FEATURE_NUM)])
        # print(vector1.shape, vector2.shape)
        dot_res = np.dot(vector1, vector2)
        length = np.sqrt(np.sum(vector1**2)) * np.sqrt(np.sum(vector2**2))
        cos = dot_res / length
        # print(cos)
        return cos


if __name__ == '__main__':
    from gensim.corpora import WikiCorpus
    import jieba
    from langconv import *


    def test_save_wiki_txt():
        space = ' '
        i = 0
        l = []
        zhwiki_name = zhwiki_corpus_path
        jieba.load_userdict(USER_DIC_PATH)
        f = open(sentence_zhwiki_corpus_txt_path, 'w', encoding='utf-8')
        wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            for temp_sentence in text:
                temp_sentence = Converter('zh-hans').convert(temp_sentence)
                temp_sentence = re.sub('[^\u4E00-\u9FD5\d\w\]\[://@.]+', '', temp_sentence)
                seg_list = list(jieba.cut(temp_sentence))
                for temp_term in seg_list:
                    l.append(temp_term)
            f.write(space.join(l) + '\n')
            l = []
            i = i + 1
            if i % 200 == 0:
                print('Saved ' + str(i) + ' articles')
        f.close()


    def test_train_word2vec():
        """
            1.加载来自于多分类训练集的语料
            2.加载来自于中文WIKI的训练语料
            corpus = create_corpus()  # 来自多分类训练集的语料
            save_corpus(corpus, sentence_path=sentence_train_data_path)
            """

        # sentence = gensim.models.word2vec.Text8Corpus(ROOT_PATH + '\\job_corpus.txt')
        # model = gensim.models.word2vec.Word2Vec(sentence, size=100, window=5, min_count=1, iter=5, negative=5)
        # model.save(word2vec_model_path_2021_2_5)
        # sentence = []
        model = gensim.models.Word2Vec.load(word2vec_model_path_zhwiki_rnn_update_20_923)  # 加载模型，加入IT语料训练
        sentences2 = gensim.models.word2vec.Text8Corpus(ROOT_PATH + '\\job_corpus.txt')
        # # # print(sentences2)
        model.build_vocab(sentences=sentences2, update=True)
        model.train(sentences2, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(word2vec_model_path_2021_2_5)
        model = gensim.models.Word2Vec.load(word2vec_model_path_2021_2_5)  # 加载模型，加入IT语料训练
        print(model.most_similar('销售'))
        print(model.most_similar('钻工'))
        print(model.most_similar('教师'))
        print(model.most_similar('运营'))
        print(model.most_similar('美容'))
        print(model.most_similar('汽车'))
        print(model.most_similar('工程师'))
        print(model.most_similar('算法'))
        print(model.most_similar('后端'))
        # model = gensim.models.Word2Vec.load(word2vec_model_path_it_new)
        # print("begin it corpus")
        # generator = load_word2vec_wiki_corpus(add_count=10000)  # 加载语料的迭代器，一次加载5000条语料
        # # 由于训练语料过大，采用分批次增量训练
        # i = 0
        # for i, sentences in enumerate(generator):
        #     model.build_vocab(sentences=sentences, update=True)
        #     print('number', i, 'build over')
        #     model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        #     print('number', i, 'end train')
        #     print(model.most_similar('数据挖掘'))
        #     print(model.most_similar('人工智能'))
        #     print(model.most_similar('杭州电子科技大学'))
        #     print(model.most_similar('大数据'))
        #     print(model.most_similar('数据结构'))
        #     print(model.most_similar('计算机'))
        #     print(model.most_similar('字节跳动'))
        #     print(model.most_similar('算法'))
        #     print(model.most_similar('积极'))
        # model.save(ROOT_PATH + '\\word2vec_update_814_chwiki')
        # model = gensim.models.Word2Vec.load(word2vec_model_path_zhwiki_update_814)
        # # sentences = gensim.models.word2vec.Text8Corpus(sentence_rnn_data_word2vec_corpus_path)
        # # model.build_vocab(sentences=sentences, update=True)
        # # print('vocab build over')
        # # model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        # # print('end train')
        # print(model.most_similar('数据挖掘'))
        # print(model.most_similar('人工智能'))
        # print(model.most_similar('杭州电子科技大学'))
        # print(model.most_similar('大数据'))
        # print(model.most_similar('数据结构'))
        # print(model.most_similar('计算机'))
        # print(model.most_similar('字节跳动'))
        # print(model.most_similar('算法'))
        # print(model.most_similar('积极'))
        # model.save(word2vec_model_path_zhwiki_rnn_update_20_923)
        # model = gensim.models.Word2Vec.load(word2vec_model_path_zhwiki_rnn_update_20_923)
        # print(type(model[['积极', '数据结构']]))


    def test_cos_distance_by_word2vec():
        obj = Word2vecFunction(word2vec_model_path_zhwiki_rnn_update_20_923)
        # cos = obj.cos_distance_by_word2vec('我的学习成绩非常好', '我的学习成绩真的很棒')
        # print(cos)
        # print(obj.model.most_similar('南京航空航天大学'))
        word = '测试代码'
        print(jieba.lcut(word))
        print(obj.model[word])

    # --------------------------------------测试------------------------------------------ #
    test_train_word2vec()
    # test_save_wiki_txt()
    # test_cos_distance_by_word2vec()







