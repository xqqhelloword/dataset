import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense,MaxPooling1D, GRU, Bidirectional, Input
from tensorflow.keras import Model
from resume_extract.feature_engineer import trans_to_wordvec_by_word2vec
from load_resume_data import load_data_for_single_muti_classification
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from static_num import WORD2VEC_FEATURE_NUM, MAX_LEN
from special_string import *

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(AttentionLayer, self).__init__()
        self.n_outputs = n_outputs
        # self.weight_k, self.weight_q, self.weight_v, self.w = None, None, None, None
        self.softmax = tf.nn.softmax

    def build(self, input_shape):
        self.weight_k = self.add_weight('k',
                                          shape=[int(input_shape[-1]),
                                               self.n_outputs])
        self.weight_q = self.add_weight('q',
                                          shape=[int(input_shape[-1]),
                                                 self.n_outputs])
        self.weight_v = self.add_weight('v',
                                          shape=[int(input_shape[-1]),
                                                 self.n_outputs])
        # 用于得到查询值
        self.w = self.add_weight('w', shape=[int(input_shape[-2]), 1])  # (word_num, 1)

    def call(self, x):
        """
        input.shape:(sample_size, word_num, feature_size)
        :param x:
        :return:
        """
        shape0 = tf.shape(x)[0]
        if shape0 is None:
            print(True)
        # print('????')
        Q, K, V = [], [], []  # 所有样本的查询、键、值向量
        # 按时间步展开计算所有样本在每一步上的键值向量
        for item in tf.unstack(x, axis=1):
            # item.shape:(sample_size, feature_size)
            # print(item)
            # print(self.weight_k)
            # print('item.shape:', item.shape, 'self.weight_k.shape:', self.weight_k.shape)
            K.append(tf.matmul(item, self.weight_k))  # tf.matmul(item, self.weight_k)：(sample_size, n_outputs)
            V.append(tf.matmul(item, self.weight_v))  # tf.matmul(item, self.weight_v)：(sample_size, n_outputs)
        # 按样本展开计算每一个样本的查询向量
        # count = 0
        # print('x.shape:', x.shape)
        for one_sample in tf.unstack(x, axis=0):
            # print(count)
            # count += 1
            # one_sample.shape:(word_num, feature_size)
            t = tf.matmul(tf.transpose(one_sample), self.w)  # (feature_size, 1) 列向量
            # print('t.shape:', t.shape)
            # print(self.weight_q.shape)
            q = tf.matmul(tf.transpose(t), self.weight_q)  # (1, n_outputs) 行向量
            Q.append(q)
        Q = tf.concat(Q, axis=0)
        # print('Q.shape:', Q.shape)
        # Q:(sample_size, n_outputs). 每一个样本只有一个查询向量，而每一个样本的键值向量则有word_num个
        # 注意力计算公式：softmax(q*K/(sqrt(d_k)) * V
        weights = []  # 存储所有时间步上的所有样本的注意力weight
        for one_step in K:  # 计算每一个时间步（单词）对分类的注意力权重
            # one_step.shape:(sample_size, n_outputs), tensor
            weight = tf.divide(tf.reduce_sum(tf.multiply(Q, one_step), axis=1), tf.sqrt(float(self.n_outputs)))
            # weight.shape:(sample_size, ) 所有样本在一个时间步上的权重
            # print('weight.shape:', weight.shape)
            weights.append(weight)

        weights = tf.stack(weights, axis=-1)  # (sample_size, word_num)  # 一行代表一个样本在所有步长上的权重
        # print('weights:', weights)
        weights = self.softmax(weights, axis=1)  # (sample_size, word_num)
        # print('after softmax:', weights)
        V = tf.stack(V, axis=1)  # (sample_size, word_num, n_outputs)
        # print('V.shape:',V.shape)
        weights = tf.stack([weights], axis=-1)
        # print('before broadcast, weights.shape:', weights.shape)
        weights = tf.broadcast_to(weights, [V.shape[0], V.shape[1], V.shape[2]])
        # print('after broadcast, weights.shape:', weights.shape)
        final_output = tf.reduce_sum(tf.multiply(weights, V), axis=1)  # (sample_size, n_outputs)
        # print('final_output.shape:', final_output.shape)  # n个样本的最终表征向量

        return final_output


def load_structure_data_by_word2vec_new(feature_num, word2vec_path, type='cnn', max_len=MAX_LEN, data=1,
                                        train_size=50, test_size=400, val_size=100):
    train_x, train_y, test_x, test_y, val_x, val_y = load_data_for_single_muti_classification(data_set=data,
                                                            train_num=train_size, test_num=test_size, val_num=val_size)
    train_x = np.array(trans_to_wordvec_by_word2vec(train_x, feature_size=feature_num,
                                                    word2vec_model=word2vec_path, type=type, max_len=max_len))
    test_x = np.array(trans_to_wordvec_by_word2vec(test_x, feature_size=feature_num,
                                                   word2vec_model=word2vec_path, type=type, max_len=max_len))
    val_x = np.array(trans_to_wordvec_by_word2vec(val_x, feature_size=feature_num,
                                                    word2vec_model=word2vec_path, type=type, max_len=max_len))
    train_y, test_y, val_y = np.array(train_y), np.array(test_y), np.array(val_y)
    return train_x, train_y, test_x, test_y, val_x, val_y


def save_train_wbrnn(word2vec_feature_size, word2vec_path, max_len=MAX_LEN, output_size=10, data=1, train_num=100, test_num=400):
    """
    word2vec_feature_size:词嵌入向量的维数
    max_len:输入（句子）的最大长度，大于该长度则截断，小于该长度补0
    output_size：神经网络最终输出
    """
    # 加载符合cnn输入的数据集
    # train_frame, test_frame = load_structure_data_by_word2vec(word2vec_feature_size, type='cnn',
    # max_len=max_len, word2vec_path=word2vec_path)
    train_x, train_y, test_x, test_y, val_x, val_y = load_structure_data_by_word2vec_new(word2vec_feature_size,
                                                                           type='cnn',
                                                                           max_len=max_len,
                                                                           word2vec_path=word2vec_path, data=data,
                                                                           train_size=train_num, test_size=test_num, val_size=100)
    print('train_x.shape', train_x.shape)
    input_tensor = Input(shape=(max_len, word2vec_feature_size), name='input')
    # print('after input_tensor')
    x = Bidirectional(GRU(units=64, return_sequences=True, activation='relu'))(input_tensor)
    x = Bidirectional(GRU(units=64, return_sequences=True, activation='relu'))(x)

    # x = GRU(units=100, return_sequences=True, activation='relu')(x)
    # print('befor maxpooling, x.shape:', x.shape)
    x = MaxPooling1D(MAX_LEN)(x)
    # x的shape应为(128, )
    x = Flatten()(x)
    # drop = Dropout(0.05)(x)
    # full = BatchNormalization()(drop)
    # dense = Dropout(0.1)(full1)
    x = Dense(128, activation='relu')(x)
    output = Dense(output_size, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, tf.convert_to_tensor(train_y), batch_size=64, epochs=7)
    # scores = model.evaluate(test_x, tf.convert_to_tensor(test_y), verbose=0)
    pres = model.predict(test_x)
    # print('pres.shape:', pres.shape)
    pres = np.argmax(pres, axis=1)
    # print(pres)
    # print(test_y)
    # print('pres.shape:', pres.shape)
    P = precision_score(test_y, pres, average='weighted')
    R = recall_score(test_y, pres, average='weighted')
    f1 = f1_score(test_y, pres, average='weighted')
    # print('test data score:', scores)
    print('test data precision , recall, f1-score:', P, R, f1)
    # model.save(model_save_path)


if __name__ == '__main__':
    # print(train_frame)
    # print(test_frame)
    save_train_wbrnn(word2vec_feature_size=WORD2VEC_FEATURE_NUM, max_len=MAX_LEN,
                     word2vec_path=word2vec_model_path_2021_2_5,
                     data=3, train_num=700, test_num=400)
    # 100, [3,4,5], 0.1, 128, 0.2
