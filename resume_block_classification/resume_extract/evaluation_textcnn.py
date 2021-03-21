import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dropout,BatchNormalization, Dense,MaxPooling1D,Conv1D,Input
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from resume_extract.feature_engineer import trans_to_wordvec_by_word2vec
from load_resume_data import load_data_for_single_muti_classification
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from special_string import *
from static_num import WORD2VEC_FEATURE_NUM,MAX_LEN

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


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


def cross_validation(train_x, train_y, model):
    train_y = np.array(train_y)
    kf = KFold(n_splits=5)
    score_list = []
    for train_index,validate_index in kf.split(train_x):
        x_train, y_train = train_x[np.ix_(train_index)], train_y[np.ix_(train_index)]
        x_validate, y_validate = train_x[np.ix_(validate_index)], train_y[np.ix_(validate_index)]
        model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=0)
        scores = model.evaluate(x_validate, y_validate, verbose=0)
        score_list.append(scores[1])
    return np.mean(score_list)


def train_text_cnn(word2vec_feature_size, word2vec_path, max_len=MAX_LEN, output_size=10, data=1, train_num=50, test_num=400):
    """
    word2vec_feature_size:词嵌入向量的维数
    max_len:输入（句子）的最大长度，大于该长度则截断，小于该长度补0
    conv:卷积核的高度，相当于n-gram的大小,默认有4,5,6,7,8 这几种大小的卷积核
    output_size：神经网络最终输出
    """
    # 加载符合cnn输入的数据集
    # train_frame, test_frame = load_structure_data_by_word2vec(word2vec_feature_size, type='cnn',
    # max_len=max_len, word2vec_path=word2vec_path)
    train_x, train_y, test_x, test_y, val_x, val_y= load_structure_data_by_word2vec_new(word2vec_feature_size, type='cnn', max_len=max_len,
    word2vec_path=word2vec_path, data=data, train_size=train_num, test_size=test_num, val_size=100)
    filter_num = [50, 100, 150]
    filter_size = [[2, 3, 4],
                   [3, 4, 5],
                   [4, 5, 6]]
    drop1 = [0.1, 0.2, 0.3, 0.4]
    # train_x = np.zeros((train_frame['Text'].shape[0], max_len, word2vec_feature_size))
    # test_x = np.zeros((test_frame['Text'].shape[0], max_len, word2vec_feature_size))
    # for i, item in enumerate(train_frame['Text']):
    #     train_x[i] = item
    # for i, item in enumerate(test_frame['Text']):
    #     test_x[i] = item
    print('train_model:train_text_cnn:', train_x.shape)
    for filter_n in filter_num:
        for filter_s in filter_size:
            for drop1_p in drop1:
                model = Sequential()
                model.add(Input(shape=(max_len, word2vec_feature_size), name='input'))
                for item in filter_s:
                    model.add(Conv1D(filter_n, item, padding='same', activation='relu'))
                    model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 3, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 4, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 5, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                model.add(Flatten())
                model.add(Dropout(drop1_p))
                model.add(BatchNormalization())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(output_size, activation='softmax'))
                model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                # model.fit(train_x, train_frame['Label'], batch_size=128, epochs=15)
                # scores = model.evaluate(test_x, test_frame['Label'], verbose=0)
                print('entry cross_validation:')
                # score = cross_validation(train_x, train_frame['Label'], model)
                score = cross_validation(train_x, train_y, model)
                print(filter_n, filter_s, drop1_p, ':', score)


def save_train_text_cnn(word2vec_feature_size, word2vec_path, model_save_path, max_len=MAX_LEN, output_size=10, data=1, train_num=50, test_num=400):
    """
    word2vec_feature_size:词嵌入向量的维数
    max_len:输入（句子）的最大长度，大于该长度则截断，小于该长度补0
    conv:卷积核的高度，相当于n-gram的大小,默认有4,5,6,7,8 这几种大小的卷积核
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
    print('train_model:train_text_cnn:', train_x.shape)
    input_tensor = Input(shape=(max_len, word2vec_feature_size), name='input')
    print('after input_tensor')
    cnn1 = Conv1D(100, 3, padding='same', activation='relu')(input_tensor)
    print('after cnn1')
    cnn1 = MaxPooling1D(padding='same')(cnn1)
    cnn2 = Conv1D(100, 4, padding='same', activation='relu')(input_tensor)
    cnn2 = MaxPooling1D(padding='same')(cnn2)
    cnn3 = Conv1D(100, 5, padding='same', activation='relu')(input_tensor)
    cnn3 = MaxPooling1D(padding='same')(cnn3)
    print('after cnn3')
    cnn = tf.concat([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    # drop = Dropout(0.1)(flat)
    full = BatchNormalization()(flat)
    dense = Dense(128, activation='relu')(full)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    # dense = Dropout(0.1)(dense)
    # dense = Dropout(0.1)(full1)
    output = Dense(output_size, activation='softmax')(dense)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, tf.convert_to_tensor(train_y), batch_size=32, epochs=10)
    scores = model.evaluate(test_x, tf.convert_to_tensor(test_y), verbose=0)
    pres = model.predict(test_x)
    # print('pres.shape:', pres.shape)
    pres = np.argmax(pres, axis=1)
    print('pres.shape:', pres.shape)
    P = precision_score(test_y, pres, average='weighted')
    R = recall_score(test_y, pres, average='weighted')
    f1 = f1_score(test_y, pres, average='weighted')
    print('test data score:', scores)
    print('test data precision , recall, f1-score:', P, R, f1)
    # model.save(model_save_path)


if __name__ == '__main__':
    # print(train_frame)
    # print(test_frame)
    save_train_text_cnn(word2vec_feature_size=WORD2VEC_FEATURE_NUM, max_len=MAX_LEN,
                        word2vec_path=word2vec_model_path_2021_2_5,
                        model_save_path=muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec,
                        data=3, train_num=100, test_num=400)
    # 100, [3,4,5], 0.1, 128, 0.2
