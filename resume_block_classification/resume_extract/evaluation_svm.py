# coding=utf-8

"""
@File  : evaluation_svm.py
"""
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score
import joblib
from load_resume_data import load_data_for_single_muti_classification
from feature_engineer import trans_to_wordvec_by_word2vec
from special_string import *

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


def load_model(model_path):
    return joblib.load(model_path)


if __name__ == '__main__':
    def test_svm():
        tests1 = [100, 200, 300, 400, 500, 600, 700]
        tests = [5, 10,20,50, 60, 70, 100]
        for train_num in tests:
            train_x, train_y, test_x, test_y = load_data_for_single_muti_classification(data_set=3, train_num=train_num, test_num=400)
            print(len(train_x), len(train_y))
            train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                                                       word2vec_model=word2vec_model_path_2021_2_5, type='full')
            test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                                                   word2vec_model=word2vec_model_path_2021_2_5, type='full')
            # print(train_y)
            # train_x, train_y = np.array(train_x), np.array(train_y)
            # print(np.isnan(train_x).all())
            model = SVC()
            # model = GradientBoostingClassifier()
            model.fit(train_x, train_y)

            pre = model.predict(test_x)
            total = len(test_y)
            trues = 0
            total_p, total_p_t =  0, 0
            total_r, total_r_p = 0, 0
            for i in range(total):
                if pre[i] == test_y[i]:
                    trues += 1
            acc = trues / total
            # print('accuracy is:', acc)
            pre_c = [[], [], [], [], [], [], [], [], [], []]  # 每一个类别对应一个P
            recall_c = [[], [], [], [], [], [], [], [], [], []]  # 每一个类别对应一个R
            p_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            c_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(total):
                recall_c[test_y[i]].append(i)
                pre_c[pre[i]].append(i)

            weight_r = [len(recall_c[i]) / total for i in range(len(recall_c))]
            # print(weight_r)
            for i in range(len(pre_c)):
                total_p = len(pre_c[i])
                total_r = len(recall_c[i])
                total_p_t = 0
                # print('numer', str(i), 'pre and actual:', total_p, total_r)
                for ele in pre_c[i]:
                    if test_y[ele] == i:
                        total_p_t += 1
                if total_p != 0:
                    p_arr[i] = total_p_t / total_p * weight_r[i]
            precision = sum(p_arr)
            # print('precision is:', precision)
            # print(p_arr)
            for i in range(len(recall_c)):
                total_r = len(recall_c[i])
                total_r_p = 0
                for ele in recall_c[i]:
                    if pre[ele] == i:
                        total_r_p += 1
                if total_r != 0:
                    c_arr[i] = total_r_p / total_r * weight_r[i]
            recall = sum(c_arr)
            # print('recall is:', recall)
            # print(c_arr)
            print('')
            print('')
            print('')
            print('sklearn-precision-score:', precision_score(test_y, pre, average='weighted'))
            print('sklearn-recall-score:', recall_score(test_y, pre, average='weighted'))
            print('sklearn-f1-score:', f1_score(test_y, pre, average='weighted'))
            print('')
            print('')
            print('')


    test_svm()


