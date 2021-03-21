# coding=utf-8

"""
@File  : my_metric.py
@Author: Xu Qiqiang
@Date  : 2020/11/20 0020
"""
import tensorflow as tf
import math


class MyRecall(tf.keras.metrics.Metric):
    def __init__(self, name=None, classes=10):
        """
        召回率 :所有实际为正例的类别预测也为正例的数量占所有实际为正例的数量的比例
        :param name:
        :param classes:类别总数
        """
        super().__init__(name=name)
        self.true_positives, self.true_nagtives = [], []  # 总共有classes个类别，每个类别都对应一个预测为正例且真实为正确， 预测为正例且真实为错误的指标
        for i in range(classes):
            self.true_positives.append(self.add_weight(name='true_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))
            self.true_nagtives.append(self.add_weight(name='false_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))

    def update_state1(self, y_pred_val, y_true_val, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true_val: 真实类别序列
        :param y_pred_val: 预测类别序列
        :param sample_weight:
        :return:
        """
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        # print(y_true_argmax.shape)
        length = y_true_val.shape[0]
        # print('length is:', length)
        for i in range(length):
            if y_pred_val[i] == y_true_val[i]:
                self.true_positives[y_true_val[i]].assign_add(1)
            else:
                self.true_nagtives[y_true_val[i]].assign_add(1)

    def update_state(selfself, pre, act, sample_weight=None):
        pass

    def result(self):
        TP, TN = tf.stack(self.true_positives, axis=0), tf.stack(self.true_nagtives, axis=0)
        res = tf.reduce_mean(TP/(TP + TN))
        if math.isnan(res):
            return 0.0
        return res


class MyPrecision(tf.keras.metrics.Metric):
    def __init__(self, name=None, classes=10):
        """
        精确率 :所有预测为正例的类别实际也为正例的数量占所有预测为正例的数量的比例
        :param name:
        :param classes:类别总数
        """
        super().__init__(name=name)
        self.true_positives, self.false_positives = [], []  # 总共有classes个类别，每个类别都对应一个预测为正例且真实为正确， 预测为正例且真实为错误的指标
        for i in range(classes):
            self.true_positives.append(self.add_weight(name='true_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))
            self.false_positives.append(self.add_weight(name='false_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))

    def update_state1(self, y_pred_val, y_true_val, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true: 真实类别序列 shape:(sample_size, time_step, one_hot_size)
        :param y_pred: 预测类别序列 shape:(sample_size, time_step, one_hot_size)
        :param sample_weight:
        :return:
        """
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        # print(y_true_argmax.shape)
        length = y_true_val.shape[0]
        # print('length is:', length)
        for i in range(length):
            if y_pred_val[i] == y_true_val[i]:
                self.true_positives[y_pred_val[i]].assign_add(1)
            else:
                self.false_positives[y_pred_val[i]].assign_add(1)

    def update_state(selfself, pre, act, sample_weight=None):
        pass

    def result(self):
        TP, FP = tf.stack(self.true_positives, axis=0), tf.stack(self.false_positives, axis=0)
        res = tf.reduce_mean(TP/(TP + FP))
        if math.isnan(res):
            return 0.0
        return res


class MyAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true: 真实类别序列 shape:(sample_size, time_step, one_hot_size)
        :param y_pred: 预测类别序列 shape:(sample_size, time_step, one_hot_size)
        :param sample_weight:
        :return:
        """
        y_true = tf.reshape(y_true, [y_true.shape[0]*y_true.shape[1], y_true.shape[2]])
        y_pred = tf.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]])
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        # print('y_true_argmax.shape:', y_true_argmax.shape)

        values = tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.int32)
        # print('values.shape:', values.shape)
        # self.total = self.total+tf.shape(y_true)[0]
        self.total.assign_add(y_true.shape[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

