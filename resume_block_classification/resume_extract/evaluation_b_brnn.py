import tensorflow as tf
import random
from special_string import *
from load_resume_data import load_data_for_rnn_new_add_noise
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
import math

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


class MyDense(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyDense, self).__init__()
        self.n_outputs = n_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                        shape=[int(input_shape[-1]),
                                               self.n_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


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

    # def reset_states(self):
    #     self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
    #     self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())


class BBRNNModel(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=1):
        super(BBRNNModel, self).__init__()
        self.dense = tf.keras.layers.Dense(units=11, activation='softmax')
        self.time_step = time_step
        self.feature_size = feature_size
        self.rnn_layers_num = rnn_layers_num
        self.rnn_utils_num = rnn_utils
        self.batch_size = None
        self.gru = []
        for i in range(rnn_layers_num):
            self.gru.append(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rnn_utils, return_sequences=True, activation='relu')))

    def call(self, x, training=True):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :return:
        """
        # print(x.shape)
        for i in range(self.rnn_layers_num):
            x = self.gru[i](x)
        outputs = []
        for i, item in enumerate(tf.unstack(x, axis=1)):
            out = self.dense(item)
            outputs.append(out)
        return tf.stack(outputs, axis=1)  # 最终结果shape:(batchsz, time_step, 10)

    @staticmethod
    def cct(actual, pred):
        """
        计算N个样本的平均交叉熵
        :param actual: (batch_size, one_hot_size)
        :param pred: (batch_size, one_hot_size)
        :return:loss
        """
        actual, pred = tf.unstack(actual, axis=0), tf.unstack(pred, axis=0)
        length = len(actual)
        cross_entropy = 0
        for i in range(length):
            # if np.all(actual[i].numpy() == 0):
            # if actual[i].numpy()[-1] == 1:
            #     continue
            cross_entropy += -tf.reduce_sum(actual[i] * tf.math.log(tf.clip_by_value(pred[i], 1e-10, 1.0)))
        return cross_entropy

    @staticmethod
    def my_loss(actual, predict):
        # predict, actual shape:(batchsz, time_step,10)
        # print("actual is", actual)
        # print("predict is", predict)
        # actual = tf.one_hot(actual, depth=10)
        # print(actual.shape)
        total_loss = 0
        length = actual.shape[1]
        actual_items = tf.unstack(actual, axis=1)  # shape:(batchsz, 10)
        predict_items = tf.unstack(predict, axis=1)  # shape:(batchsz, 10)
        # cct = tf.keras.losses.CategoricalCrossentropy()
        for i in range(length):
            total_loss += BBRNNModel.cct(actual_items[i], predict_items[i])
        return total_loss

    def train_step(self, inputs, labels, optimizer, train_loss, train_precision, train_recall, lr=0.1):  # 执行一个batch样本数量
        # labels.shape:(batch_size, time_step, 10)
        # print('inputs.shape:', inputs.shape)
        # print(inputs)
        if inputs.shape[0] < self.batch_size:
            print('not equal batch_size, concat to batch_size', inputs.shape, labels.shape)
            # 扩充至batch_size
            slices = tf.unstack(inputs, axis=0)  # 解堆叠，得到每一个样本
            slices_labels = tf.unstack(labels, axis=0)
            extra_index = [random.randint(0, len(slices) - 1) for i in range(self.batch_size - inputs.shape[0])]
            extra = []
            extra_labels = []
            length = self.batch_size - inputs.shape[0]
            for i in range(length):
                extra.append(slices[extra_index[i]])
                extra_labels.append(slices_labels[extra_index[i]])
            slices.extend(extra)  # 扩展至batch_size个
            slices_labels.extend(extra_labels)
            inputs = tf.stack(slices, axis=0)
            labels = tf.stack(slices_labels, axis=0)
            # extra_labels, extra_index, extra = None, None, None
            print(inputs.shape, labels.shape)

        with tf.GradientTape() as tape:
            predictions = self.call(inputs)
            loss = BBRNNModel.my_loss(labels, predictions)
            # print('loss:', loss)
            gradients = tape.gradient(loss, self.trainable_variables)  # 反向求导
            for i, gradient in enumerate(gradients):
                gradients[i] = gradient * lr
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 参数更新
        train_loss.update_state(loss)
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions, [predictions.shape[0] * predictions.shape[1], predictions.shape[2]])
        # 去除真实标签onehot全为0的空白标记
        y_true, y_pred = tf.unstack(y_true, axis=0), tf.unstack(y_pred)
        for i, label in enumerate(y_true):
            # if np.all(label.numpy() == 0):
                # print('white space mark:', label.numpy())
            if label.numpy()[-1] == 1:
                y_true.pop(i)
                y_pred.pop(i)
        y_true, y_pred = tf.stack(y_true, axis=0), tf.stack(y_pred, axis=0)

        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32).numpy()
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32).numpy()
        # print(y_pred_argmax)
        train_precision.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        train_recall.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # print(labels, predictions)

        # train_loss(loss)
        # train_acc(labels, predictions)

    def fit(self, x_train, y_train, batchsz=16, epochs=5):
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_acc = MyAccuracy(name='train_acc')
        train_precision = MyPrecision(name='precision', classes=11)
        train_recall = MyRecall(name='recall', classes=11)
        self.batch_size = batchsz
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batchsz)
        if x_train.shape[0] % batchsz == 0:
            batch_count = x_train.shape[0] // batchsz
        else:
            batch_count = x_train.shape[0] // batchsz + 1
        for epoch in range(epochs):
            print('--------------------Epoch', epoch+1, '/', epochs,'-----------------------')
            # 在下一次epoch开始时，重置评估指标
            train_loss.reset_states()
            # train_acc.reset_states()
            train_recall.reset_states()
            train_precision.reset_states()
            count = 0
            if epoch >= 3*epochs/4:
                lr = 0.01
            else:
                lr = 0.1
            for inputs, labels in train_ds:  # 执行一个batch
                # print(inputs, labels)
                self.train_step(inputs, labels, optimizer, train_loss, train_precision, train_recall, lr)
                count += 1
                template = 'Batch {}/{}, Loss: {}, Precision:{}%, Recall:{}%'
                print(template.format(count, batch_count, train_loss.result(), train_precision.result(), train_recall.result()))
            print('')
            template = 'Epoch {}, Loss: {}, Precision:{}%, Recall:{}%'
            print(template.format(epoch + 1, train_loss.result(), train_precision.result(), train_recall.result()))
            for i in range(10):
                print('')

    def predict(self, inputs, number2label=None):
        """

        :param inputs: 输入样本数据 shape:(sample_size, time_step, feature_size)
        :param number2label: 索引对应标签名，若不为空，则返回具体类别名字 type:list
        :return:[[label1, label2,...], [label1, label2,...], ...]  label为字符串或者数字索引
        """
        outputs = self.call(inputs)  # (sample_size, time_step, one_hot_size)
        y_pre_list = []
        if number2label is not None:
            if isinstance(number2label, list):
                samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
                for sample in samples:
                    y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                    labels = []
                    for i in range(y_pre.shape[0]):
                        labels.append(number2label[tf.gather(y_pre, i).numpy()])
                    y_pre_list.append(labels)
            else:
                return None
        else:
            samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
            for sample in samples:
                print(sample.numpy())
                y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                labels = []
                for i in range(y_pre.shape[0]):
                    labels.append(tf.gather(y_pre, i).numpy())
                y_pre_list.append(labels)
        return y_pre_list

    def evaluate(self, inputs, labels):
        """
        评估测试集上的表现
        :param inputs:输入样本 (sample_size, time_step, feature_size)
        :param labels:(sample_size, time_step, one_hot_size)
        :return:返回test_loss, test_acc, test_precision, test_recall
        """
        # test_acc, test_precision, test_recall = MyAccuracy(name='test_acc'), MyPrecision(name='test_precision'),
        # MyRecall(name='test_recall')
        predictions = self.call(inputs)
        # test_acc.update_state(labels, predictions)
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions, [predictions.shape[0] * predictions.shape[1], predictions.shape[2]])
        # 去除空白标记
        y_true, y_pred = tf.unstack(y_true, axis=0), tf.unstack(y_pred)
        for i, label in enumerate(y_true):
            # if np.all(label.numpy() == 0):
                # print('white space mark:', label.numpy())
            if label.numpy()[-1] == 1:
                y_true.pop(i)
                y_pred.pop(i)
        y_true, y_pred = tf.stack(y_true, axis=0), tf.stack(y_pred, axis=0)
        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32).numpy()
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32).numpy()
        t_pre = precision_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        t_rec = recall_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        f1 = f1_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        return {'precision': t_pre, 'recall': t_rec, 'f1-score': f1}


if __name__ == '__main__':
    def evaluation_b_brnn():
        train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=700, train_noise_percent=9, test_num=400, noise_percent=0)
        brnn = BBRNNModel(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=2)
        begin = 0
        process_rnn_label_list(train_y, time_step=brnn.time_step, begin=begin)  # 原地修改label_list,统一维度
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        # print(train_y)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn', time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                time_step=brnn.time_step, begin=begin)
        train_x, train_y = tf.constant(train_x, dtype=tf.float32), tf.constant(train_y, dtype=tf.float32)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        brnn.fit(train_x, train_y, batchsz=5, epochs=12)
        ev = brnn.evaluate(test_x, test_y)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        # model_path = ROOT_PATH + '\\b_brnn_model_11-29'
        brnn.save_weights(NOISE_9_BRNN_PATH)

    def evaluation_b_brnn_load_model():
        """
        evaluate the b_brnn model.
        There are seven models, which are trained with 100,200,300,400,500,600,700 resumes.
        BRNN_700_PATH----the path of brnn model which are trained with 700 resumes.
        BRNN_600_PATH----the path of brnn model which are trained with 600 resumes.
        BRNN_500_PATH----the path of brnn model which are trained with 500 resumes.
        BRNN_400_PATH----the path of brnn model which are trained with 400 resumes.
        BRNN_300_PATH----the path of brnn model which are trained with 300 resumes.
        BRNN_200_PATH----the path of brnn model which are trained with 200 resumes.
        BRNN_100_PATH----the path of brnn model which are trained with 100 resumes.
        use the noise_percent to load test data that contains abnormal sequence features resume sample.
        for instance:
        noice_percent=10, means that the proportion of abnormal sample is 100%
        noice_percent=2, means that the proportion of abnormal sample is 20%
        :return:
        """
        test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, test_num=400, noise_percent=0, load_train=False)
        brnn = BBRNNModel(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=2)
        brnn.load_weights(BRNN_700_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        ev = brnn.evaluate(test_x, test_y)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))


    evaluation_b_brnn_load_model()



