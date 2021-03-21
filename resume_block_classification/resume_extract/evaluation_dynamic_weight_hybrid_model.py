# coding=utf-8

"""
@File  : evaluation_dynamic_decision_hybrid_model.py
"""
import tensorflow as tf
import random
from special_string import *
from static_num import *
from load_resume_data import load_data_for_rnn_new_add_noise
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
from my_metric import MyRecall, MyPrecision
from evaluation_joint import FNNModel
import matplotlib.pyplot as plt

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


class BRNNFeatureFusion(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=1):
        super(BRNNFeatureFusion, self).__init__()
        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense = tf.keras.layers.Dense(units=11, activation='softmax')
        self.time_step = time_step
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.1)
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
        x1 = x
        for i in range(self.rnn_layers_num):
            x1 = self.gru[i](x1)
        outputs = []
        for i, item in enumerate(tf.unstack(x1, axis=1)):
            out_brnn = self.dense0(item)  # (sample_size, 64)  # rnn
            out_fnn = self.dense1(x[:, i, :])  # (sample_size, 64)  # fnn
            out_fnn = self.dense2(out_fnn)
            out = tf.concat([out_brnn, out_fnn], axis=1)  # (sample_size, 128)
            out = self.batch_normalization1(out)
            out1 = self.drop(out)
            out2 = self.dense3(out1)
            out2 = self.batch_normalization2(out2)
            out = tf.concat([out1, out2], axis=1)  # (sample_size, 256)
            out = self.dense(out)
            outputs.append(out)
        return tf.stack(outputs, axis=1)  # 最终结果shape:(batchsz, time_step, 11)

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
            total_loss += BRNNFeatureFusion.cct(actual_items[i], predict_items[i])
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
            # loss = GRUModel.my_loss(labels, predictions)
            # loss_brnn = GRUModel.my_loss(labels, predictions_brnn)
            loss= BRNNFeatureFusion.my_loss(labels, predictions)
            # print('loss:', loss)
            # print(self.trainable_variables)
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
            elif epoch >= 2*epochs/3:
                lr = 0.05
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
                    # print('predict y is:', y_pre)
                    labels = []
                    for i in range(y_pre.shape[0]):
                        labels.append(number2label[tf.gather(y_pre, i).numpy()])
                    y_pre_list.append(labels)
            else:
                return None
        else:
            samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
            for sample in samples:
                # print(sample.numpy())
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
        t_pre = precision_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        t_rec = recall_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        f1 = f1_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        # test_precision.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # test_recall.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # t_acc, t_pre, t_rec = test_acc.result(), test_precision.result(), test_recall.result()
        return {'precision': t_pre, 'recall': t_rec, 'f1-score': f1}


class DynamicWeightHybridModel(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=2, brnn_model_path=None, fnn_model_path=None):
        super(DynamicWeightHybridModel, self).__init__()
        self.time_step = time_step
        self.batch_size = None
        self.brnn= BRNNFeatureFusion(time_step=time_step, feature_size=feature_size, rnn_utils=rnn_utils, rnn_layers_num=rnn_layers_num)
        if brnn_model_path is not None:
            self.brnn.load_weights(brnn_model_path)
        # for layer in self.brnn.layers:
        #     layer.trainable = False
        self.brnn.trainable = False
        self.fnn = FNNModel(time_step=time_step, feature_size=feature_size)
        if fnn_model_path is not None:
            self.fnn.load_weights(fnn_model_path)
        self.fnn.trainable = False
        # for i in range(len(self.fnn.layers)):
        #     self.fnn.layers[i].trainable = False

    def call(self, x, training=True, return_weight=False):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :param return_weight:
        :return:
        """
        # print(x.shape)
        out_brnn = self.brnn(x)  # (sample_size, time_step, 11)
        out2_fnn, out2_lspd = self.fnn(x)  # (sample_size, time_step, 11)
        out2_fnn1 = tf.argmax(out2_fnn, axis=-1)  # (sample_size, time_step)
        out2_fnn1 = tf.one_hot(out2_fnn1, depth=11, axis=-1)  # (sample_size, time_step, 11) 得到fnn的one_hot输出
        weight = tf.multiply(out2_lspd, out2_fnn1)  # 相当于做掩码，只保留对应于fnn预测类别的概率 (sample_size, time_step, 11)
        weight = tf.reduce_sum(weight, axis=2)  # (sample_size, time_step)
        weight = tf.stack([weight], axis=-1)
        weight = tf.broadcast_to(weight, shape=[weight.shape[0], weight.shape[1], 11])  # (sample_size, time_step, 11)
        # mean_weight = tf.reduce_mean(weight, axis=0)
        sum_weight = tf.reduce_sum(weight, axis=1)  # sample_size, 11
        sum_weight = tf.reduce_mean(sum_weight, axis=1)  # (sample_size, )
        # print('weights of all samples are:', sum_weight)
        # print(weight[0])
        weight_fnn = tf.ones(weight.shape) - weight
        out = tf.add(tf.multiply(weight_fnn, out2_fnn), tf.multiply(weight, out_brnn))
        print(out.shape)
        if return_weight:
            return out, sum_weight
        return out  # 最终结果shape:(batchsz, time_step, 11)

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
            total_loss += DynamicWeightHybridModel.cct(actual_items[i], predict_items[i])
        return total_loss

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

    def evaluate(self, inputs, labels, return_weight=False):
        """
        评估测试集上的表现
        :param inputs:输入样本 (sample_size, time_step, feature_size)
        :param labels:(sample_size, time_step, one_hot_size)
        :return:返回test_loss, test_acc, test_precision, test_recall
        :param return_weight:
        """
        weights = None
        if return_weight:
            predictions, weights = self.call(inputs, return_weight=return_weight)
        else:
            predictions = self.call(inputs)
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions, [predictions.shape[0] * predictions.shape[1], predictions.shape[2]])
        # 去除真实标签onehot全为0的空白标记
        y_true, y_pred = tf.unstack(y_true, axis=0), tf.unstack(y_pred)
        for i, label in enumerate(y_true):
            if label.numpy()[-1] == 1:
                y_true.pop(i)
                y_pred.pop(i)
        y_true, y_pred = tf.stack(y_true, axis=0), tf.stack(y_pred, axis=0)
        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32).numpy()
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32).numpy()
        t_pre = precision_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        t_rec = recall_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        f1 = f1_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        if return_weight:
            return {'precision': t_pre, 'recall': t_rec, 'f1-score': f1}, weights
        return {'precision': t_pre, 'recall': t_rec, 'f1-score': f1}


if __name__ == '__main__':
    def evaluation_dynamic_wieght_hybrid_model():
        test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=1, test_num=400, load_train=False, noise_percent=0)
        brnn = DynamicWeightHybridModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM, rnn_utils=RNN_UTILS,
                                        rnn_layers_num=RNN_LAYERS_NUM, brnn_model_path=BRNN_700_PATH, fnn_model_path=JOINT_100_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        # print(train_y)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=WORD2VEC_FEATURE_NUM,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        ev, weights = brnn.evaluate(test_x, test_y, return_weight=True)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        brnn.save_weights(DW_HYBRID_600_PATH)


    def evaluation_dynamic_weight_load_model():
        test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=1, test_num=400,
                                                                           noise_type='shuffle', noise_percent=0, load_train=False)
        test_x_noise, test_y_noise = load_data_for_rnn_new_add_noise(data_set=3, train_num=1, test_num=400,
                                                         noise_type='shuffle', noise_percent=10, load_train=False)
        brnn = DynamicWeightHybridModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM, rnn_utils=RNN_UTILS,
                                        rnn_layers_num=RNN_LAYERS_NUM)
        brnn.load_weights(DW_HYBRID_700_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)

        process_rnn_label_list(test_y_noise, time_step=brnn.time_step, begin=begin)
        test_x_noise = trans_to_wordvec_by_word2vec(test_x_noise, feature_size=100,
                                              word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                                              time_step=brnn.time_step, begin=begin)
        test_x_noise, test_y_noise = tf.constant(test_x_noise, dtype=tf.float32), tf.constant(test_y_noise, dtype=tf.float32)
        ev_noise, weight_noise = brnn.evaluate(test_x_noise, test_y_noise, return_weight=True)
        ev, weight = brnn.evaluate(test_x, test_y, return_weight=True)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        print(template.format(ev_noise['precision'], ev_noise['recall'], ev_noise['f1-score']))
        x = [i for i in range(0, 400)]
        # colors = []
        # for i in range(400):
        #     colors.append('r')
        # for i in range(400, 800):
        #     colors.append('g')
        # plt.scatter(x, tf.concat([weight, weight_noise], axis=-1), c=colors, s=20)
        # plt.scatter(x, weight_noise, c=['g'], s=20)
        plt.scatter(x, weight_noise, c="r", alpha=0.5, label="abnormal test set", s=15, marker='^')
        # 第二个散点图，颜色为蓝色，透明度50%，图例为散点图2
        plt.scatter(x, weight, c="g", alpha=0.5, label="normal test set", s=15, marker='*')
        plt.xlabel('index of resume sample')
        plt.ylabel('total weight value of each time step ')
        plt.legend(loc='best')
        plt.title('weight value distribution')
        plt.savefig(ROOT_PATH + '\\lspd_weight_distribution.pdf')
        plt.show()

    evaluation_dynamic_weight_load_model()
    # evaluation_mbrnn()
    # test_model()




