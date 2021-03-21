import tensorflow as tf
import random
from special_string import *
from load_resume_data import load_data_for_rnn_new_add_noise
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow.keras.layers import GRUCell, BatchNormalization, Dense, Embedding
from my_metric import MyPrecision, MyRecall

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


class LSPD(tf.keras.Model):
    def __init__(self, rnn_utils=32, rnn_layer_num=2, dense_num=3, embedding_size=32):
        super(LSPD, self).__init__()
        self.input_dim = None
        self.time_step = 0
        self.sample_size = 0
        self.rnn_utils = rnn_utils
        self.gru_cell_left = []
        self.gru_cell_right = []
        self.batchnormals = []
        self.denses = []
        self.embedding = None
        self.embedding_size = embedding_size
        self.denses_batchnormal = []
        self.rnn_layer_num = rnn_layer_num
        for i in range(rnn_layer_num):
            self.gru_cell_left.append(GRUCell(units=rnn_utils, activation='relu'))
            self.gru_cell_right.append(GRUCell(units=rnn_utils, activation='relu'))
            self.batchnormals.append(BatchNormalization())
        for i in range(dense_num-1):
            self.denses.append(Dense(units=32, activation='relu'))
            self.denses_batchnormal.append(BatchNormalization())
        self.dense = Dense(units=11, activation='softmax')

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.time_step = input_shape[-2]
        self.sample_size = input_shape[0]
        self.embedding = Embedding(self.input_dim, output_dim=self.embedding_size)
        pass

    def call(self, x, training=True):
        """
        (sample_size, time_step, input_dim) 最开始的输入是 软标签，需要先转成数字标签
        产生每一个时间步上的标签概率分布
        :param x:
        :return:
        """
        temp = tf.argmax(x, axis=2)
        temp = self.embedding(temp)  # (sample_size, time_step, output_dim)
        temp = tf.unstack(temp, axis=1)
        for i in range(self.rnn_layer_num):
            outputs = []
            outputs_right = []  # 存储逆向context（右边）
            outputs_left = []  # 存储正向context（左边）
            state_right, state_left = tf.fill([x.shape[0], self.rnn_utils], 0.0), tf.fill([x.shape[0], self.rnn_utils], 0.0)  # 初始状态
            length = len(temp)  # 时间步长
            # 得到当前双向gru层的输出
            for j, item in enumerate(temp):  # 左边context
                if j == 0:
                    outputs_left.append(None)
                    continue
                out_top, state_right = self.gru_cell_left[i](inputs=item, states=state_right)
                outputs_left.append(out_top)
            for j in range(length-1, -1, -1):  # 右边context
                if j == length - 1:
                    outputs_right.append(None)
                    continue
                out_top, state_left = self.gru_cell_right[i](temp[j], state_left)
                outputs_right.append(out_top)
            """
            对于一个时间步长为3的sequence:[0, 1, 2]
            example: outputs_left = [None, [0], [0, 1] ]
                     outputs_right = [[1, 2], [2], None]
                     合并后：outputs = [[1, 2], [0, 2], [0, 1] ]
            """
            for n, left in enumerate(outputs_left):  # left.shape:(sample_size, rnn_utils)
                if n == 0:
                    outputs.append(outputs_right[length-n-1])  # 左边界
                    continue
                if n == length - 1:
                    outputs.append(left)  # 右边界
                    continue
                outputs.append(tf.add(left, outputs_right[length-n-1]))
            temp = outputs  # 将当前双向GRU层的输出传递给temp作为下一层的输入, (sample_size, time_step, rnn_utils)
            temp = tf.stack(temp, axis=1)
            # print(temp[0].shape)
            temp = self.batchnormals[i](temp)
            temp = tf.unstack(temp, axis=1)
        outputs = []
        for item in temp:
            output = None
            for j, dense in enumerate(self.denses):
                output = dense(item)
                output = self.denses_batchnormal[j](output)
            output = self.dense(output)
            outputs.append(output)
        outputs = tf.stack(outputs, axis=1)
        return outputs


class FNNSubModel(tf.keras.Model):
    def __init__(self, time_step, feature_size):
        super(FNNSubModel, self).__init__()
        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_fnn = tf.keras.layers.Dense(units=11, activation='softmax')
        self.time_step = time_step
        self.drop = tf.keras.layers.Dropout(0.1)
        self.feature_size = feature_size
        self.batch_size = None

    def call(self, x, y=None, training=True, mode='predict'):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :param y:ground truth label
        :param mode:
        :return:
        """
        # print(x.shape)
        outputs_fnn = []
        for i, item in enumerate(tf.unstack(x, axis=1)):
            out_fnn = self.dense1(item)  # (sample_size, 64)  # fnn
            out_fnn = self.dense2(out_fnn)
            out_fnn = self.dense_fnn(out_fnn)
            outputs_fnn.append(out_fnn)
            # print('out_fnn.shape, final:', out_fnn.shape)
        outputs_fnn = tf.stack(outputs_fnn, axis=1)
        return outputs_fnn


class FNNModel(tf.keras.Model):
    def __init__(self, time_step, feature_size, T=0.5):
        super(FNNModel, self).__init__()
        self.dense_brnn = tf.keras.layers.Dense(units=11, activation='softmax')
        self.T = T
        self.fnn = FNNSubModel(time_step=time_step, feature_size=feature_size)
        self.lspd = LSPD(rnn_utils=64, embedding_size=100, rnn_layer_num=3, dense_num=2)
        self.time_step = time_step
        self.drop = tf.keras.layers.Dropout(0.1)
        self.feature_size = feature_size
        self.batch_size = None

    def call(self, x, y=None, training=True, mode='predict'):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :param y:ground truth label
        :param mode:
        :return:
        """
        # print(x.shape)
        outputs_fnn = self.fnn(x)
        if mode == 'predict':  # 测试模式
            outputs_lspd = self.lspd(outputs_fnn)
            return outputs_fnn, outputs_lspd

        else:  # 训练模式
            inputs_lspd = tf.add((1-self.T) * y, self.T * outputs_fnn)
            outputs_lspd = self.lspd(inputs_lspd)
            return outputs_fnn, inputs_lspd, outputs_lspd
        # 最终结果
        # outputs_fnn.shape:(sample_size, time_step, 11)
        # outputs_lspd.shape:(sample_size, time_step, 11)

    @staticmethod
    def cct(actual, pred, label_smoothing=0.):
        """
        计算N个样本的平均交叉熵
        :param actual: (batch_size, one_hot_size)
        :param pred: (batch_size, one_hot_size)
        :param label_smoothing:
        :return:loss
        """
        cross_entropy = tf.losses.categorical_crossentropy(actual, pred, label_smoothing=label_smoothing)
        return cross_entropy

    @staticmethod
    def my_loss(actual, predict, label_smoothing=0.):
        total_loss = 0
        length = actual.shape[1]
        actual_items = tf.unstack(actual, axis=1)  # shape:(batchsz, 10)
        predict_items = tf.unstack(predict, axis=1)  # shape:(batchsz, 10)
        for i in range(length):
            total_loss += FNNModel.cct(actual_items[i], predict_items[i], label_smoothing=label_smoothing)
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
            predictions_fnn, inputs_lspd, outputs_lspd = self.call(x=inputs, mode='training', y=labels)
            # loss = GRUModel.my_loss(labels, predictions)
            # loss_brnn = GRUModel.my_loss(labels, predictions_brnn)
            loss = FNNModel.my_loss(labels, predictions_fnn) + FNNModel.my_loss(inputs_lspd, outputs_lspd, label_smoothing=0.59)
            # print('loss:', loss)
            # print(self.trainable_variables)
            gradients = tape.gradient(loss, self.trainable_variables)  # 反向求导
            for i, gradient in enumerate(gradients):
                # print(gradient)
                if isinstance(gradient, tf.IndexedSlices):
                    continue
                gradients[i] = gradient * lr
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 参数更新
        train_loss.update_state(loss)
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions_fnn, [predictions_fnn.shape[0] * predictions_fnn.shape[1], predictions_fnn.shape[2]])
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
        N = 4 * epochs // 5
        for epoch in range(epochs):
            print('--------------------Epoch', epoch+1, '/', epochs,'-----------------------')
            # if epoch > N:
            #     self.T = 1
            # else:
            # self.T = (0.85/(N-1)) * epoch + 0.05
            if epochs < 8:
                self.T = 0.01
            else:
                self.T = 0.3
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
        outputs = self.call(inputs)[0]  # (sample_size, time_step, one_hot_size)
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

    def evaluate(self, inputs, labels, choose=0):
        """
        评估测试集上的表现
        :param inputs:输入样本 (sample_size, time_step, feature_size)
        :param labels:(sample_size, time_step, one_hot_size)
        :param choose:选择评估FNN或者LSPD
        :return:返回test_loss, test_acc, test_precision, test_recall
        """
        # test_acc, test_precision, test_recall = MyAccuracy(name='test_acc'), MyPrecision(name='test_precision'),
        # MyRecall(name='test_recall')
        predictions = self.call(inputs)[choose]
        # print(predictions)
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


if __name__ == '__main__':
    def evaluation_joint():
        train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=700, test_num=400, noise_percent=10)
        brnn = FNNModel(time_step=12, feature_size=100)
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
        inputs, label_list = None, None
        brnn.fit(train_x, train_y, batchsz=5, epochs=12)
        ev = brnn.evaluate(test_x, test_y, choose=0)
        # ev_b = brnn.evaluate(test_x, test_y, 1)
        # ev_f = brnn.evaluate(test_x, test_y, 2)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        # print(template.format(ev_b['precision'], ev_b['recall'], ev_b['f1-score']))
        # print(template.format(ev_f['precision'], ev_f['recall'], ev_f['f1-score']))
        # model_path = ROOT_PATH + '\\fnn_11_30'
        brnn.save_weights(JOINT_100_PATH)
        # print(brnn.layers)

    def evaluation_load_model():
        test_x, test_y = load_data_for_rnn_new_add_noise(load_train=False, data_set=3, train_num=700, test_num=400, noise_percent=10)
        brnn = FNNModel(time_step=12, feature_size=100)
        brnn.load_weights(JOINT_100_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_2021_2_5, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        ev = brnn.evaluate(test_x, test_y, choose=1)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))

    def test_lspd_output():
        lspd = LSPD(rnn_utils=32, dense_num=3, embedding_size=32)
        inputs = tf.random.normal([5, 12, 11])
        # inputs = tf.constant([
        #     [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
        # ]
        # )
        # print('inputs:', inputs)
        outputs = lspd(inputs)
        # tf.keras.Model.summary(lspd)
        print(outputs)

    evaluation_joint()

