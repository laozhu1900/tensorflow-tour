# coding:utf-8

# 参考：知乎专栏《超智能体》： https://zhuanlan.zhihu.com/p/27853766  作者：YJango

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class FNN(object):
    """
    build a general FeedForward neural network
    :parameter
    learning_rate: float
    drop_out: float
    Layers: list
        The number of layers
    N_hidden: list
        The numbers of nodes in layers
    D_input: int
        Label dimension
    Task_type: string
        'regression' or 'classification'
    L2_lambda: float

    """

    def __init__(self, learning_rate, drop_keep, Layers, N_hidden, D_input,
                 D_label, Task_type='regression', L2_lambda=0.0):
        # 全部共有属性
        self.learning_rate = learning_rate
        self.drop_keep = np.array(drop_keep).astype(np.float32)
        self.Layers = Layers
        self.N_hidden = N_hidden
        self.D_input = D_input
        self.D_label = D_label

        # 类型控制los函数的选择
        self.Task_type = Task_type
        # L2 regularization的惩罚强弱，过高会使得输出都拉向0
        self.L2_lambda = L2_lambda

        # 用于存放所累积的每层l2 regularization
        self.l2_penalty = tf.constant(0.0)

        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, D_input], name='inputs')
        with tf.name_scope('Label'):
            self.labels = tf.placeholder(tf.float32, [None, D_label], name='labels')
        with tf.name_scope('keep_rate'):
            self.drop_keep_rate = tf.placeholder(tf.float32, name='dropout_keep')

        self.l2_penalty = tf.constant(0.0)
        # 初始化直接生成
        self.build('F')

    def weight_init(self, shape):
        # shape: list [in_dim, out_dim]
        # can change initialization here
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_init(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var, name):
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope(name + "_stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        # 记录每次训练后变量的数值变化
        # 随着训练记录一个变量的最大值、最小值、方差、的变化，以及直方图
        tf.summary.scalar('_stddev/' + name, stddev)
        tf.summary.scalar('_max/' + name, tf.reduce_max(var))
        tf.summary.scalar('_min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

    def layer(self, in_tensor, in_dim, out_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name + '_weights'):
                # 用所建立的weight_int()
                weights = self.weight_init([in_dim, out_dim])

                # 存放着每一个权重
                self.W.append(weights)
                self.variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope(layer_name + '_biases'):
                biases = self.bias_init([out_dim])
                self.variable_summaries(biases, layer_name + '/biases')

            with tf.name_scope(layer_name + '_Wx_plus_b'):
                # 计算Wx+b
                pre_activate = tf.matmul(in_tensor, weights) + biases

                # 记录直方图
                tf.summary.histogram(layer_name + '/pre_activations', pre_activate)

            # 计算 a(Wx+b)
            activations = act(pre_activate, name='activation')
            tf.summary.histogram(layer_name + "/activations", activations)
        # 返回该层的输出， 以及权重W的L2
        return activations, tf.nn.l2_loss(weights)

    def drop_layer(self, in_tensor):
        dropped = tf.nn.dropout(in_tensor, self.drop_keep_rate)
        return dropped

    def build(self, prefix):
        # 建立网络
        # incoming也代表当前tensor的流动位置
        incoming = self.inputs
        # 如果没有隐藏层
        if self.Layers != 0:
            layer_nodes = [self.D_input] + self.N_hidden
        else:
            layer_nodes = [self.D_input]

        # hid_layers用于存储所有隐藏层的输出
        self.hid_layers = []
        # hid_layers 用于存储所有层的权重
        self.W = []
        # b用于存储所有层的偏移
        self.b = []
        # total_l2 用于存储素有层的L2
        self.total_l2 = []
        # drop存储dropout后的输出
        self.drop = []

        # 开始叠加隐藏层
        for l in range(self.Layers):
            # 使用刚才编写的函数建立层，更新incoming的位置
            incoming, l2_loss = self.layer(incoming, layer_nodes[l], layer_nodes[l + 1], prefix + '_hid_' + str(l + 1),
                                           act=tf.nn.relu)
            # 累计l2
            self.total_l2.append(l2_loss)

            print('Add dense layer: relu with drop_keep: %s' % self.drop_keep)
            print('    %sD --> %sD' % (layer_nodes[l], layer_nodes[l + 1]))

            # 存储所有隐藏层的输出
            self.hid_layers.append(incoming)

            # 加入dropout层
            incoming = self.drop_layer(incoming)
            # 存储所有dropout后的输出
            self.drop.append(incoming)

        # 输出层的建立。输出层需要特别对待的原因是输出层的activation function要根据任务来变。
        # 回归任务的话，下面用的是tf.identity，也就是没有activation function

        if self.Task_type == 'regression':
            out_act = tf.identity

        else:
            # 分类任务使用softmax 来你还拟合概率
            out_act = tf.nn.softmax

        self.output, l2_loss = self.layer(incoming, layer_nodes[-1], self.D_label, layer_name='output', act=out_act)
        self.total_l2.append(l2_loss)
        print('Add output layer: linear')
        print('    %sD --> %sD' % (layer_nodes[-1], self.D_label))

        # l2 loss缩放图
        with tf.name_scope('total_l2'):
            for l2 in self.total_l2:
                self.l2_penalty += l2
            tf.summary.scalar('l2_penalty', self.l2_penalty)

        # 不同任务的loss
        # 若为回归， 则loss是用于判断所有预测值和实际值差别的函数

        if self.Task_type == 'regression':
            with tf.name_scope('SSE'):
                self.loss = tf.reduce_mean((self.output - self.labels) ** 2)

                tf.summary.scalar('loss', self.loss)


        else:

            # 若为分类，cross entropy的loss function
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
            with tf.name_scope('cross entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        # 整合所有loss，形成最终loss
        with tf.name_scope('total_loss'):
            self.total_loss = self.loss + self.l2_penalty * self.L2_lambda
            tf.summary.scalar('total_loss', self.total_loss)

        # 训练操作
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)


if __name__ == '__main__':
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]
    X = np.array(inputs).reshape((4, 1, 2)).astype('int16')
    Y = np.array(outputs).reshape((4, 1, 1)).astype('int16')
    # 生成网络实例
    # 初始化学习速率为0.001, 没有dropout(1.0表示全部保留，一个不扔，一个隐藏层(layers表示隐藏层的个数), 输入维度是2 ，目标维度是1)
    # 回归任务，L2的惩罚强度是0.01,生成后, 程序会按照事先编写的格式输出一些内容。随后我们就可以用ff.xxx的形式来获取ff内的所有属性
    ff = FNN(learning_rate=1e-3, drop_keep=1.0, Layers=1, N_hidden=[2], D_input=2, D_label=1, Task_type='regression',
             L2_lambda=1e-2)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 将所有的summary合成一个ops
    merged = tf.summary.merge_all()

    # 用于记录训练中内容，前一个参数是记录地址，后一个参数是记录的graph
    train_writer = tf.summary.FileWriter('log' + "/train", sess.graph)

    W0 = sess.run(ff.W[0])
    W1 = sess.run(ff.W[1])

    print('W_0:\n%s' % sess.run(ff.W[0]))
    print('W_1:\n%s' % sess.run(ff.W[1]))

    # 训练前的输出
    pY = sess.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
    print(pY)
    # 画图（4个数据循序用红、绿、蓝、黑表示）
    plt.scatter([0, 1, 2, 3], pY, color=['red', 'green', 'blue', 'black'], s=25, alpha=0.4, marker='o')

    print('训练前隐藏层的输出')
    pY = sess.run(ff.hid_layers[0], feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
    print(pY)

    print('当keep_rate：1时，保持所有节点（就是隐藏层的原有输出')
    print(sess.run(ff.drop[0],
                   feed_dict={ff.inputs: X.reshape((4, 2)), ff.labels: Y.reshape((4, 1)), ff.drop_keep_rate: 1}))

    print('当keep_rate：0.5时，保持所有节点（就是隐藏层的原有输出')
    print(sess.run(ff.drop[0],
                   feed_dict={ff.inputs: X.reshape((4, 2)), ff.labels: Y.reshape((4, 1)), ff.drop_keep_rate: 0.5}))

    # 训练并记录
    # k表示训练了多少次
    k = 0.0
    # i每增加1，就表示所有的训练数据偶读被训练完了一次。叫做epoch
    for i in range(10000):
        k +=1
        # summary是merged得出的值，即所有统计内容
        summary, _ = sess.run([merged, ff.train_step],
                              feed_dict={ff.inputs: X.reshape((4, 2)), ff.labels: Y.reshape((4, 1)),
                                         ff.drop_keep_rate: 1.0})
        # 将统计内容写入指定log文件中
        train_writer.add_summary(summary, k)


    # 权重的读取

    # # 该操作可以用于读取已经训练好的权重W和b
    # # 每层W想要读取的值
    # W_0 = np.array([[-0.82895017, 0.82891428], [0.82915729, -0.82918972]], dtype='float32')
    # W_1 = np.array([[1.17231631], [1.1722393]], dtype='float32')
    # # 每层b想要读取的值
    # b_0 = np.array([0, 0], dtype='float32')
    # b_1 = np.array([0], dtype='float32')
    # # 读取ops
    # reload1 = tf.assign(ff.W[0], W_0)
    # reload2 = tf.assign(ff.W[1], W_1)
    # reload3 = tf.assign(ff.b[0], b_0)
    # reload4 = tf.assign(ff.b[1], b_1)
    # # 执行ops
    # print(sess.run([reload1, reload2, reload3, reload4]))
