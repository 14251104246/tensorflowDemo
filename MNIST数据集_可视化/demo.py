# -*- coding: utf-8 -*-  
#coding=utf-8  
import input_data
import tensorflow as tf
#加载训练数据集合
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#为输入图像和目标输出类别创建节点
#x和y并不是特定的值，他们都只是一个占位符
x = tf.placeholder("float", shape=[None, 784])#输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度
y_ = tf.placeholder("float", shape=[None, 10])#输出类别值y_是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别

#在机器学习的应用过程中，模型参数一般用Variable(变量)来表示
#一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用、修改
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#在一个Session里面启动我们的模型，并且初始化变量
sess = tf.InteractiveSession()
#变量需要通过seesion初始化后，才能在session中使用
#这里一次性为所有变量分配0
sess.run(tf.initialize_all_variables())

#回归模型
#我们把向量化后的图片x和权重矩阵W相乘，加上偏置b
y = tf.nn.softmax(tf.matmul(x,W) + b)

#为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#添加到可视化
tf.summary.scalar('cross_entropy', cross_entropy)

#用最速下降法让交叉熵下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Merge all the summaries and write them out to
# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./summary" + '/train', sess.graph)
test_writer = tf.summary.FileWriter(".summary" + '/test')

#返回的train_step操作对象，在运行时会使用梯度下降来更新参数
for i in range(1000):
  #加载50个训练样本
  batch = mnist.train.next_batch(50)
  #反复地运行train_step
  feed_dict={x: batch[0], y_: batch[1]}
  '''运行同时记录数据到summary'''
  summary, _=sess.run([merged, train_step],feed_dict)
  train_writer.add_summary(summary, i)

#tf.argmax 函数能给出某个tensor对象在某一维上的其数据最大值所在的索引值
#tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#最后，打印我们计算所学习到的模型在测试数据集上面的正确率。
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})