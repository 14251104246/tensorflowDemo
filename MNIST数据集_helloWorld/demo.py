# -*- coding: utf-8 -*-  
#coding=utf-8  
import input_data
import tensorflow as tf
#训练数据集合
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#通过操作符号变量来描述这些可交互的操作单元
#x不是一个特定的值，而是一个占位符placeholder
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#概率
y = tf.nn.softmax(tf.matmul(x,W) + b)

#为了计算交叉熵，添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#最小化成本值，梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#添加一个操作来初始化我们创建的变量
init = tf.initialize_all_variables()

#在一个Session里面启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)

#让模型循环训练1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
#tf.argmax 函数能给出某个tensor对象在某一维上的其数据最大值所在的索引值
#tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#最后，打印我们计算所学习到的模型在测试数据集上面的正确率。
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})