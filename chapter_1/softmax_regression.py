# coding:utf-8
# 导入tensorflow。
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x，x是一个占位符（placeholder），代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784]) # None 表示这一维的大小是任意的，即可传递任意张图片给这个占位符
# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在TensorFlow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))
# b是又一个Softmax模型的参数（bias）
b = tf.Variable(tf.zeros([10]))

# y=softmax(Wx + b)，y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b) 
'''tf.matmul does matrix multiplication'''

# y_是实际的图像标签，同样以占位符表示。
y_ = tf.placeholder(tf.float32, [None, 10])

# 至此，我们得到了两个重要的Tensor：y和y_。
# y是模型的输出，y_是实际的图像标签，不要忘了y_是独热表示的
# 下面我们就会根据y和y_构造损失

# 根据y, y_构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
'''
"tf.reduce_mean": reduces input_tensor along the dimensions given in axis by computing the mean of elements across the dimensions in axis. Unless keepdims is true, the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are retained with length 1.

"tf.reduce_sum": reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are retained with length 1.
'''

# 有了损失，我们就可以用随机梯度下降针对模型的参数（W和b）进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 0.01 is learning rate

# 创建一个Session。只有在Session中才能运行优化步骤train_step。
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存。
tf.global_variables_initializer().run()
print('start training...')

# 在 session 中不需要系统计算占位符的值，而是直接把占位符的值传递给 session
# 与变量不同，占位符的值不会被保存，每次可以给占位符传递不同的值

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
    # batch_xs, batch_ys对应着两个占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
'''
"te.equal": Use equal to test each element for equality.
"tf.agrmax": Returns the index with the largest value across axes of a tensor. 取出数组中最大值的下标
tf.argmax(input, axis=None, name=None, dimension=None)
'''

# 计算预测准确率，它们都是Tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''
"tf.cast": casts x (in case of Tensor) or x.values (in case of SparseTensor or IndexedSlices) to dtype.
True -> 1, False ->0
'''
# 在Session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 0.9185
# feed_dict 传入占位符的值
