import tensorflow as  tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 10
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None, 28*28])
y = tf.placeholder(tf.float32,[None, 10])

W1 = tf.Variable(tf.zeros([784,10]))
b1 = tf.Variable(tf.zeros([10]))
Wx_plus_b_L1 = tf.matmul(x,W1) + b1
prediction = tf.nn.softmax(Wx_plus_b_L1)

loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iteration: " + str(epoch) + ", Testing Accuracy: " + str(acc))
    saver.save(sess,'net/my_net.ckpt')