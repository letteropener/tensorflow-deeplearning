import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

# graph variable
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# tensorboard name scope
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 28*28], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name = 'y-input')


keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

with tf.name_scope('layer1'):
    with tf.name_scope('weights1'):
        W1 = tf.Variable(tf.truncated_normal([28*28, 500],stddev=0.1),name='W1')
        variable_summaries(W1)
    with tf.name_scope('biases1'):
        b1 = tf.Variable(tf.zeros([500])+0.1,name='b1')
        variable_summaries(b1)
    with tf.name_scope('wx1_plus_b1'):
        wx1_plus_b1 = tf.matmul(x,W1)+ b1
    with tf.name_scope('L1'):
        L1 = tf.tanh(wx1_plus_b1)
        L1_drop = tf.nn.dropout(L1, keep_prob)
with tf.name_scope('layer2'):
    with tf.name_scope('weights2'):
        W2 = tf.Variable(tf.truncated_normal([500, 300],stddev=0.1), name='W2')
        variable_summaries(W2)
    with tf.name_scope('biases2'):
        b2 = tf.Variable(tf.zeros([300])+0.1,name='b2')
        variable_summaries(b2)
    with tf.name_scope('wx2_plus_b2'):
        wx2_plus_b2= tf.matmul(L1_drop,W2)+ b2
    with tf.name_scope('L2'):
        L2 = tf.tanh(wx2_plus_b2)
        L2_drop = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('layer3'):
    with tf.name_scope('weights3'):
        W3 = tf.Variable(tf.truncated_normal([300, 10],stddev=0.1), name='W3')
        variable_summaries(W3)
    with tf.name_scope('biases3'):
        b3 = tf.Variable(tf.zeros([10])+0.1, name='b3')
        variable_summaries(b3)
    with tf.name_scope('wx3_plus_b3'):
        wx3_plus_b3 = tf.matmul(L2_drop, W3) + b3
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx3_plus_b3)

with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.square(y - prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_predition'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# combine all the summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001*(0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})

        writer.add_summary(summary, epoch)
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})

        print("Iter " + str(epoch)+ ". Testing Accuracy " + str(acc) + " Learning Rate " + str(learning_rate))