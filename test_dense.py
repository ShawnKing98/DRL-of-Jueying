import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=(4,))

y = tf.layers.dense(inputs=x, units=5, activation=tf.tanh)

writer = tf.summary.FileWriter('tmp_board', tf.get_default_graph())
writer.close()