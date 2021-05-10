import tensorflow as tf

tf.compat.v1.disable_eager_execution() 
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)

with tf.compat.v1.Session() as sess:
    output = sess.run(result)
    print(output)
