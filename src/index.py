import tensorflow as tf

print(tf.version)

t = tf.zeros([5, 5])
t = tf.reshape(t, [625])

print(t)
