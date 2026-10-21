"""Day 7: TensorFlow introduction - installation check and tensors.
Install:  pip install tensorflow"""
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

scalar = tf.constant(7)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4]])
print("Scalar rank:", tf.rank(scalar).numpy(), "| Vector shape:", vector.shape,
      "| Matrix shape:", matrix.shape)

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])
print("Add:\n", (a + b).numpy())
print("Matmul:\n", tf.matmul(a, b).numpy())
print("Reshape:", tf.reshape(tf.range(6), (2, 3)).numpy())
print("Cast:", tf.cast(vector, tf.int32).numpy())

v = tf.Variable([1.0, 2.0]); v.assign_add([1.0, 1.0])
print("Variable:", v.numpy())
