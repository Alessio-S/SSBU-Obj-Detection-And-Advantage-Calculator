import tensorflow as tf

"""THIS WORKS but requires me to use tensor flow version 1"""

for example in tf.compat.v1.python_io.tf_record_iterator("./tfrecords/Proto_Kola_test_train.tfrecord"):
    result = tf.train.Example.FromString(example)
print(result)
