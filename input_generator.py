import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS

def input_generator(filename):
    def _parser(serialized_example):
        columns = {
            'img_feat': tf.FixedLenFeature([2048], dtype=tf.float32),
            'unigram': tf.FixedLenFeature([FLAGS.max_len], dtype=tf.int64),
            'pid': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], dtype=tf.int64),
            'b': tf.FixedLenFeature([], dtype=tf.int64),
            'm': tf.FixedLenFeature([], dtype=tf.int64),
            's': tf.FixedLenFeature([], dtype=tf.int64),
            'd': tf.FixedLenFeature([], dtype=tf.int64),
        }

        parsed_features = tf.parse_single_example(
            serialized_example, features=columns)

        features = {
            'img_feat': parsed_features['img_feat'],
            'unigram': parsed_features['unigram'],
            'label': parsed_features['label'],
            'b': parsed_features['b'],
            'm': parsed_features['m'],
            's': parsed_features['s'],
            'd': parsed_features['d'],
        }

        return features, parsed_features['pid'], parsed_features['label']

    with tf.name_scope('TFRecord'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(_parser)
        dataset = dataset.batch(FLAGS.batch_size) # e.g. 1024
        dataset = dataset.prefetch(buffer_size=FLAGS.batch_size*4) # e.g. 4096
        iterator = dataset.make_initializable_iterator()

    return iterator
