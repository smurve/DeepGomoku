import numpy as np
import tensorflow as tf
from tensorflow.feature_column import numeric_column
from tensorflow.feature_column import crossed_column
from tensorflow.feature_column import indicator_column
from tensorflow.feature_column import categorical_column_with_identity
from tensorflow_transform.tf_metadata import dataset_schema

#
# The input function: See InputFunctions.ipynb for explanations 
#
feature_spec = {
    'state': tf.io.FixedLenFeature([21, 21, 2], tf.float32),
    'distr': tf.io.FixedLenFeature([21, 21, 1], tf.float32)
}
schema = dataset_schema.from_feature_spec(feature_spec)

def make_tfr_input_fn(filename_pattern, batch_size, options):
    
    def _input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=filename_pattern,
            batch_size=batch_size,
            features=feature_spec,
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'],
            reader_num_threads=options['reader_num_threads'],
            parser_num_threads=options['parser_num_threads'],
            sloppy_ordering=options['sloppy_ordering'],
            num_epochs=options['num_epochs'],
            label_key='distr')

        if options['distribute']:
            return dataset 
        else:
            return dataset.make_one_shot_iterator().get_next()
    return _input_fn

