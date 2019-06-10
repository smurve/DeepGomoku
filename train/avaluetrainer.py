import numpy as np
import tensorflow as tf
from tensorflow.feature_column import numeric_column as num
from tensorflow.estimator import RunConfig
from tensorflow.contrib.distribute import MirroredStrategy




def make_tfr_input_fn(filename_pattern, batch_size, board_size, options):
    
    N_p = board_size + 2
    feature_spec = {
        'state': tf.FixedLenFeature([N_p * N_p * 2], tf.float32),
        'advantage': tf.FixedLenFeature([N_p * N_p], tf.float32)
    }

    def _input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=filename_pattern,
            batch_size=batch_size,
            features=feature_spec,
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'],
            reader_num_threads=options['reader_num_threads'],
            parser_num_threads=options['parser_num_threads'],
            label_key='advantage')

        if options['distribute']:
            return dataset 
        else:
            return dataset.make_one_shot_iterator().get_next()
    return _input_fn


def make_model_fn(board_size, options):


    N = board_size
    
    feature_columns = [num('state', shape=((N+2)*(N+2)*2))]

    optimizers={
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=options['learning_rate']),
        "adam": tf.train.AdamOptimizer(learning_rate=options['learning_rate']),
        "adagrad": tf.train.AdagradOptimizer(learning_rate=options['learning_rate'])
    }
    
    def _model_fn(features, labels, mode):

        mask = np.ones([22, 22], dtype=int)
        mask[0] = 0
        mask[21] = 0
        mask[:,0]=0
        mask[:,21]=0
        mask = tf.constant(mask, dtype=tf.float32)
        mask = tf.expand_dims(mask,-1)
    
        from train.hypotheses import conv_2x1024_5, conv_1024_4, conv_512_3, conv_gomoku

        hypotheses_dict = {
            'conv_2x1024_5': conv_2x1024_5,
            'conv_1024_4': conv_1024_4,
            'conv_512_3': conv_512_3,
            'conv_gomoku': conv_gomoku
        }
        
        choice=options['hypothesis']
        hypothesis = hypotheses_dict[choice]
        
        out = hypothesis(N, features, feature_columns, options)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=out)

        labels = tf.expand_dims(labels, -1)
        labels = tf.reshape(labels, [-1, 22, 22, 1], name='model_reshape')
        loss = tf.losses.mean_squared_error(labels, out*mask)
        mean_error=tf.metrics.mean(tf.abs(labels-out))

        if mode == tf.estimator.ModeKeys.EVAL:    
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss = loss,
                eval_metric_ops={'mean_error': mean_error}
            )

        else:
            optimizer = optimizers[options['optimizer']]
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

            grads = optimizer.compute_gradients(loss)
            for g in grads:
                name = "%s-grad" % g[1].name
                name = name.replace(":", "_")
                tf.summary.histogram(name, g[0])
            
            return tf.estimator.EstimatorSpec(  
                mode,
                loss = loss,
                train_op = train_op)
        
    return _model_fn


def make_serving_input_fn(board_size):
    N = board_size
    def _serving_input_fn():
        placeholders = {
            'state': tf.placeholder(name='state', shape=[(N+2)*(N+2)*2, None], dtype=tf.float32)
        }
        return tf.estimator.export.ServingInputReceiver(placeholders, placeholders)

    return _serving_input_fn


def train_and_evaluate(board_size, options):

    train_input_fn = make_tfr_input_fn(options['train_data_pattern'], 
                                   options['train_batch_size'],
                                   board_size, options)    

    eval_input_fn = make_tfr_input_fn(options['eval_data_pattern'], 
                                  options['eval_batch_size'],
                                  board_size, options)

    model_fn = make_model_fn(board_size, options)

    serving_input_fn = make_serving_input_fn(board_size)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=options['max_train_steps'])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, exporters=exporter,
        steps = options['eval_steps'],
        throttle_secs=options['throttle_secs'],
        start_delay_secs=0)

    strategy = MirroredStrategy() if options['distribute'] else None
    config = RunConfig(model_dir=options['model_dir'],
                       save_summary_steps=options['save_summary_steps'],
                       train_distribute=strategy, 
                       save_checkpoints_steps=options['save_checkpoints_steps'],
                       log_step_count_steps=options['log_step_count_steps'])

    estimator = tf.estimator.Estimator(
            config=config,
            model_fn=model_fn)

    
    ##################################################################
    #   Finally, train and evaluate the model
    ##################################################################
    final_eval = tf.estimator.train_and_evaluate(
        estimator, 
        train_spec=train_spec, 
        eval_spec=eval_spec)