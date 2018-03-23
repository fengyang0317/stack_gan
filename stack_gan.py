from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.gan.python.eval.python import eval_utils

tfgan = tf.contrib.gan

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string('job_dir', 'saving', 'job dir')

tf.flags.DEFINE_integer('ef_dim', 128, 'sentence emb dim')

tf.flags.DEFINE_integer('gf_dim', 128, 'generator dim')

tf.flags.DEFINE_integer('df_dim', 64, 'dis dim')

tf.flags.DEFINE_integer('z_dim', 100, 'z dim')

tf.flags.DEFINE_float('w_kl', 2, 'w kl')

tf.flags.DEFINE_integer('batch_size', 64, 'batch size')

tf.flags.DEFINE_integer('crop_size', 64, 'crop size')

tf.flags.DEFINE_integer('max_steps', 80000, 'max steps')

tf.flags.DEFINE_integer('sample_num', 4, 'context sample num')

tf.flags.DEFINE_float('weight_decay', 0, ' weight decay')

tf.flags.DEFINE_string('data_dir', '/home/yfeng23/lab/StackGAN/Data/birds/CUB_200_2011/', 'data')

tf.flags.DEFINE_integer('save_checkpoint_secs', 1800, 'save interval')

tf.flags.DEFINE_float('generator_lr', 2e-4, 'generator lr')

tf.flags.DEFINE_float('discriminator_lr', 2e-4, 'discriminator lr')

tf.flags.DEFINE_integer('decay_steps', 7000, 'decay step')

FLAGS = tf.flags.FLAGS


def generator_fn(inputs, weight_decay=1e-5, is_training=True):
  noize, sentence, _ = inputs
  sentence = slim.fully_connected(sentence, FLAGS.ef_dim * 2, activation_fn=tf.nn.leaky_relu,
                                  weights_initializer=tf.random_normal_initializer(stddev=0.02))
  mu, log_sigma = tf.split(sentence, 2, axis=1)
  eps = tf.truncated_normal(mu.shape)
  stddev = tf.exp(log_sigma)
  c = mu + stddev * eps
  net = tf.concat([noize, c], axis=1)

  with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, is_training=is_training):
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        biases_initializer=None):
      net10 = slim.fully_connected(net, 4 * 4 * FLAGS.gf_dim * 8, activation_fn=None, scope='net10',
                                   weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                   normalizer_fn=slim.batch_norm)
      net10 = tf.reshape(net10, [-1, 4, 4, FLAGS.gf_dim * 8])
      net11 = slim.conv2d(net10, FLAGS.gf_dim * 2, 1, scope='net11_1')
      net11 = slim.conv2d(net11, FLAGS.gf_dim * 2, 3, scope='net11_2')
      net11 = slim.conv2d(net11, FLAGS.gf_dim * 8, 3, activation_fn=None, scope='net11_3')
      net1 = tf.nn.relu(net10 + net11)
      net20 = tf.image.resize_nearest_neighbor(net1, [8, 8])
      net20 = slim.conv2d(net20, FLAGS.gf_dim * 4, 3, activation_fn=None, scope='net20')
      net21 = slim.conv2d(net20, FLAGS.gf_dim, 1, scope='net21_1')
      net21 = slim.conv2d(net21, FLAGS.gf_dim, 3, scope='net21_2')
      net21 = slim.conv2d(net21, FLAGS.gf_dim * 4, 3, activation_fn=None, scope='net21_3')
      net2 = tf.nn.relu(net20 + net21)
      net = tf.image.resize_nearest_neighbor(net2, [16, 16])
      net = slim.conv2d(net, FLAGS.gf_dim * 2, 3, scope='net3_1')
      net = tf.image.resize_nearest_neighbor(net, [32, 32])
      net = slim.conv2d(net, FLAGS.gf_dim, 3, scope='net3_2')
      net = tf.image.resize_nearest_neighbor(net, [64, 64])
      net = slim.conv2d(net, 3, 3, activation_fn=tf.tanh, normalizer_fn=None, scope='output')
      return [net, mu, log_sigma]


# real image consists of correct and wrong.
def discriminator_fn(inputs, generator_inputs, weight_decay=1e-5, is_training=True):
  if type(inputs) == list:
    inputs = inputs[0]
    tile_times = 1
  else:
    inputs = tf.concat([inputs, generator_inputs[2]], axis=0)
    tile_times = 2
  sentence = generator_inputs[1]
  with slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=1e-5, is_training=is_training):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        biases_initializer=None):
      net10 = slim.conv2d(inputs, FLAGS.df_dim, 4, stride=2, normalizer_fn=None, scope='net10_1')
      net10 = slim.conv2d(net10, FLAGS.df_dim * 2, 4, stride=2, scope='net10_2')
      net10 = slim.conv2d(net10, FLAGS.df_dim * 4, 4, stride=2, activation_fn=None, scope='net10_3')
      net10 = slim.conv2d(net10, FLAGS.df_dim * 8, 4, stride=2, activation_fn=None, scope='net10_4')
      net11 = slim.conv2d(net10, FLAGS.df_dim * 2, 1, scope='net11_1')
      net11 = slim.conv2d(net11, FLAGS.df_dim * 2, 3, scope='net11_2')
      net11 = slim.conv2d(net11, FLAGS.df_dim * 8, 3, scope='net11_3')
      net1 = tf.nn.leaky_relu(net10 + net11)

      context = slim.fully_connected(sentence, FLAGS.ef_dim, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.random_normal_initializer(stddev=0.02))
      context = tf.expand_dims(tf.expand_dims(context, 1), 1)
      context = tf.tile(context, [tile_times, 4, 4, 1])
      net = tf.concat([net1, context], axis=3)

      net = slim.conv2d(net, FLAGS.df_dim * 8, 1, scope='net2')
      net = slim.conv2d(net, 1, 4, padding='VALID', activation_fn=None, normalizer_fn=None, scope='output')
      net = tf.squeeze(net, [1, 2])
      return net


def generator_loss(gan_model, add_summaries=False):
  loss = tfgan.losses.modified_generator_loss(gan_model, add_summaries=add_summaries)
  _, mu, log_sigma = gan_model.generated_data
  kl_loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
  kl_loss = tf.reduce_mean(kl_loss) * FLAGS.w_kl
  loss += kl_loss
  if add_summaries:
    tf.summary.scalar('kl_loss', kl_loss)
  return loss


def discriminator_loss(gan_model, add_summaries=False):
  with tf.name_scope('discriminator_minmax_loss'):
    correct, wrong = tf.split(gan_model.discriminator_real_outputs, 2)
    loss_on_real = tf.losses.sigmoid_cross_entropy(
      tf.ones_like(correct),
      correct)
    loss_on_wrong = tf.losses.sigmoid_cross_entropy(
      tf.zeros_like(wrong),
      wrong)
    loss_on_generated = tf.losses.sigmoid_cross_entropy(
      tf.zeros_like(gan_model.discriminator_gen_outputs),
      gan_model.discriminator_gen_outputs)
    loss = loss_on_real + (loss_on_wrong + loss_on_generated) / 2
    if add_summaries:
      tf.summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
      tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
      tf.summary.scalar('discriminator_wrong_minimax_loss', loss_on_wrong)
      tf.summary.scalar('discriminator_minimax_loss', loss)
  return loss


def split(line):
  sp = tf.string_split([line], delimiter=',')
  name = sp.values[0]
  label = sp.values[1]
  label = tf.string_to_number(label, out_type=tf.int32)
  return name, label


def read_img(t1, t2, data_dir):
  def imread(im_path):
    img = tf.read_file(tf.string_join([data_dir, 'lr_imgs/', im_path, '.png']))
    img = tf.image.decode_png(img, 3)
    img = tf.random_crop(img, [FLAGS.crop_size, FLAGS.crop_size, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img * 2 - 1
    return img
  correct = imread(t1[0])
  wrong = imread(t2[0])
  return correct, wrong, t1[0], t1[1]


def read_sen(correct, wrong, n1, l1, data_dir):
  context = np.load(data_dir + 'sentence/' + n1 + '.npy')
  idx = np.random.choice(context.shape[0], FLAGS.sample_num, replace=False)
  context = np.mean(context[idx], axis=0)
  return correct, wrong, context, l1


def resize_img(correct, wrong, context, label):
  correct.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
  wrong.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
  context.set_shape([FLAGS.batch_size, 1024])
  label.set_shape(FLAGS.batch_size)
  return correct, wrong, context, label


def train_input_fn(data_dir, subset='train'):
  dataset = tf.data.TextLineDataset(tf.string_join([data_dir, 'my', subset, '.txt']))
  dataset = dataset.repeat()
  dataset = dataset.shuffle(2000)
  dataset = dataset.map(split)
  dataset = tf.data.Dataset.zip((dataset, dataset))
  dataset = dataset.filter(lambda t1, t2: tf.not_equal(t1[1], t2[1]))
  dataset = dataset.map(functools.partial(read_img, data_dir=data_dir), num_parallel_calls=8)
  dataset = dataset.map(lambda c, w, n, l: tuple(tf.py_func(read_sen, [c, w, n, l, data_dir], [tf.float32, tf.float32, tf.float32, tf.int32])),
                        num_parallel_calls=8)
  dataset = dataset.batch(FLAGS.batch_size)
  dataset = dataset.map(resize_img)
  dataset = dataset.prefetch(4)
  iterator = dataset.make_one_shot_iterator()
  correct, wrong, context, targets = iterator.get_next()
  return correct, wrong, context, targets


def my_summary_image(name, tensor, grid_size=2):
  num_images = grid_size ** 2
  inp_image_shape = tensor.shape.as_list()[1:3]
  inp_channels = tensor.shape.as_list()[3]
  tensor = (tensor + 1) / 2
  tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
  tf.summary.image(
    name,
    eval_utils.image_grid(
      tensor[:num_images],
      grid_shape=(grid_size, grid_size),
      image_shape=inp_image_shape,
      num_channels=inp_channels),
    max_outputs=1)


def main(_):
  if not tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.MakeDirs(FLAGS.job_dir)
  sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

  with tf.name_scope('inputs'):
    with tf.device('/cpu:0'):
      correct, wrong, context, label = train_input_fn(data_dir=FLAGS.data_dir)
  gan_model = tfgan.gan_model(
    generator_fn=functools.partial(generator_fn, weight_decay=FLAGS.weight_decay),
    discriminator_fn=functools.partial(discriminator_fn,
                                       weight_decay=FLAGS.weight_decay),
    real_data=correct,
    generator_inputs=(tf.random_normal([FLAGS.batch_size, FLAGS.z_dim]), context, wrong),
    check_shapes=False)
  my_summary_image('1_input', gan_model.real_data)
  my_summary_image('2_fake', gan_model.generated_data[0])
  my_summary_image('3_wrong', gan_model.generator_inputs[2])

  with tf.name_scope('loss'):
    g_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=generator_loss,
      discriminator_loss_fn=discriminator_loss,
      add_summaries=True)
    tfgan.eval.add_regularization_loss_summaries(gan_model)

  global_step = tf.train.get_or_create_global_step()
  generator_lr = tf.train.exponential_decay(FLAGS.generator_lr, global_step, FLAGS.decay_steps, 0.5, staircase=True)
  discriminator_lr = tf.train.exponential_decay(FLAGS.discriminator_lr, global_step, FLAGS.decay_steps, 0.5, staircase=True)
  tf.summary.scalar('learning_rate/generator', generator_lr)
  tf.summary.scalar('learning_rate/discriminator', discriminator_lr)
  with tf.name_scope('train'):
    train_ops = tfgan.gan_train_ops(
      gan_model,
      g_loss,
      generator_optimizer=tf.train.AdamOptimizer(generator_lr, 0.5),
      discriminator_optimizer=tf.train.AdamOptimizer(discriminator_lr, 0.5),
      summarize_gradients=True)

  status_message = tf.string_join(
    ['Starting train step: ',
     tf.as_string(tf.train.get_or_create_global_step())],
    name='status_message')
  if FLAGS.max_steps == 0:
    return
  tfgan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_steps),
           tf.train.LoggingTensorHook([status_message], every_n_iter=100)],
    logdir=FLAGS.job_dir,
    get_hooks_fn=tfgan.get_joint_train_hooks(),
    config=sess_config,
    save_checkpoint_secs=FLAGS.save_checkpoint_secs)


if __name__ == '__main__':
  tf.app.run()
