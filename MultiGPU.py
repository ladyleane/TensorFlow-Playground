import os.path
import re
import time
import numpy as np
import tensorflow as tf
import cifar10

batch_size = 128
max_steps = 1000000
num_gpus = 4

def tower_loss(scope):
	images, labels = cifar10.distorted_inputs()
	logits = cifar10.inference(images)
	_ = cifar10.loss(logits, labels)
	losses = tf.get_collection('losses', scope)
	total_loss = tf.add_n(losses, name = 'total_loss')
	return total_loss

def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)

		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)

		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_gradsappend(grad_and_var)
	return average_grads

def train():
	with tf.Graph().as_default(), tf.device('/cpu: 0'):
		global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)
		num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
		decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
		lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE, global_step, decay_steps, cifar10.INITIAL_LEARNING_RATE_DECAY_FACTOR, staircase = True)
		opt = tf.train.GradientDescentOptimzer(lr)

		tower_grads = []
		for i in range(num_gpus):
			with tf.device('/gpu: %d' % i):
				with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
					loss = tower_loss(scope)
					tf.get_variable_scope().reuse_variables()
					grads = opt.compute_gradients(loss)
					tower_grads.append(grads)

		grads = average_gradients(tower_grads)
		apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)

		saver = tf.train.Saver(tf.all_variables())
		init = tf.global_variables_initializer()
		sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
		sess.run(init)
		tf.train.start_queue_runners(sess = sess)

		for step in range(max_steps):
			start_time = time.time()
			_, loss_value = sess.run([apply_gradient_op, loss])
			duration = time.time() - start_time

			if step % 10 == 0:
				num_examples_per_step = batch_size * num_gpus
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration / num_gpus

				format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
				print (format_str % (step, loss_value, examples_per_sec, sec_per_batch))
			if step % 1000 == 0 or (step + 1) == max_steps:
				saver.save(sess, '/tmp/cifar10_train/model.ckpt', global_step = step)

cifar10.maybe_download_and_extract()
train()

