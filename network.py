from percolation import Grid
import tensorflow as tf 
import numpy as np 
print "Finished importing modules."


# Constants and Parameters

GRID_SIZE = 20
FIXED_PERCOLATION_COUNT = True

LEARN_RATE = 0.1
BATCH_SIZE = 128
TRAIN_ITERATIONS = 100

n_input = GRID_SIZE * GRID_SIZE
n_hidden = 256
n_output = 1


# Pre-processing Input

def generate_batch(batch_size = BATCH_SIZE):
	grids = [Grid(GRID_SIZE, FIXED_PERCOLATION_COUNT) for i in range(batch_size)]
	X = np.array([g.grid.reshape((-1,)) for g in grids])
	Y = np.array([g.hasPath for g in grids]).astype(np.float32)
	print X.shape
	print Y.shape
	print Y
	return X, Y


# Network Architecture

trainInputs = tf.placeholder(tf.float32, shape=(None, n_input))
trainLabels = tf.placeholder(tf.float32, shape=(None, n_output))

# Train inputs: flattened batch_size x n^2 array
# Train labels: a batch_size x 1 array which is either a 1 (connected) or 0 (not)


weights = {
	"h1": tf.Variable(tf.random_normal([n_input, n_hidden])),
	"out": tf.Variable(tf.random_normal([n_hidden, n_output]))
}

biases = {
	"h1": tf.Variable(tf.random_normal([n_hidden])),
	"out": tf.Variable(tf.random_normal([n_output]))
}

def shallow_model(x, weights, biases):
	h1 = tf.matmul(x, weights["h1"]) + biases["h1"]
	h1 = tf.nn.relu(h1)

	out = tf.matmul(h1, weights["out"]) + biases["out"]
	out = tf.sigmoid(out)
	return out

def compute_accuracy(predicted, actual):
	# Inputs: batch_size x 1
	predicted = tf.round(predicted)
	equality = tf.equal(predicted, actual)
	return tf.reduce_mean(tf.cast(equality, tf.float32))

def loss_optim_accuracy(model, x, y):
	pred = model(x, weights, biases)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)
	accuracy = compute_accuracy(pred, y)

	return loss, optimizer, accuracy

loss, optimizer, trainAcc = loss_optim_accuracy(shallow_model, trainInputs, trainLabels)


generate_batch(7)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	print "Initialized variables."

	for epoch in range(TRAIN_ITERATIONS):
		_, l, acc_train = sess.run([optimizer, loss, trainAcc])
		print "Train Acc:", trainAcc










