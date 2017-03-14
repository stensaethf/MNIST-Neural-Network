# NeuralNetwork
# Neural network implementation.
# 03.15.17

import loadData, numpy as np, random, cPickle
from operator import add

# TODO:
# 1. comment.
# 2. README file

# Random seed. Remove when done.
random.seed(50)

class NeuralNetwork:
	def __init__(self, iterations, layers):
		"""
        Constructor for the neural network.
        Sets up the weights and biases.
		
		@params iterations - number of iterations to be run.
        @params layers - list of the size of the layers in the network.
        	Example: [5, 10] - input and output layers have 5 and 10
        	neurons respectively.
        @return n/a.
        """
		self.iterations = iterations
		self.layers = layers

		# Sets up the matrices used for storing the weights and biases.
		# Example: weights[0] is the matrix storing weights connecting the
		# 	first and second layers of neurons.
		# Example: weights[0][j][k] is the weight connecting the kth neuron
		# 	in the first layer to the jth neuron in the second layer.
		# The initial weights and biases are random in the range: [-0.5, 0.5].
		self.weights = []
		self.bias = []
		# The first layer has no connections entering it, so we skip it.
		for i in range(1, len(self.layers)):
			self.weights.append(np.zeros((self.layers[i], self.layers[i-1])))
			self.bias.append(np.zeros((self.layers[i], 1)))
			for j in range(self.layers[i]):
				self.bias[-1][j][0] = random.random() - 0.5
				for k in range(self.layers[i - 1]):
					self.weights[-1][j][k] = random.random() - 0.5

	def feedForward(self, result):
		"""
		xx

		@params result - xx
		@return thresholds - xx
		@return acts - xx
		"""
		thresholds = []
		acts = [result]
		for layer in range(len(self.weights)):
			weights = self.weights[layer]
			bias = self.bias[layer]
			threshold = np.dot(weights, result) + bias
			thresholds.append(threshold)
			result = self.sigmoid(threshold)
			acts.append(result)
		return thresholds, acts

	def backpropagate(self, example, label):
		"""
		xx

		@params example - xx
		@params label - xx
		@return n/a.
		"""
		# compute the Del values for output units using observed error (eq. 18.8)
		# starting at output layer, repeat for each hidden layer until earliest reached
			# propagate the Del values back to previous layer
			# update the weights between the two layers
		# Del_j = g'(in_j)\sum_k{w_{j,k}Del_k}
		# w_{j,k}=w_{j,k}+\alpha*a_j*Del_k
		# 18.8: w_i=w_i+\alpha(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
		# h_w(x)=Log(w*x)=1/(1+e^{-w*x})  ---Threshold function---

		weights_change, bias_change = self.getWeightsChange(example, label)

		# update every weight and bias in network.
		for l in range(len(self.layers)-1):
			self.weights[l] = (
				self.weights[l] + 
				self.alpha * weights_change[l]
			)
			self.bias[l] = (
				self.bias[l] + 
				self.alpha * bias_change[l]
			)

	def getWeightsChange(self, x, y):
		"""
		xx

		@params x - xx
		@params y - xx
		@return weights_change - xx
		@return bias_change - xx
		"""
		# feed forward
		thresholds, acts = self.feedForward(x)

		# propagate deltas backward from output layer to input layer

		# start by calculating the deltas of the output layer.
		# len(self.layers)-1 gives us the last layer
		delta = (
			self.sigmoidPrime(thresholds[-1]) *
			self.error(y, acts[-1])
		)

		# Propagate back through the rest of the layers.

		# each bias and weight has a change associated with it, so
		# let's create matrices of the same structure as we already have.
		weights_change = []
		for weights in self.weights:
			weights_change.append(np.zeros(weights.shape))

		bias_change = []
		for bias in self.bias:
			bias_change.append(np.zeros(bias.shape))

		# the change made to the weights are the dot product of the
		# delta for that layer and the activations of the previous
		# layer.
		# activations for biases are always 1, so the change needed
		# is just the delta.
		weights_change[-1] = np.dot(delta, acts[-2].transpose())
		bias_change[-1] = delta

		# now we want to find the changes needed to be made to the
		# rest of the weights and biases.
		for l in range(2, len(self.layers)):
			delta = (
				self.sigmoidPrime(thresholds[-l]) *
				np.dot(self.weights[-l + 1].transpose(), delta)
			)

			bias_change[-l] = delta
			weights_change[-l] = np.dot(delta, acts[-l - 1].transpose())

		return weights_change, bias_change

	def batchPropagate(self, examples, labels):
		"""
		xx

		@params examples - xx
		@params labels - xx
		@return n/a.
		"""

		""" Does backpropagation in a batch, updating only after feeding forward all the examples """
		""" Similar to backpropagation, but it only updates weights
		after looking at all of the examples """

		batch_size = len(examples)

		total_weights_change = []
		total_bias_change = []

		for i in range(batch_size):
			change = self.getWeightsChange(examples[i], labels[i])
			if i == 0:
				total_weights_change = change[0]
				total_bias_change = change[1]
			else:
				total_weights_change = map(add, total_weights_change, change[0])
				total_bias_change = map(add, total_bias_change, change[1])

		# update every weight and bias in network.
		for l in range(len(self.layers)-1):
			self.weights[l] = (
				self.weights[l] +
				self.alpha * total_weights_change[l] / batch_size
			)
			self.bias[l] = (
				self.bias[l] +
				self.alpha * total_bias_change[l] / batch_size
			)

	def train(self, examples, labels, alpha):
		"""
		Trains the weights and biases of the neural network using examples
		and labels.
		"""
		for t in range(self.iterations):
			# Calculate decaying alpha.
			self.alpha = alpha - (alpha*t/self.iterations)

			# Shuffle the data so that we see the data in different
			# orders while training.
			examples, labels = self.doubleShuffle(examples, labels)

			for e in range(len(examples)):
				# Backpropogate the example and label.
				self.backpropagate(examples[e], labels[e])

			# Check how many examples we classify correctly.
			self.numberCorrect(examples, labels)

	def batchTrain(self, examples, labels, alpha, batch_size):
		"""
		Trains the weights and biases of the neural network using examples
		and labels with a provided batch size.
		"""
		for t in range(self.iterations):
			# Calculate decaying alpha.
			self.alpha = alpha - (alpha*t/self.iterations)

			# Shuffle the data so that we get different batches each
			# iteration.
			examples, labels = self.doubleShuffle(examples, labels)
			maxSize = len(examples)
			for e in range(0, maxSize, batch_size):
				num_in_batch = batch_size
				if e + num_in_batch >= maxSize:
					num_in_batch = maxSize - 1 - e
				if num_in_batch == 0:
					continue
				exs = examples[e:e+num_in_batch]
				labs = labels[e:e+num_in_batch]

				# Backpropogate the batch.
				self.batchPropagate(exs, labs)

			# Check how many examples we classify correctly.
			self.numberCorrect(examples, labels)

	def error(self, label, activation):
		"""
		Calculates the difference between a label and an activation.
		"""
		return (label - activation)

	def numberCorrect(self, images, labels):
		"""
		Classifies a given set of images and labels using the trained
		weights and biases. Prints the number of correct classifications.
		"""
		count = 0
		for i in range(len(images)):
			image = images[i]
			label = labels[i]

			# Feed forward the image.
			thresholds, acts = self.feedForward(image)
			index = np.argmax(acts[-1])
			label_index = np.argmax(label)

			# Check whether the classification was correct.
			if index == label_index:
				count += 1

		print str(count) + "/" + str(len(images))

	def doubleShuffle(self, list1, list2):
		"""
		Shuffles two corresponding lists of equal length.
		"""
		list1_new = []
		list2_new = []
		index_shuf = range(len(list1))
		# Randomly shuffle the input
		random.shuffle(index_shuf)
		for i in index_shuf:
			list1_new.append(list1[i])
			list2_new.append(list2[i])

		return list1_new, list2_new

	def sigmoid(self, x):
		"""
		Calculates the sigmoid value for a given x.
		"""
		return 1.0 / (1 + np.exp(-x))

	def sigmoidPrime(self, x):
		"""
		Calculates the sigmoid prime value for a given x.
		"""
		return self.sigmoid(x)*(1.0-self.sigmoid(x))

	def saveWeightsAndBias(self, filename):
		"""
		Pickles (saves) the weights and biases.
		"""
		f = open(filename, "w")
		cPickle.dump({
			'weights' : self.weights,
			'bias' : self.bias
		}, f)
		f.close()

	def loadWeightsAndBias(self, filename):
		"""
		Loads saved weights and biases.
		"""
		f = open(filename, "r")
		weights_bias = cPickle.load(f)
		f.close()
		weights = weights_bias['weights']
		bias = weights_bias['bias']

		self.weights = weights
		self.bias = bias

def main():
	# Loads the train, dev and test sets.
	# 50,000, 10,000, 10,000
	train, dev, test = loadData.loadMNIST()
	# Gets the training images.
	train_images = train[0]
	# Gets the training labels.
	train_labels = train[1]

	for i in range(1, 10, 2):
		network = NeuralNetwork(10, [784, 397, 10])
		# network.train(train_images, train_labels, 0.1)
		network.batchTrain(train_images, train_labels, i/10.0, 100)
		# try it on the dev set.
		network.numberCorrect(dev[0], dev[1])

if __name__ == '__main__':
	main()
