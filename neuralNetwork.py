# NeuralNetwork
# Neural network implementation.
# Eric Walker, Frederik Roenn Stensaeth.
# 03.15.17

import loadData, numpy as np, random, cPickle, sys
from operator import add

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
		Feeds forward an example through the network.
		Returns the various thresholds and activations found along the way.
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
		Performs backpropagation on an example and label.
		Updates the weights and biases in the neural network by propagating
		backwards the change calculated for the output layer.
		"""
		# Find changes that needs to be made to the weights and biases.
		weights_change, bias_change = self.getWeightsChange(example, label)

		# Update every weight and bias in network with the found changes.
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
		Calcualates the changes needed to be made to each weight and bias
		given an x (example) and a y (label).
		"""
		# Feed forward the x value to find thresholds and activations.
		thresholds, acts = self.feedForward(x)

		# Propagate deltas backward from output layer to input layer.
		# Start by calculating the deltas of the output layer.
		delta = (
			self.sigmoidPrime(thresholds[-1]) *
			self.error(y, acts[-1])
		)

		# Each bias and weight has a change associated with it, so
		# let's create matrices of the same structure as we already have.
		weights_change = []
		for weights in self.weights:
			weights_change.append(np.zeros(weights.shape))

		bias_change = []
		for bias in self.bias:
			bias_change.append(np.zeros(bias.shape))

		# The change made to the weights are the dot product of the
		# delta for that layer and the activations of the previous
		# layer.
		# Activations for biases are always 1, so the change needed
		# is just the delta.
		weights_change[-1] = np.dot(delta, acts[-2].transpose())
		bias_change[-1] = delta

		# Now we want to find the changes needed to be made to the
		# rest of the weights and biases and apply them.
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
		Performs backpropagation on a batch of examples and labels.
		Updates the weights and biases only after feeding forward all the
		examples.
		"""

		batch_size = len(examples)

		total_weights_change = []
		total_bias_change = []

		# Find the weigth changes for the different examples and labels
		# in the batch.
		for i in range(batch_size):
			change = self.getWeightsChange(examples[i], labels[i])
			if i == 0:
				total_weights_change = change[0]
				total_bias_change = change[1]
			else:
				total_weights_change = map(add, total_weights_change, change[0])
				total_bias_change = map(add, total_bias_change, change[1])

		# Update weights and biases in network with the changes found.
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

def printUsage():
	print """Invalid arguments. Usage:\n
--batch\t\t\tsets mode to minibatch (default false)\n
--test\t\t\tsets the mode to run weights on the test set after training
batch-size [size]\tsets the batch size for minibatch (default 100)\n
alpha [alpha]\t\tsets the alpha (default 0.3)\n
iterations [iterations]\tsets the number of iterations to run (default 10)\n
layers [layers]\t\tdefine the number of hidden layers as the first param
\t\t\tthen list the hidden layers. e.g. layers 2 100 50 makes 2 hidden layers
\t\t\t(default 1 hidden layer with 100 neurons)\n
load-weights [filename]\tload the weights from a pickled file"""

def main():
	# Check the command line arguments
	regular = True
	test = False
	alpha = 0.3
	batch_size = 100
	iterations = 10
	layerlist = [784, 100, 10]
	filename = None
	if len(sys.argv) > 1:
		i = 1
		while i < len(sys.argv):
			if sys.argv[i] == '--batch':
				regular = False
			elif sys.argv[i] == '--test':
				test = True
			elif sys.argv[i] == 'batch-size':
				batch_size = int(sys.argv[i+1])
				i += 1
			elif sys.argv[i] == 'alpha':
				alpha = float(sys.argv[i+1])
				i += 1
			elif sys.argv[i] == 'iterations':
				iterations = int(sys.argv[i+1])
				i += 1
			elif sys.argv[i] == 'layers':
				layerlist = [784, 10]
				i += 1
				for j in range(i+1, i+1+int(sys.argv[i])):
					layerlist.insert(j-i, int(sys.argv[j]))
				i += int(sys.argv[i])
			elif sys.argv[i] == '--help':
				printUsage()
				return 0
			elif sys.argv[i] == 'load-weights':
				filename = sys.argv[i+1]
				i += 1

			else:
				printUsage()
				return -1
			i += 1

	# Loads the train, dev and test sets.
	# 50,000, 10,000, 10,000
	train, dev, test = loadData.loadMNIST()
	# Gets the training images.
	train_images = train[0]
	# Gets the training labels.
	train_labels = train[1]

	network = NeuralNetwork(iterations, layerlist)
	if filename is not None:
		network.loadWeightsAndBias(filename)
	else:
		if regular:
			network.train(train_images, train_labels, alpha)
		else:
			network.batchTrain(train_images, train_labels, alpha, batch_size)
	if test == True:
		network.numberCorrect(test[0], test[1])
	else:
		network.numberCorrect(dev[0], dev[1])

if __name__ == '__main__':
	main()
