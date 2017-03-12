# NeuralNetwork
# Neural network implementation.
# 03.11.17

import loadData, numpy as np, random

# TODO:
# 1. play around with alpha (learning rate).
# 2. play around with size of random numbers.
# 3. store weights/bias matrices.
# 4. different learning methods? stochastic gradient descent?
# 5. move reshaping into data loader or some other function?
# 6. comment.
# 7. <something />

# Random seed. Remove when done.
random.seed(50)

class NeuralNetwork:
	def __init__(self, iterations, layers):
		# initialize the neural network as a dict of dicts of weights
		self.iterations = iterations
		self.layers = layers
		self.weights = []
		# bias[layer][node]
		# we do not count the input layer, so layer = 0 is the layer after
		# the input layer.
		self.bias = []

		# weights[0] --> matrix storing weights connecting first and second
		# neuron layers.
		# weights[0][j][k] --> kth neuron in the first layer connecting to
		# 	jth neuron in the second layer.
		for i in range(1, len(self.layers)):
			self.weights.append(np.zeros((self.layers[i], self.layers[i-1])))
			self.bias.append(np.zeros((self.layers[i], 1)))
			for j in range(self.layers[i]):
				self.bias[-1][j][0] = random.random()/100
				for k in range(self.layers[i - 1]):
					self.weights[-1][j][k] = random.random()/100

	def feedForward(self, result):
		thresholds = []
		acts = [result]
		for layer in range(len(self.weights)):
			weights = self.weights[layer]
			bias = self.bias[layer]
			threshold = np.dot(weights, result) + bias
			thresholds.append(threshold)
			result = self.sigma(threshold)
			acts.append(result)
		return thresholds, acts

	def backpropagate(self, example, label):
		# compute the Del values for output units using observed error (eq. 18.8)
		# starting at output layer, repeat for each hidden layer until earliest reached
			# propagate the Del values back to previous layer
			# update the weights between the two layers
		# Del_j = g'(in_j)\sum_k{w_{j,k}Del_k}
		# w_{j,k}=w_{j,k}+\alpha*a_j*Del_k
		# 18.8: w_i=w_i+\alpha(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
		# h_w(x)=Log(w*x)=1/(1+e^{-w*x})  ---Threshold function---

		x = example
		y = label

		# feed forward
		thresholds, acts = self.feedForward(x)

		# propagate deltas backward from output layer to input layer

		# start by calculating the deltas of the output layer.
		# len(self.layers)-1 gives us the last layer
		delta = (
			self.error(y, acts[-1]) *
			self.sigmaPrime(thresholds[-1])
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
				np.dot(self.weights[-l+1].transpose(), delta) *
				self.sigmaPrime(thresholds[-l])
			)

			# once again, the bias change is just the delta, as acts
			# would be all 1.
			bias_change[-l] = delta
			weights_change[-l] = np.dot(delta, acts[-l-1].transpose())

		# update every weight in network using deltas.
		for l in range(len(self.layers)-1):
			self.weights[l] = (
				self.weights[l] + 
				self.alpha * weights_change[l]
			)
			self.bias[l] = (
				self.bias[l] + 
				self.alpha * bias_change[l]
			)

	def train(self, examples, labels):
		for t in range(self.iterations): #repeat some number of times
			# self.alpha = (1000/1000+t)
			self.alpha = 0.3

			# shuffle the examples and labels to that we do not train on
			# them in the same order every iteration.
			examples_new = []
			labels_new = []
			index_shuf = range(len(examples))
			random.shuffle(index_shuf)
			for i in index_shuf:
				examples_new.append(examples[i])
				labels_new.append(labels[i])

			examples = examples_new
			labels = labels_new

			for e in range(len(examples)):
				x = np.reshape(examples[e], (784, 1))
				y = np.zeros((self.layers[-1], 1))
				y[labels[e]] = 1.0

				self.backpropagate(x, y)

			self.numberCorrect(examples, labels)

	def error(self, label, activation):
		return (label - activation)

	def numberCorrect(self, images, labels):
		count = 0
		for i in range(len(images)):
			image = np.reshape(images[i], (784, 1))
			label = labels[i]

			# feed forward
			thresholds, acts = self.feedForward(image)
			index = np.argmax(acts[-1])

			if index == label:
				count += 1

		print str(count) + "/" + str(len(images))

	def sigma(self, x):
		return 1.0 / (1 + np.exp(-x))

	def sigmaPrime(self, x):
		return self.sigma(x)*(1.0-self.sigma(x))

def main():
	# # Loads the train, dev and test sets.
	# # 50,000, 10,000, 10,000
	train, dev, test = loadData.loadMNIST()
	# Gets the training images.
	train_images = train[0]
	# Gets the training labels.
	train_labels = train[1]

	network = NeuralNetwork(5, [784, 30, 10])
	network.train(train_images, train_labels)
	# try it on the dev set.
	network.numberCorrect(dev[0], dev[1])

if __name__ == '__main__':
	main()
