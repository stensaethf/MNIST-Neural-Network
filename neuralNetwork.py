# NeuralNetwork
# Neural network implementation.
# 03.11.17

import loadData, numpy as np, random

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
			# self.bias.append(np.zeros(self.layers[i]))
			self.bias.append(np.zeros((self.layers[i])))
			for j in range(self.layers[i]):
				self.bias[-1][j] = random.random()/100
				for k in range(self.layers[i - 1]):
					# self.bias[-1][j][k] = random.random()/100
					self.weights[-1][j][k] = random.random()/100

	def feedForward(self, result):
		thresholds = []
		acts = [result]
		for layer in range(len(self.layers)-1):
			weights = self.weights[layer]
			bias = self.bias[layer]
			threshold = np.dot(weights, result) + bias
			thresholds.append(threshold)
			result = self.sigma(threshold)
			acts.append(result)
		return thresholds, acts

	def backpropogate(self, examples, labels):
		# compute the Del values for output units using observed error (eq. 18.8)
		# starting at output layer, repeat for each hidden layer until earliest reached
			# propagate the Del values back to previous layer
			# update the weights between the two layers
		# Del_j = g'(in_j)\sum_k{w_{j,k}Del_k}
		# w_{j,k}=w_{j,k}+\alpha*a_j*Del_k
		# 18.8: w_i=w_i+\alpha(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
		# h_w(x)=Log(w*x)=1/(1+e^{-w*x})  ---Threshold function---

		for t in range(self.iterations): #repeat some number of times
			# self.alpha = (1000/1000+t)
			self.alpha = 0.3
			for e in range(len(examples)):
				x = examples[e]
				y = np.zeros(self.layers[-1])
				y[labels[e]] = 1

				# feed forward
				thresholds, acts = self.feedForward(x)

				# propagate deltas backward from output layer to input layer

				# start by calculating the deltas of the output layer.
				# len(self.layers)-1 gives us the last layer
				delta = (
					self.sigmaPrime(thresholds[-1]) *
					self.error(y, acts[-1])
				)

				# Propagate back through the rest of the layers.

				# first layer does not have weights, so actually only
				# len(self.layers)-1 with weights.
				# len(self.layers)-2 would give us the correct indexing.
				# however, we want to skip the last layer, so
				# len(self.layers)-2 to 1 (we write 0 because it doesn't execute it).

				# each bias and weight has a change associated with it, so
				# let's create matrices of the same structure as we already have.
				weights_change = [np.zeros(weights.shape) for weights in self.weights]
				bias_change = [np.zeros(bias.shape) for bias in self.bias]

				# the change made to the weights are the dot product of the
				# delta for that layer and the activations of the previous
				# layer.
				# activations for biases are always 1, so the change needed
				# is just the delta.
				# acts[-2]
				weights_change[-1] = np.dot(delta, acts[-2].transpose())
				bias_change[-1] = delta

				# now we want to find the changes needed to be made to the
				# rest of the weights and biases.
				# want: delta_i = sigmaPrime(threshold_i) * dot(weight_i, delta_j)
				# where delta_j is the next layer.
				for l in range(2, len(self.layers)):
					delta = (
						self.sigmaPrime(thresholds[-l]) *
						np.dot(self.weights[-l+1].transpose(), delta)
					)

					# once again, the bias change is just the delta, as acts
					# would be all 1.
					bias_change[-l] = delta
					weights_change[-l] = np.dot(delta, acts[-l].transpose())

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

			self.numberCorrect(examples, labels)

	def error(self, label, activation):
		return (label - activation)

	def numberCorrect(self, images, labels):
		count = 0
		for i in range(len(images)):
			image = images[i]
			label = labels[i]
			# feed forward
			thresholds, acts = self.feedForward(image)
			index = np.argmax(acts[-1])

			if index == label:
				count += 1

		print str(count) + "/" + str(len(images))

	def sigma(self, x):
		# return 1.0 / (1 + Math.exp(-x))
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

	network = NeuralNetwork(5, [784, 10, 10])
	network.backpropogate(train_images, train_labels)
	# network.numberCorrect(dev[0][:1000], dev[1][:1000])

if __name__ == '__main__':
	main()
