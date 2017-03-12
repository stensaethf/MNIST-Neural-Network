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
			self.bias.append([])
			for j in range(self.layers[i]):
				self.bias[-1].append(random.random()/100)
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
			self.alpha = (1000/1000+t)
			for e in range(len(examples)):
				x = examples[e]
				y = np.zeros(self.layers[-1])
				y[labels[e]] = 1

				# feed forward
				thresholds, acts = self.feedForward(x)

				# propagate deltas backward from output layer to input layer

				# start by calculating the deltas of the output layer.
				# len(self.layers)-1 gives us the last layer
				delta = dict()
				delta[len(self.layers)-1] = (
					self.sigmaPrime(thresholds[-1]) *
					self.error(y, acts[-1])
				)

				# Propagate back through the rest of the layers.

				# first layer does not have weights, so actually only
				# len(self.layers)-1 with weights.
				# len(self.layers)-2 would give us the correct indexing.
				# however, we want to skip the last layer, so
				# len(self.layers)-2 to 1 (we write 0 because it doesn't execute it).
				for l in range(len(self.layers)-2, 0, -1):
					# we want weights going from layer l to l+1 and the
					# delta associated with the target from the weight
					# also thresholds are 0-indexed
					delta[l] = (
						self.sigmaPrime(thresholds[l-1]) *
						np.dot(self.weights[l].transpose(), delta[l+1])
					)

				# update every weight in network using deltas.
				for l in range(1, len(self.layers)-1):
					for j in range(self.layers[l]-1):
						for m in range(self.layers[l-1]-1):
							self.weights[l-1][j][m] += self.alpha * acts[l - 1][m] * delta[l][j]

			self.numberCorrect(examples, labels)

		#learning rate: \alpha(t)=1000/(1000+t) seems good

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

	network = NeuralNetwork(5, [784, 10, 10, 10])
	network.backpropogate(train_images[:200], train_labels[:200])
	# network.numberCorrect(dev[0][:1000], dev[1][:1000])

if __name__ == '__main__':
	main()
