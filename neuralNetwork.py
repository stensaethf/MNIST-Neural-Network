# NeuralNetwork
# Neural network implementation.
# 03.11.17

import loadData, numpy as np, random

class NeuralNetwork:
	def __init__(self):
		# initialize the neural network as a dict of dicts of weights
		self.layers = [784, 10]
		# self.layers = [1, 1, 1]
		self.weights = dict()
		# bias[layer][node]
		self.bias = dict()
		self.alpha = 0.1
		for i in range(1, len(self.layers)):
			self.weights[i] = []
			self.bias[i] = []
			for j in range(self.layers[i]):
				self.bias[i].append(1)
				self.weights[i].append([])
				for k in range(self.layers[i-1]):
					# self.weights[layer][node1][node2] where node1 is in layer and node2 in in layer-1
					self.weights[i][j].append(1)

	def feedForward(self, result):
		# weights: [layer][node]
		output = [result]
		ins = []
		for layer in self.weights:
			output.append([])
			ins.append([])
			for node_index in range(len(self.weights[layer])):
				dot_product = np.dot(self.weights[layer][node_index], result)
				bias = self.bias[layer][node_index]
				ins[-1].append(dot_product + bias)
				output[-1].append(self.sigma(dot_product + bias))
			result = output[-1]

		return output, ins

	def backpropogate(self, examples, labels):
		# compute the Del values for output units using observed error (eq. 18.8)
		# starting at output layer, repeat for each hidden layer until earliest reached
			# propagate the Del values back to previous layer
			# update the weights between the two layers
		# Del_j = g'(in_j)\sum_k{w_{j,k}Del_k}
		# w_{j,k}=w_{j,k}+\alpha*a_j*Del_k
		#18.8: w_i=w_i+\alpha(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
		#h_w(x)=Log(w*x)=1/(1+e^{-w*x})  ---Threshold function---

		"""
		for i in range(len(network)):
			for j in range(len(network[i])):
				network[i][j] = random.random() / 10.0  # small random number"""

		for k in range(50): #repeat some number of times
			for i in range(len(examples)):
				x = examples[i]
				y = labels[i]

				# feed forward
				# a, ins = [[]]
				a, ins = self.feedForward(x)

				Del = []

				# propagate deltas backward from output layer to input layer

				# start by calculating the Dels
				for j in range(self.layers[-1]):
					Del.append(self.sigmaPrime(ins[-1][j])*(y-a[-1][j]))

				# propagate back through the rest of the layers
				for l in range(len(self.layers)-1, 0, -1):
					for j in range(len(self.weights[l])):
						Del[j] = self.sigmaPrime(ins[l-1][j])*sum(self.weights[l][j][m]*Del[j] for m in range(self.layers[l-1]))
						#Del[j] = self.sigmaPrime(ins[j])*np.dot(self.weights[l][j], Del)

				# update every weight in network using dels
				for l in range(1, len(self.layers)):
					for j in range(self.layers[l]):
						for m in range(len(self.weights[l][j])):
							# print "l-1"
							# print a[m]
							# print a[l-1]
							# print m
							# print a[l-1][m]
							# print "DEL"
							# print Del[j]
							self.weights[l][j][m] = self.weights[l][j][m] + self.alpha*a[l-1][m]*Del[j]

			self.numberCorrect(examples, labels)

		#learning rate: \alpha(t)=1000/(1000+t) seems good

	def numberCorrect(self, images, labels):
		count = 0
		for i in range(len(images)):
			image = images[i]
			label = labels[i]
			# feed forward
			a, ins = self.feedForward(image)
			index = a.index(max(a))

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

	network = NeuralNetwork()
	network.backpropogate(train_images, train_labels)
	network.numberCorrect(dev[0], dev[1])

if __name__ == '__main__':
	main()
