# NeuralNetwork
# Neural network implementation.
# 03.11.17

import loadData, numpy as np, random

class NeuralNetwork:
	def __init__(self):
		# initialize the neural network as a dict of dicts of weights
		# self.layerlens = [784, 10]
		self.layers = [1, 1, 1]
		self.weights = dict()
		# bias[layer][node]
		self.bias = dict()
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
		for layer in self.weights:
			output = []
			ins = []
			for node_index in range(len(self.weights[layer])):
				dot_product = np.dot(self.weights[layer][node_index], result)
				bias = self.bias[layer][node_index]
				ins.append(dot_product + bias)
				output.append(self.sigma(dot_product + bias))
			result = output

		return result, ins

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
				a, ins = self.feedForward(x)

				Del = []

				# propagate deltas backward from output layer to input layer

				# start by calculating the Dels
				for j in range(self.layers[-1]):
					Del[j] = self.sigmaPrime(ins[j])*(y[j]-a[j])

				# propagate back through the rest of the layers
				for l in range(len(self.layers)-1, 0, -1):
					for j in self.weights[l]:
						Del[j] = self.sigmaPrime(ins[j])*sum(self.weights[l][m][j]*Del[m] for m in range(self.layers[l-1]))
						#Del[j] = self.sigmaPrime(ins[j])*np.dot(self.weights[l][j], Del)

				# update every weight in network using deltas
				for i in range(len(network)):
					for j in range(len(network[i])):
						weights[i][j] = weights[i][j] + self.alpha*a-[i]*Del[j]

		return network
		#learning rate: \alpha(t)=1000/(1000+t) seems good

		# filler, remove later
		x = 1

	def sigma(self, x):
		# return 1.0 / (1 + Math.exp(-x))
		return 1.0 / (1 + np.exp(-x))

	def sigmaPrime(self, x):
		return self.sigma(x)*(1.0-self.sigma(x))

def main():
	# # Loads the train, dev and test sets.
	# # 50,000, 10,000, 10,000
	# train, dev, test = loadData.loadMNIST()
	# # Gets the training images.
	# train_images = train[0]
	# # Gets the training labels.
	# train_labels = train[1]

	# print len(train_images)
	# print len(train_labels)

	net = NeuralNetwork()
	result = net.feedForward([0])
	print result

if __name__ == '__main__':
	main()
