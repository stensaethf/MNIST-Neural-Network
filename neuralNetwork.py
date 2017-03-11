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
			for node_index in range(len(self.weights[layer])):
				dot_product = np.dot(self.weights[layer][node_index], result)
				bias = self.bias[layer][node_index]
				output.append(self.sigma(dot_product + bias))
			result = output

		return result

	def backpropogate(self, examples):
		# compute the Del values for output units using observed error (eq. 18.8)
		# starting at output layer, repeat for each hidden layer until earliest reached
			# propagate the Del values back to previous layer
			# update the weights between the two layers
		# Del_j = g'(in_j)\sum_k{w_{j,k}Del_k}
		# w_{j,k}=w_{j,k}+\alpha*a_j*Del_k
		#18.8: w_i=w_i+\alpha(y-h_w(x))*h_w(x)(1-h_w(x))*x_i
		#h_w(x)=Log(w*x)=1/(1+e^{-w*x})  ---Threshold function---

		for k in range(50): #repeat some number of times
			for i in range(len(network)):
				for j in range(len(network[i])):
					network[i][j] = random.random()/10.0 #small random number

			for (x,y) in examples:
				# propagate the inputs forward to compute the inputs
				a = []
				ins = []
				for i in range(len(inputLayer)):
					a[i] = x[i]
				for l in range(2,self.layers):
					for j in range(len(network[l])):
						ins[j] = sum(weights[i][j]*a[i] for i in range(len(weights[j])))
						a[j] = sigma(ins[j])
				# propagate deltas backward from output layer to input layer
				for input in outputLayer:
					Del[j] = self.sigmaPrime(ins[j])*(y[j]-a[j])
				for l in range(self.layers-1,1,-1):
					for i in layer[l]:
						Del[i] = sigmaPrime(ins[i])*sum(w[i][j]*Del[j] for j in range(len(w[i])))

				#update every weight in netowkr using deltas
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
