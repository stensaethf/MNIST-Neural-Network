# NeuralNetwork
# Neural network implementation.
# 03.11.17

import loadData, numpy

class NeuralNetwork:
	def __init__(self):
		# do stuff

		# filler, remove later
		x = 1

	def feedForward(self):
		# do stuff

		# filler, remove later
		x = 1

	def backpropogate(self):
		# do stuff

		# filler, remove later
		x = 1

	def sigma(self):
		# do stuff

		# filler, remove later
		x = 1

def main():
	# Loads the train, dev and test sets.
	# 50,000, 10,000, 10,000
	train, dev, test = loadData.loadMNIST()
	# Gets the training images.
	train_images = train[0]
	# Gets the training labels.
	train_labels = train[1]

	print len(train_images)
	print len(train_labels)

if __name__ == '__main__':
	main()
