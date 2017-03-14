# loadData.py
# Thanks to Deep Learning (http://deeplearning.net/) for the code and
# nicely pickled data.
# 03.11.17

import cPickle, gzip, numpy as np

def loadMNIST():
	"""
	Function for loading the MNIST dataset.
	"""
	# Load the dataset.
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, dev_set, test_set = cPickle.load(f)
	f.close()

	# Reshape the dataset.
	train_set = reshapeSet(train_set)
	dev_set = reshapeSet(dev_set)
	test_set = reshapeSet(test_set)

	return train_set, dev_set, test_set

def reshapeSet(data):
	"""
	Function for reshaping the dataset. We want the images to be numpy arrays
	of 781x1 and the labels to be numpy arrays of 10x1.
	"""
	images = data[0]
	labels = data[1]

	images_reshaped = []
	labels_reshaped = []

	# Reshape the images and labels to be numpy arrays of dimensions 784x1 and
	# 10x1, respectively.
	for i in range(len(images)):
		image_reshaped = np.reshape(images[i], (784, 1))
		label_reshaped = np.zeros((10, 1))
		label_reshaped[labels[i]] = 1.0
		images_reshaped.append(image_reshaped)
		labels_reshaped.append(label_reshaped)

	return (images_reshaped, labels_reshaped)

def main():
	# Loads the train, dev and test sets.
	train, dev, test = loadMNIST()
	# Gets the training images.
	train_images = train[0]
	# Gets the training labels.
	train_labels = train[1]

	print len(train_images)
	print len(train_labels)

if __name__ == '__main__':
	main()