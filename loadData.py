# loadData.py
# Thanks to Deep Learning (http://deeplearning.net/) for the code and
# nicely pickled data.
# 03.11.17

import cPickle, gzip, numpy

def loadMNIST():
	# Load the dataset.
	f = gzip.open('mnist.pkl.gz', 'rb')
	# Each set is a list of the form [images[], labels[]].
	# images[]: list of images, each a numpy array of length 784 (28 x 28)
	# 	where each item is a grayscale pixel (float values between 0 and 1
	# 	(0 stands for black, 1 for white).
	# labels[]: list of labels for the images (integers 0-9).
	train_set, dev_set, test_set = cPickle.load(f)
	f.close()
	return train_set, dev_set, test_set

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