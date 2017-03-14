# MNIST-Neural-Network
Neural network for recognizing handwritten digits (MNIST dataset)

To run this:
1. Have numpy installed
2. Run with default settings by typing 'python neuralNetwork.py'
    This will train on all 50,000 training points, then test accuracy
    on the 10,000 dev set points.
3. Try out some of the options! You can change whether or not you run
    mini-batch, the size of the batch, the learning rate alpha, the
    number of iterations, and the size(s) of the hidden layers. Type
    'python neuralNetwork.py --help' if you ever forget.
4. We found maximum accuracy with hidden layers 397-204, alpha=0.3
    and 15 iterations. Note: this will take a long time to run. The
    default settings should run in less than 3 minutes.