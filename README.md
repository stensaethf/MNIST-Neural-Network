# MNIST-Neural-Network
Neural network for recognizing handwritten digits (MNIST dataset)

To run this:
1. Have numpy installed
2. Run with default settings by typing 'python neuralNetwork.py'
    This will train on all 50,000 training points, then test accuracy
    on the 10,000 dev set points.
3. Try out some of the options! You can change whether or not you run
    mini-batch, the size of the batch, the learning rate alpha, the
    number of iterations, the size(s) of the hidden layers, and much
    more! Type 'python neuralNetwork.py --help' if you ever forget.
4. We found maximum accuracy with hidden layers 397, alpha=0.3, and
    10 iterations. This takes a while to run, so we have saved the
    weights from running it. Use the 'load-weights [filename]' option
    to just load up the weights and not train. The default settings
    (one hidden layer with 100 neurons) should run in 3 minutes or less.
