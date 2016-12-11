# Neural-Networks

In the directory gestures, there is a set of images1 that display "down" gestures (i.e., thumbs-down
images) or other gestures. In this assignment, you are required to implement the Back Propagation
algorithm for Feed Forward Neural Networks to learn down gestures from training images available in
downgesture_train.list. The label of an image is 1 if the word "down" is in its file name; otherwise the
label is 0. The pixels of an image use the gray scale ranging from 0 to 1. In your network, use one input
layer, one hidden layer of size 100, and one output node. Use the value 0.1 for the learning rate. For
each perceptron, use the sigmoid function Ɵ(s) = 1/(1+e-s). Use 10000 training epochs; and then use the
trained network to predict the labels for the gestures in the test images available in
downgesture_test.list. For the error function, use the standard least square error function.
The image file format is "pgm" <http://netpbm.sourceforge.net/doc/pgm.html>. Please follow the link
for the format details.
