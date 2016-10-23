# deephypercnn
Classification of Hyperspectral Satellite Image Using Deep Convolutional Neural Networks. This is implementation of the paper 

[1] K. Makantasis, K. Karantzalos, A. Doulamis and N. Doulamis, "Deep supervised learning for hyperspectral data classification through convolutional neural networks," 2015 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), Milan, 2015, pp. 4959-4962.

# Method details
1) For each non-zero labelled pixel, we extract 5 x 5 x c neighbourhood and corresponding label.
2) Dimensionality reduction using PCA is performed. Final dimension is 5 x 5 x cr.
3) Training using CNN is performed with the following architecture:
  conv1-conv2-conv3-conv4-hidden1-hidden2-16way-softmax
4) Training : testing split ratio is maintained at 0.8:0.2

# Results

Table 1 : Comparison of accuracy for various classification methods

|      Dataset     	| No. of Components 	| RBF-SVM 	| CNN [1] 	| Our CNN 	|
|:----------------:	|:-----------------:	|:-------:	|:-------:	|:-------:	|
|   Indian Pines   	|         30        	|  82.79  	|  98.88  	|  98.94  	|
| Pavia University 	|         10        	|  93.94  	|  99.62  	|  99.66  	|


![alt text](/assets/Image_results.png "Classification results on both dataset. Only 16 non-zero labels are tested")

# Implementation
Data preparation : Matlab (Mat file)
CNN classification : Theano + Lasagne+ Nolearn
