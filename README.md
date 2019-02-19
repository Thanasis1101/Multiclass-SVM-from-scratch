# Multiclass SVM from scratch
Multiclass (one vs one) Support Vector Machine implementation from scratch in Matlab

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*csqbt5-K4GVi4i4Lrcx_eA.png" width="350" title="SVM">
</p>

This repository is an effort to build an SVM (for classifying multiple classes) from scratch. It uses the one vs one apprach to classify
the data. It is not guaranteed to produce the best results and can not be compared to famous libraries such as
[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or [scikit-learn](https://scikit-learn.org/stable/modules/svm.html). It was made for educational purposes. To test the code,
I used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (digits 0-9), but the code is general so that the SVM can be trained
on any dataset. To read the data in matlab these [mnistHelper functions](http://ufldl.stanford.edu/wiki/index.php/MATLAB_Modules)
were helpful. I would appreciate any bug report, any correction on the code, or any suggestion to improve this repository.

## Index

[Files explained](#files-explained)

[How to build the SVM](#how-to-build-the-svm)

## Files explained

- **Main**
  - [mysvm.m](/mysvm.m)
   
   Here we read the input dataset, train all the support vector machines, test them to find the precision and save the model
   ([svm_model.mat](/svm_model.mat)). For the classification we use the one vs one approach, in which we train one SVM for every
   class combination. For our problem with MNIST we created 45 SVMs (digit 0 vs digit 1, 0 vs 2, ..., 1 vs 2, ..., 8 vs 9).
   
- **Functions for the input**
  - [readMNISTImages.m](/readMNISTImages.m)
  - [readMNISTLabels.m](/readMNISTLabels.m)
  - [labels2goals.m](/labels2goals.m)
  
  In order to read the input samples for training ([train-images.idx3-ubyte](/train-images.idx3-ubyte)) and testing
  ([t10k-images.idx3-ubyte](/t10k-images.idx3-ubyte)) we use the ``readMNISTImages`` function, whereas in order to read the
  labels of the input samples for training ([train-labels.idx1-ubyte](/train-labels.idx1-ubyte)) and testing
  ([t10k-labels.idx1-ubyte](/t10k-labels.idx1-ubyte)) we use the ``readMNISTLabels`` function. Now the labels are a number for
  every input, which is the digit show in the input image, so in order to convert this to a vector with desired values for the
  output neurons we use the ``labels2goals``.
    
  
- **SVM Training**
  - [fit_training_data.m](/fit_training_data.m)
  
  Here we train each individual SVM. This function performs the qp optimisation and returns alpha (solutions of QP), the
  support vectors (the samples from x that are the supported vectors), the corresponding signs of supported vectors (y) and the bias.
  It needs 6 parameters. The first is a matrix with the inputs (each line is a different input). The second is an 1D maatrix with the
  class (-1 or 1) of the corresponding input. The third is the soft margin parameter (C). The fourth is one very small value (e.g. 10^-4)
  according to which, all alphas that are smaller than it, will be cosidered as 0. The fifth parameter is the kernel name (``linear``,
  ``polynomial``, ``rbf``) and the sixth is the corresponding parameter of the kernel function. For the linear, use any value (it does
  not matter). For the polynomial, input the degree of the polynomial. For the rbf, input the g (γ) parameter (considering this 
  [rbf kernel formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/513a31a936b91e04dae78cdf630d1d8c7ab5186b)).
  
  
- **Kernel function**
  - [kernel.m](/kernel.m)
  
  This is the function for calculating the kernel value. It accepts four parameters. The first one is a string with the name of
  the kernel function that will be used. Currently there are three functions that are supported: ``linear``, ``polynomial``, ``rbf``.
  The next 2 parameters are the x1 and x2 matrices for the calculations (for the SVMs it is x and x transpose). The fourth parameter is
  the parameter needed for the   corresponding function (see SVM Training above).
  
  

## How to build the SVM

```Matlab
% ===== Inputs ===== %

train_set_inputs = readMNISTImages('train-images.idx3-ubyte')';
train_labels = readMNISTLabels('train-labels.idx1-ubyte');
train_set_goals = labels2goals(train_labels, 10);
test_set_inputs = readMNISTImages('t10k-images.idx3-ubyte')';
test_labels = readMNISTLabels('t10k-labels.idx1-ubyte');
test_set_goals = labels2goals(test_labels, 10);

C = 2;
number_of_samples_from_every_class = 500; % set to Inf for all samples
kernel_name = 'linear';
param = 1;
epsilon = 10^-4; % from 0 to epsilon all alphas are cosidered as zero
```
#### Variables needed:
- **train_set_inputs, test_set_inputs** (2D matrix with the vector of an input sample in each row)
- **train_set_goals, test_set_goals** (2D matrix with the desired values of the output neurons of the corresponding input)
- **C** (soft margin parameter)
- **number_of_samples_from_every_class** (how many samples to use from each class)
- **kernel_name** (see SVM Training in [Files explained](#files-explained))
- **param** (see SVM Training in [Files explained](#files-explained))
- **epsilon** (see SVM Training in [Files explained](#files-explained))

