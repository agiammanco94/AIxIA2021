# Adversarial Machine Learning in e-Health: a Fooled Smart Prescription Case Study
==========================================================

- [Setup](#Setup)
- [Modules](#Modules)
	- [Dataset](#Dataset)
	- [Neural_Network](#Neural_Network)
	- [Mock_Example](#Mock_Example)
	- [Experiments](#Experiments)
	- [Results_Evaluation](#Results_Evaluation)
	- [Classification_Utilities](#Classification_Utilities)
	- [Other_Utilities](#Other_Utilities)
- [Credits](#Credits)

==========================================================

## Setup

[Python3](https://www.python.org/downloads/) is required for this project.

The AMR-UTI dataset has to be downloaded from the 
[official site](https://physionet.org/content/antimicrobial-resistance-uti/1.0.0/).

It is highly recommended to configure a 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) virtual environment. 
For example, using the Miniconda distribution, the following steps provide the needed setup.

1) Download the Miniconda installer: [linux](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

2) Open a terminal window in the download folder, give the execution permission to the downloaded installer, and run
```
./Miniconda3-latest-Linux-x86_64.sh
```
follow the instructions on screen accepting the defaults. Type "yes" when prompted whether to run conda init.

3) Close the terminal and reopen it inside the project folder.

4) Create a conda virtual environment with all the dependencies listed in <i>requirements.txt</i>
```
conda create --name conda_env --file requirements.txt
```

5) Activate the virtual environment
```
conda activate conda_env
```

6) Now the python modules can be correctly executed from terminal. For example, to launch the script in 
   <i>py/classification/mock_example</i>, from the root folder of the project launch the command:
```
python -m py.classification.mock_example
```

## Modules

The project is structured into two packages:
<ol>
	<li>
		<b>classification</b>: this package contains the modules with the neural network implementation, with all its 
training procedures and other
							utilities functions, as well as the implementation of the Fast Gradient Sign Method, and 
the attack algorithm
							proposed in this paper. The package is made up of five modules:
		<ul>
			<li><i>class_utils</i></li>
			<li><i>experiments</i></li>
			<li><i>mock_example</i></li>
			<li><i>neural_net</i></li>
			<li><i>results_evaluation</i></li>
		</ul>
	</li>
	<li>
		<b>utilities</b>: this package contains the preprocessing procedure to treat the AMR-UTI dataset, as well as 
other utilities functions.
						The package is made up of two modules:
		<ul>
			<li><i>dataset</i></li>
			<li><i>miscellaneous</i></li>
		</ul>
	</li>
</ol>

The comments accompanying the code are written following the 
[Google Python Style](https://google.github.io/styleguide/pyguide.html).

In "docs/sphinx_docs/_build/latex/vasariproject-recommender.pdf" it can be found the documentation of the source code,
generated with [Sphinx](https://www.sphinx-doc.org/en/master/).

In what follows, the single modules will be described with a brief explanation of their intended use.

### Dataset

The module <i>py/utilities/dataset.py</i> contains the preprocessing algorithm adopted for the AMR-UTI dataset, 
described in Section V-A of the paper.

The AMR-UTI dataset has to be downloaded, and its main files ("<i>all_prescriptions.csv</i>", 
"<i>all_uti_features.csv</i>", "<i>all_uti_resist_labels.csv</i>", "<i>data_dictionary.csv</i>") have to be 
placed in the <i>datasets/AMR-UTI</i> folder.

### Neural_Network

The module <i>py/classification/neural_net.py</i> provides an implementation of neural network, which is based on the 
Numpy implementation in [Credits](#Credits).
    
The Fast Gradient Sign Method, and the attack algorithm proposed in our paper are implemented as methods of the
NeuralNetwork class.

### Mock_Example

The module <i>py/classification/mock_example.py</i> contains the reproducible mock example which is described in what
follows.

Let:

![Alt text](pics/1.svg?raw=true)

![Alt text](pics/2.svg?raw=true)

be the input vector we want to perturb and its associated ground truth class label.

The adversary aims at finding the perturbation ![equation](https://latex.codecogs.com/svg.latex?%5Cdelta)
to add to the input vector so that:

![Alt text](pics/3.svg?raw=true)

where ![equation](https://latex.codecogs.com/svg.latex?%5Ctheta) is the hypothesis of the model with the trained
parameters ![equation](https://latex.codecogs.com/svg.latex?%5Ctheta).

The first step consists in computing the actual hypothesis of the network for the input sample 
![equation](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D), and verifying that it is coherent with the ground truth 
label ![equation](https://latex.codecogs.com/svg.latex?y).

Let ![equation](https://latex.codecogs.com/svg.latex?%5Ctheta%20%3D%20%5C%7BW%5E%7B%281%29%7D%2C%20W%5E%7B%282%29%7D%2C%20b%5E%7B%281%29%7D%2C%20b%5E%7B%282%29%7D%5C%7D) be the parameters 
of the trained neural network we want to elude.

In particular, in order to demonstrate this point we trained a neural network with 5 neurons on the first hidden layer 
and 2 neurons in the output layer.

Following the preprocessing methodology that is further discussed in Section 5.1 of the paper, we selected a 
subset of the adopted dataset choosing the 7 most meaningful features, of which 
![equation](https://latex.codecogs.com/svg.latex?x) represents a sample of the test set.

We trained this two-layers neural network in 100 epochs with gradient descent and the following training 
hyperparameters: as learning rate ![equation](https://latex.codecogs.com/svg.latex?%5Calpha%3D0.01); as weight decay 
coefficient ![equation](https://latex.codecogs.com/svg.latex?%5Clambda%3D0.1); as momentum coefficient 
![equation](https://latex.codecogs.com/svg.latex?%5Cnu%3D0.6); as loss function the cross entropy 
![equation](https://latex.codecogs.com/svg.latex?L%28%5Ctheta%2C%20x%2C%20y%29%20%3D%20-%20y%20%5Codot%20log%5C%2C%20%5Chat%7By%7D), 
where 
![equation](https://latex.codecogs.com/svg.latex?%5Codot) is the element-wise multiplication between vectors, 
![equation](https://latex.codecogs.com/svg.latex?y) and 
![equation](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) are the known and the predicted probability 
for the input vector ![equation](https://latex.codecogs.com/svg.latex?x).

The learned parameters of the neural network (approximated at the second decimal digit) are the following:

![Alt text](pics/4.svg?raw=true)

![Alt text](pics/5.svg?raw=true)

![Alt text](pics/6.svg?raw=true)

![Alt text](pics/7.svg?raw=true)

The first step of the attack algorithm consists in computing the forward pass of the neural network w.r.t. the input 
vector we want to perturb.

The linear computation of the first layer is:

![Alt text](pics/8.svg?raw=true)

The activation function of the first layer is the [ReLU](https://www.deeplearningbook.org/):

![Alt text](pics/9.svg?raw=true)

The linear computation of the second layer is:

![Alt text](pics/10.svg?raw=true)

The activation function of the second layer is the Softmax, which we perform in the numerically stable variant
described in the [Deep Learning book](https://www.deeplearningbook.org/).
Let ![equation](https://latex.codecogs.com/svg.latex?%5Coverline%7Bz%7D%5E%7B%282%29%7D%20%3D%20z%5E%7B%282%29%7D%20-%20%5Cunderset%7Bi%7D%7Bmax%7D%5C%3B%20z%5E%7B%282%29%7D), 
and let ![equation](https://latex.codecogs.com/svg.latex?j) represent the 
![equation](https://latex.codecogs.com/svg.latex?j)-th  neuron in the layer, the activation layer performs the 
computation:

![Alt text](pics/11.svg?raw=true)

and this concludes the forward pass of the network: ![equation](https://latex.codecogs.com/svg.latex?h_%5Ctheta%28x%29%20%3D%20%5Chat%7By%7D%20%3D%20%5B0.40%2C%200.60%5D).

By considering the index of the maximum predicted probability as the predicted class, the input vector sampled is 
classified as a sample belonging to the second class, which corresponds to the ground truth layer.

The objective of the adversary is to flip the predicted label for the input vector 
![equation](https://latex.codecogs.com/svg.latex?x). In other terms, he wants to find the perturbation 
![equation](https://latex.codecogs.com/svg.latex?%5Cdelta) such that 
![equation](https://latex.codecogs.com/svg.latex?h_%5Ctheta%28x%20&plus;%20%5Cdelta%29%20%5Cneq%20y).

First, the adversary puts in place a revised version of the well known 
[Fast Gradient Sign Method](https://arxiv.org/pdf/1412.6572.pdf).

The traditional approach aims at climbing up the gradient by adding the perturbation 
![equation](https://latex.codecogs.com/svg.latex?%5Cxi%20%3D%20%5Cepsilon%20%5Ccdot%20sign%28%5Cnabla_x%20L%28%5Ctheta%2C%20x%2C%20y%29%29), 
so that the perturbation vector ![equation](https://latex.codecogs.com/svg.latex?%5Cxi) is composed by values equal to 
![equation](https://latex.codecogs.com/svg.latex?%5Cpm%20%5Cepsilon).

In this paper we decided to consider only the term:

![Alt text](pics/12.svg?raw=true)

so that each single ![equation](https://latex.codecogs.com/svg.latex?%5Cxi_i%20%5Cin%20%5Cmathbb%7BR%7D). 
The reason lies in the 
need to select only a small subset of the input features to perturb. Opposed to the image processing domain, where each 
pixel may be perturbed with a small step in its scale of representation, in different domains where the features may 
assume a limited set of values (e.g., binary values), each single perturbation added to the input features has to be 
selected with care.

Let us now consider the computations performed by the adversary.
In order to compute the gradient of the cost function w.r.t. the input vector, he needs to propagate the gradient back 
starting from the output layer.

The derivative of the cross-entropy loss function w.r.t. ![equation](https://latex.codecogs.com/svg.latex?z%5E%7B%282%29%7D), 
given that ![equation](https://latex.codecogs.com/svg.latex?%5Csigma%5E%7B%282%29%7D) is the Softmax activation 
function, is [simply](https://www.deeplearningbook.org/):

![Alt text](pics/13.svg?raw=true)

For the chain rule of calculus, and considering that the local gradient is 
![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20z%5E%7B%282%29%7D%7D%7B%5Cpartial%20%5Csigma%5E%7B%281%29%7D%7D%20%3D%20W%5E%7B%282%29%7D):

![Alt text](pics/14.svg?raw=true)

The activation function of the first layer is the ReLU function, whose local derivative is:

![Alt text](pics/15.svg?raw=true)

applying the derivatives chain rule:

![Alt text](pics/16.svg?raw=true)

The last step of backward propagation, considering that the local gradient is 
![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20z%5E%7B%281%29%7D%7D%7B%5Cpartial%20x%7D%20%3D%20W%5E%7B%281%29%7D), 
implies the computation:

![Alt text](pics/17.svg?raw=true)

From this real-valued perturbations vector, the adversary has to craft a binary perturbation mask to add to the input 
sample in order to flip the neural network's prediction. Let us suppose that the application domain impose some 
constraint on the features that may be altered; we represent this constraint in the form of a binary mask, that in our 
mock example has the value:

![Alt text](pics/18.svg?raw=true)

so that the third feature ![equation](https://latex.codecogs.com/svg.latex?x_3%20%5Cin%20x) must remain fixed.

The number of features the adversary may alter is supposed to be the minimum possible, i.e., 
![equation](https://latex.codecogs.com/svg.latex?%5Cpsi%20%3D%201).

A single input feature ![equation](https://latex.codecogs.com/svg.latex?x_i) may be altered in two cases:

1) ![equation](https://latex.codecogs.com/svg.latex?x_i%20%3D%3D%201) and the perturbation which results from 
		ascending along the gradient of the loss function has negative sign, so that 
		![equation](https://latex.codecogs.com/svg.latex?x_i) may be flipped by adding 
		![equation](https://latex.codecogs.com/svg.latex?%5Cdelta_i%20%3D%20-1).
		In other words, a feature value of 1 can be altered to 0 only if the sign of the gradient along that feature is 
		negative;
   
2) ![equation](https://latex.codecogs.com/svg.latex?x_i%20%3D%3D%200) and the perturbation has positive sign instead, 
		in order to flip ![equation](https://latex.codecogs.com/svg.latex?x_i), 
		![equation](https://latex.codecogs.com/svg.latex?%5Cdelta_i%20%3D%20&plus;1) may be added.


To achieve this aim, we first take the sign of the perturbation vector 
![equation](https://latex.codecogs.com/svg.latex?%5Cxi), which is then XORed with the input vector 
![equation](https://latex.codecogs.com/svg.latex?x).

The logic behind the XOR operation is that we want a function that is evaluated True only when the input feature is 1 
and the perturbation has negative sign, and vice versa.
The result of this operation is then processed with a bit-wise AND with the input 
![equation](https://latex.codecogs.com/svg.latex?mask), finally obtaining a binary 
vector which signals all those features which, if altered, make the neural network increase the error, 
since their alteration is concordant with the direction of the gradient.

![Alt text](pics/19.svg?raw=true)

Finally having the list of features he may alter to fool the neural network, the adversary chooses the 
![equation](https://latex.codecogs.com/svg.latex?%5Cpsi) 
features with maximum absolute value, in order to take the steepest step along the gradient.

In the considered example, the adversary choose to alter ![equation](https://latex.codecogs.com/svg.latex?x_5), 
which is the feature with the highest magnitude of perturbation 
(![equation](https://latex.codecogs.com/svg.latex?%5Cxi_5%20%3D%20-0.57)) among the corruptible features.
Since ![equation](https://latex.codecogs.com/svg.latex?x_5%20%3D%201), the perturbation consists in setting the bit to 0, 
that is to say:

![Alt text](pics/20.svg?raw=true)

Computing the hypothesis of the neural network on the corrupted sample 
![equation](https://latex.codecogs.com/svg.latex?%5CTilde%7Bx%7D%20%3D%20x%20&plus;%20%5Cdelta), we obtain:

![Alt text](pics/21.svg?raw=true)

We have thus shown that, by changing a single feature in an input vector which the neural network classified as a sample 
belonging to the second class with 60% confidence, the adversary succeeded in letting the neural network believe that 
the new sample belongs to the first class, ![equation](https://latex.codecogs.com/svg.latex?y_%7Btarget%7D), with 72% 
confidence.

### Experiments

The module <i>py/classification/experiments.py</i> performs the experimental evaluation described in Section V of the 
paper, by means of the Random search of the hyperparameters.

The experimental results are stored according to the following folder hierarchy:

<ol>
	<li>
		The parent folder is <i>py/classification/results/AMR-UTI</i>.
	</li>
	<li>
		At this point, there are the sub-folder with a specific value of \kappa best features selected with the
		chi-square test of independence. E.g., <i>59_features_considered</i>.
	</li>
	<li>
		Now are listed the different neural networks architectures in terms of number of layers.
		E.g., <i>2_layers</i>.
	</li>
	<li>
		The number of neurons per layer are listed at this point of the folder hierarchy, with an underscore
		separating the number of neurons in each layer. E.g., <i>215_2</i>.
	</li>
	<li>
		At this point are listed the number of training epochs, together with the number of samples in each batch of
		training. E.g., <i>500_epochs_601_batch_size</i>.
	</li>
	<li>
		Here, are finally listed all the training hyperpameters explored with random search, following the order:
		<i>learning-rate_weight-decay_momentum-coefficient</i>. E.g., <i>5.18e-04_3.25e-05_7.65e-01</i>.
	</li>
</ol>

Therefore, considering for example the third run exposed in Table III of the paper, the path containing all the results
is: 
<i>py/classification/results/59_features_considered/2_layers/215_2/500_epochs_601_batch_size/5.18e-04_3.25e-05_7.65e-01/</i>

Here, are contained all the plots which represent the f-scores of the two classes, as well as
the percentage error achieved performing the proposed attack algorithm along the different training epochs.
In particular, the folder <i>single_epoch_results</i> contains all the prediction results feeding the neural
network with the test set after every single epoch of training.
Whereas, the folder <i>attacks</i> contains the results of the attack algorithm performed after each single
epoch of training.

### Results_Evaluation

The module <i>py/classification/results_evaluation.py</i> provides the functions to evaluate the results of all 
the neural network runs with hyperparameters explored through random search.

### Classification_Utilities

The module <i>py/classification/class_utils.py</i> contains utilities functions useful for the classification task, e.g., 
for computing the one-hot encoding of class labels, for splitting the train dataset in batches, etc.

### Other_Utilities

The module <i>py/utilities/miscellaneous.py</i> contains various utilities function, mainly related to the navigation 
in the project folder structure, and to the management of time.

## Credits

Our implementation of the neural network is based on the Numpy implementation given by 
[Rafay Khan](https://github.com/RafayAK/NothingButNumPy).
