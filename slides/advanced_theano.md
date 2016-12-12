name: inverse
layout: true
class: center, middle, inverse

---

# Advanced Theano 

### Daniel C. LaCombe, Jr.

<center> Ph.D. Candidate, Florida Atlantic University Brain Institute </center>
<center> CTO, VoxelRx | Deep Learning Enthusiast </center>

---
layout: false

# Review

* Theano Introduction

  * Tensors - Symbolic & Shared
  * Basic Operations - Dot Product
  * Applying Gradients as Updates

* Intermediate Theano
  * Building Deep Neural Nets for Binary Classification
  * Initializing Weight and Bias Parameters
  * Activation Functions
  * Model Complexity

---

# Today


* We want to use Theano to solve real world problems that we can’t hard-code using machine learning
* Solving these problems often require: 

    * Quick iterations
        * _Fail as much and as quickly as possible_
    * Creativity
        * Move from excitement about learning the language to excitement about what the language can do for you
    * Intensive Code Maintenance
        * Branching out into new ideas can result in overly verbose code that is time consuming to modularize

---

# More Functionality = More Problems

* Different projects or improving existing ones usually lead to needs for more functionality
* Code length explodes and is time-consuming to manage
  
Why do all of this when somebody has already done most of the work for us?

---

# [Keras](https://keras.io/)

_"[Keras] was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research."_ - François Chollet

Use Keras if you need a deep learning library that:
* Allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
* Supports both convolutional networks and recurrent networks, as well as combinations of the two.
* Supports arbitrary connectivity schemes (including multi-input and multi-output training).
* Runs seamlessly on CPU and GPU.

.footnote[.red[*] Slide content taken directly from [project site](https://keras.io/)]

---

# Keras Import

```Python
import json
from model import model_
from utilities.data_utilities import load_data
from keras.preprocessing.image import ImageDataGenerator


# choose model type and build it
model = model_()

# load MNIST data based on number of labeled points
n_train = 600
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = \
    load_data(n_train=n_train)

# build generators for training and validation data
train_generator = ImageDataGenerator(
    rescale=1 / 256.,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
)
valid_generator = ImageDataGenerator(
    rescale=1 / 256.
)```

---

# Keras

* Light-weight library but highly functional
* Encapsulates most types of operations needed
* Easily extensible
* Multiple backends
  * Theano
  * TensorFlow
  * CNTK soon hopefully!
* Huge community contributing so improves daily

---

# Keras Functionality

Functionality out the wazoo
* Activation functions
* Regularizations
* Constraints
* Callbacks
* Datasets
* Objectives
* Metrics
* Pre-trained models

---

# Types of Layers

Fully-Connected
Convolutional
Pooling
Recurrent
Normalization
Output
Merge
Dropout

---

# Data Types and Common Layers for Them

Time series - 1d convolutions / lstm / gru
Images / spectograms - 2d convolutions / dilated convolutions
Spatiotemporal / volumetric data - 3d convolutions

---

# Pros and Cons

Pros
More compact
Less coding
Easier to manage
Quicker
Cons
Less access to lower-level API
Behind-the-scenes functionality might have unexpected effects

---

# Common Pitfalls

Nans 
Theano Nan Guard
OOM Errors
Reduce number of parameters in model
In Keras: `model.summary()`
Reduce batch size

---

# Directory Structure

ProjectName
  > approach1
  > approach2
  > utilities
  __init__.py

---

# General Workflow

Baseline Model
Better Models
Import baseline architecture from Baseline Model

