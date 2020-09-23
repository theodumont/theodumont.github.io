---
layout: post
title:  "Image Segmentation by Superpixels"
categories: posts
author: Théo Dumont
keywords:
  - deep learning
  - image segmentation
  - CNN
repository: https://github.com/theodumont/superpixels-segmentation
---

During my internship at the [Center for Mathematical Morphology](http://www.cmm.mines-paristech.fr/?l=en_US) (Mines ParisTech), the objective of my research was to develop a deep learning based algorithm to generate an image superpixel partition with improved metrics.\
Our convolutional network architecture features **contextual aggregators** and a **gradient-aware loss function**. It significantly improves on the state of the art algorithms in terms of compactness and undersegmentation.

<!-- readmore -->

In this article, I will first introduce some basics on image segmentation and superpixels. I will then detail the architecture that was chosen, and finally show its results and how it improves the state of the art.

## Table of contents
1. [ Introduction ](#1-introduction)
2. [ Model ](#2-model)\
    2.1. [ Neural Network Architecture ](#21-neural-network-architecture)\
    2.2. [ Loss function ](#22-loss-function)\
    2.3. [ Hyperparameters ](#23-hyperparameters)
3. [ Dataset and training process ](#3-dataset-and-training-process)
4. [ Results ](#4-results)

## 1. Introduction

> To skip this introduction, go to section [2. Model](#2-model).

### 1.1. Image Segmentation

While looking at an image, the human brain uses a lot of prior knowledge to understand its content. But from the perspective of a computer, an image is only a set of integer valued pixels. **Segmenting** an image consists in transforming the image in a representation that is easier to analyze, and much more meaningful. A segmentation partitions an image into **multiple connected sets of pixels** sharing certain characteristics including similar colors or texture patterns. It allows one to locate the objects on an image or to point out their boundaries.

| ![Original image](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_segm1.jpg) | ![Segmented image](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_segm2.png) |
|:--:|:--:|
| _An original image from the COCO Dataset_ [[1]](#references) | _Its segmented version_ |

The applications of such a process are numerous: control of the an object outlines on a production line, face detection, medical imaging, pedestrian detection, video surveillance... They justify our search for higher segmentation performances.


Image segmentation is a challenging task and there is currently **no comprehensive theory** in this field, not least because a given segmentation is often aimed at a specific application. I will not dig into the related works here, but you can read the full report of the project [on GitHub][report].

Sometimes, the starting point of a segmentation methods consists in finding a way to build an **over-segmentation** of the image; in previous work [[2]](#references), we were able to notice that the quality of the initial over-segmentation significantly impacts the quality of the overall segmentation.

> The question of generating **deep learning** based over-segmentations has not been extensively studied yet, mostly due to the lack of proper training datasets for this particular task. By applying an algorithm based upon the _Eikonal equation_ to an image combining its RGB channels with a ground truth segmentation mask, we were able to construct a **dataset** on which machine learning algorithms can be trained to **generate superpixels**.

The objective here is therefore to develop a deep learning based algorithm to generate a superpixel partition with improved metrics.

### 1.2. Superpixels

**Superpixel algorithms** are a class of techniques that partition an image into several small groups of pixels that share **similar properties**. As such, they constitute regions on which it is relevant to compute features including mean color, mean texture, _etc_.

| ![Original image](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_spp1.png) | ![Calculated superpixels outlines](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_spp2.png) | ![Resulting superpixel segmentation](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_spp3.png)
|:--:|:--:|:--:|
| _Original image_ | _Calculated superpixels outlines_ | _Resulting superpixel segmentation_ |

To define what a "good" superpixel segmentation is, one can use a number of **metrics**. Let $$G = \{G_i\}_i$$ and $$S = \{S_j\}_j$$ be partitions of the same image $$I : x_n \mapsto I(x_n)$$, $$1 \leq n \leq N$$. $$G$$ is the ground truth segmentation mask and $$S$$ is the segmented image obtained from a superpixel algorithm.

- **Boundary Recall** Boundary recall indicates the proportion of real boundaries being detected by the segmentation, with a tolerance margin of a few pixels.\
If $$\text{D}(G,\tilde{S})$$ is the number of detected boundary pixels and $$\text{UD}(G,\tilde{S})$$ the number of undetected boundary pixels in the segmented image $$S$$, then the _boundary recall_ is:

  $$\mathrm{Rec}(G,S)=\frac{\mathrm{D}(G,\tilde{S})}{\mathrm{D}(G,\tilde{S})+\mathrm{UD}(G,\tilde{S})} \in [0,1]$$  

  $$\tilde{S}$$ being the segmented image with its boundaries dilated by a square of side 5 pixels.
  
  > Boundary recall does not measure the regularity of the boundaries at all. That means an algorithm can have a very high boundary recall while being very tortuous. This nourishes the need of a metric that quantifies the regularity of the boundaries.

- **Compactness** computes the ratio of the region area $$A(S_j)$$ with respect to a circle with the same perimeter as the superpixel $$S_j$$, weighted by the ratio of pixel numbers inside the region [[2]](#references):

  $$\mathrm{Co}(G, S)=\frac{1}{N} \sum_{S_j}|S_j| \frac{4 \pi A(S_j)}{P(S_j)^2}$$
  
  As such, a high compactness tends to indicate regular and little tortuous contours, simplifying the superpixel segmented image as much as possible.

- **Undersegmentation Error** measures the "leakage" of the superpixels over the ground truth. We adopt the formulation proposed by [[3]](#references):

  $$\mathrm{UE}(G,S)=\frac{1}{N} \left[\sum_{G_i} \left(\sum_{S_j \vert S_j \cap G_i>B}\vert S_j\vert \right)-N\right]$$
  
  Given a region $$G_i$$ from the ground truth segmentation, $$\{S_j \vert S_j \cap G_i>B\}$$ is the set of superpixels $$S_j$$ leaking across the boundaries of $$G_i$$, $$B$$ being a tolerance margin. Superpixels that do not fit the ground truth result in a high value of UE.
            
## 2. Model

### 2.1. Neural Network architecture

The primary architecture of our network is the **Context Aggregation Network** (CAN) [[4]](#references) [[5]](#references). It gradually aggregates contextual information without losing resolution through the use of **dilated convolutions**, whose field of view increases exponentially over the network layers. This exponential growth grants a global information aggregation with a compact structure.

The input data goes through the set of layers $$\{L^0, \cdots, L^d\}$$, and we choose $$d=7$$. The input and the output are RGB images, with 3 feature maps.

Each block $$L^s$$, $$s\in [\![2,d-2]\!]$$ is made of a **dilated convolution**, with parameter $$r_s=2^s$$, an **adaptive batch normalization**, and a **leaky rectifier (ReLU)**.

- a **dilated convolution**, that enables the network to get larger receptive fields, aggregating information at increasing scales on the image while preserving the input resolution;

 | ![Simple and dilated convolutions](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/conv-gif.gif) |
 |:--:|
 | _A simple and dilated convolution, of parameters $$r_s=1$$ and $$r_d=2$$_. |

- an **Adaptive Batch Normalization (ABN)**, combining identity mapping and batch normalization:

 $$\Psi(x)=a\ x+b\ BN(x),$$
 
 where $BN$ is the classic PyTorch batch normalization, defined as:
 
 $$BN(x) = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}*\gamma+\beta;$$


- and a **Leaky rectifier (LReLU)**,
 
 $$\Phi(x)=\max(\alpha x,x)\mbox{, with } \alpha=0.2.$$


Here is the architecture of our Context Aggregation Network:

| Layer `L_s`       | 1   | 2   | 3   | 4   | 5   | 6   | 7   |
|-------------------|-----|-----|-----|-----|-----|-----|-----|
| Input `w_s`       | 3   | 24  | 24  | 24  | 24  | 24  | 24  |
| Output `w_{s+1}`  | 24  | 24  | 24  | 24  | 24  | 24  | 3   |
| Receptive field   | 3x3 | 3x3 | 3x3 | 3x3 | 3x3 | 3x3 | 1x1 |
| Dilation `r_s`    | 1   | 2   | 4   | 8   | 16  | 1   | 1   |
| Padding           | 1   | 2   | 4   | 8   | 16  | 1   | 0   |
| ABN               | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| LReLU             | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | No  |

### 2.2. Loss function

In such a search for well-segmented images, two criteria have to be fulfilled. The output needs to be as close as possible to the ground truth image; but we also need the segmented image to present a lot of zones where the **color gradient** $$\nabla f(I)$$ is equal to $$0$$. Thus, we want to implement a train loss function that could help us satisfy these two criteria.

In order to grant an improvement in the output image smoothness, we use the **Total Variation (TV) loss**, that tends to yield images with a sparse gradient:

$$L_{TV}=\frac{1}{N}\sum_{i=1}^N \vert\hat{f}(I)_i-f(I)_i\vert^2+\alpha_{TV}\frac{1}{N}\sum_{i=1}^N\vert(\nabla f(I))_i\vert$$

where $$\alpha_{TV}$$ is a hyperparameter that is going to be tuned and that allow us to give more or less importance to the gradient term.

### 2.3. Hyperparameters

In [the report][report], we discuss how the hyperparameters impact the model's performances metrics and temporal efficiency and we conduct experiments to find a good-performing architecture.
We found that the following values worked well on the BSD dataset:

| batch\_size | epochs | `d` | `lr_0` | decay for `lr_0`      | `alpha_TV`           |
|-------------|--------|-----|--------|-----------------------|----------------------|
| 32          | 80     | 7   | 10^\-2 | 10^\-3 after 10 epochs| 0 [(?)](#discussion) |


## 3. Dataset and training process

By combining an algorithm that generates superpixel partitions through the resolution of the Eikonal equation and ground truth segmentations from the COCO dataset [[1]](#references), we were able to generate training examples of superpixel partitions of the images of the dataset. Our convolutional network architecture is then trained on these images. A superpixel algorithm is finally applied to the output of the network to construct the seeked partition.

Fore more information about the dataset generation and the training of the model, please read [the full report][report].

## 4. Results

The algorithm is evaluated on the Berkeley Segmentation Dataset 500. It yields results in terms of boundary adherence that are comparable to the ones obtained with state of the art algorithms including SLIC, while significantly improving on these algorithms in terms of **compactness and undersegmentation**.

 | ![An output image](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/img_bsd_res2_readme.png) |
 |:--:|
 | _Application of the model to an image of the BSD500. Original image (left) and superpixel segmented image with each superpixel being displayed with the average color of the pixels belonging to it (right)._ |


Below are evaluated the metrics for some superpixel segmentation algorithms: SLIC, FMS and our algorithm (see [report][report] for references). We use the SLIC algorithm as a reference to evaluate the performances of our model.

![Comparisons of metrics on the BSDS500 dataset](https://raw.githubusercontent.com/theodumont/superpixels-segmentation/master/report/pics/metrics.png)

|      | Undersegm. Error | Compactness | Boundary Recall | 
|------|------------------|-------------|-----------------|
| SLIC | .10              | .31         | .90             |
| FMS  | .05              | .48         | .89             |
| Ours | .04              | .77         | .88             |

_Comparisons of metrics on the BSD500 dataset. Values are for segmentations with 400 superpixels._

Our model yields very good results: the **undersegmentation** sees a `0.01` improvement, and the **compactness** is way better (improvement of `0.23`). The **boundary recall** is slightly smaller for our model than for the SLIC algorithm, but this is not a problem as the SLIC compactness is very low. The contours oscillate and thus intersect more with the ground truth image outlines.


## Discussion

This work indicates that convolutional neural networks can achieve good performance on superpixel segmentation. Because of the complexity of both the model and the problem we are trying to resolve, we were able to conduct little experiments for the tuning of our hyperparameters. It would thus be interesting to push the study further, notably by conducting **additional tests** for a better choice of our parameters. We ran some experiments on the implementation of a U-Net network, but only have a few results yet. We would like to continue testing this network, and try to mix it with the CAN, for instance adding only one U-Net layer on the CAN. Furthermore, testing our model on other datasets would show how well it **generalizes**.

Moreover, whereas at first glance implementing a **TV-loss regularization** could totally have yielded better results, it _does not improve the model_. We tried to analyze why such an initiative didn't pay off. Actually, adding the gradient term in the loss function is enjoining the network to reduce the overall gradient function of the image. Nevertheless, while a superpixel segmented image has a lot a regions where its gradient is 0, it also has a **very high gradient** on the objects contours.\
Although the results obtained are satisfying without a TV-regularization, we could pre-process each image by dividing it into several smaller ones to reduce the amount of outlines in each image the neural network is going to process. However, the COCO dataset is not suitable for this task as its images have a very low resolution. Cutting them would result in losing a lot of global information and thus making the use of a CAN _meaningless_.

### References

See [here](https://github.com/theodumont/superpixels-segmentation/blob/master/report/main.pdf#page=15) for full list of references.

**[1]**&emsp;_Microsoft coco: Common objects in context_, 2014 ([website](http://cocodataset.org/#home)).\
**[2]**&emsp;Alexander Schick, Mika Fischer, and Rainer Stiefelhagen, _Measuring and evaluating the compactness of superpixels_ (ICPR 2012), 2012.\
**[3]**&emsp;Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurélien Lucchi, Pascal Fua, and Sabine Susstrunk, _SLIC Superpixels Compared to State-of-the-Art Superpixel Methods_, 2012.\
**[4]**&emsp;Fisher Yu and Vladlen Koltun, _Multi-scale context aggregation by dilated convolutions_, 2015.\
**[5]**&emsp;Qifeng Chen, Jia Xu, and Vladlen Koltun, _Fast image processing with fully-convolutional networks_, 2017.


<!-- references -->
[report]: https://github.com/theodumont/superpixels-segmentation/blob/master/report/main.pdf
