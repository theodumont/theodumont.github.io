---
layout: post
title:  "pyAMNESIA"
categories: posts
author: Théo Dumont
keywords:
  - clustering
  - skeleton
  - matrix factorization
  - image processing
  - calcium imaging
repository: https://gitlab.com/cossartlab/pyamnesia
---

pyAMNESIA is an **image analysis pipeline**. It can perform several general operations, such as **clustering** coactive elements on a video, extracting the **skeleton** of a still structure in a video, of perform PCA, NMF, and other **matrix factorization** techniques.

<!-- readmore -->

## Full story

![INMED logo](https://gitlab.com/cossartlab/pyamnesia/-/raw/main/docs/img/inmed.png)

In the context of an internship at the [Cossart lab](http://www.inmed.fr/en/developpement-des-microcircuits-gabaergiques-corticaux-en), [Tom](https://www.linkedin.com/in/tom-szwagier/) and I built a pipeline for performing some image analysis techniques. The Cossart lab aims at gaining an understanding of hippocampal function at circuit level in patho-physiological conditions by studying the developmental of its network dynamics.

I'll write below a few lines about the idea of the project and how we implemented the algorithm. For more information, you can check the [GitLab repository](https://gitlab.com/cossartlab/pyamnesia) or the [documentation](https://pyamnesia.readthedocs.io) of the project.

### Motivation

In the context of studying the mechanisms by which hippocampal assemblies evolve during the development of mice (5 to 12 days), we dispose of some two-photon calcium imaging videos, such as this one:

| ![A two-photon calcium imaging video](https://gitlab.com/cossartlab/pyamnesia/-/raw/main/docs/img/overview/introduction/calcium_imaging.gif) |
|:--:|
| _A two-photon calcium imaging video_ |

The analysis of these calcium imaging videos may be split into two different parts:

- the evolution of the **morphology**, being the number of elements (somata, neurites...) and their visible connexions;
- the evolution of the **neuronal activity**, being the transients characteristics and the coactivity (having similar and simultaneous neural activities over time) of the aforementioned elements.


In the state-of-the-art calcium imaging pipelines such as [Suite2p](https://github.com/MouseLand/suite2p) and [CaImAn](https://github.com/flatironinstitute/CaImAn), somata are segmented to subsequently study the evolution of their activity. In such cases, pixels in the region of a soma can reliably be considered as coactive, making the soma a *morphological* **and** *functional* entity.

Here, we want to study the **whole neural structure** of the hippocampus; not only the somata but also the neurites linking them. And there is **no obvious correlation** between the morphological "branches" that one can see on the videos, and the neurites, that are coactive functional entities. This is mainly due to the Z-axis projection - which "hides" vertical neurites and creates overlaps - but also to imaging hazards, or simply neurites that only activate partially during a neural transmission. Therefore, we do not know *a priori* whether a visible "branch" is a functional entity or not. We thus need to **separate** the morphological and the functional approaches, and try to build some **coherent structures** that could be considered as entities, in order to analyse their **coactivity** later.


Here is the problem we are trying to answer:


> How can we use calcium imaging to get **statistics** on the evolution of interneurons in the hippocampus (morphology *and* activity) during **mice development**?
>   
> - **input:** a calcium imaging ``.tif`` sequence
> - **output:** clusters of coactive pixels & morphological statistics


### Functionalities

We split our approach in three parts:

| _Structure of the approach_ |
|:--:|
| ![Structure of the approach](https://gitlab.com/cossartlab/pyamnesia/-/raw/main/docs/img/overview/introduction/structure.png) |


1. The [skeletonization](https://pyamnesia.readthedocs.io/en/latest/overview_skeleton.html) module focuses on the **morphological** analysis of the data, by computing the underlying *morphological skeleton* of the sequence.
2. The [clustering](https://pyamnesia.readthedocs.io/en/latest/overview_clustering.html) module performs an **activity analysis** of the elements in the sequence, whether they be pixels (no prior skeleton analysis) or branches (see the [skeleton module page](https://pyamnesia.readthedocs.io/en/latest/overview_skeleton.html) for more information about this). Its goal is to return clusters of coactive pixels.
3. The [factorization](https://pyamnesia.readthedocs.io/en/latest/overview_factorization.html) module has a similar goal to the clustering module. It returns independent components of **coactive pixels**, but uses **matrix factorization techniques** to do so.

| ![From left to right: a skeletonized image; a cluster; a factorization component](https://gitlab.com/cossartlab/pyamnesia/-/raw/main/docs/img/overview/introduction/output_examples.png) |
|:--:|
| _From left to right: a skeletonized image; a cluster; a factorization component_ |


Whereas the skeletonization part focuses on the **morphological** aspect of the analysis, both of the clustering and factorization modules tackle the **activity analysis** of it. As shown on the diagram above, the skeleton module is prior to the two others.


### Read more

If you would like to read about the theory of the techniques we implemented or simply to know more about the project, don't hesitate to check the [GitLab repository](https://gitlab.com/cossartlab/pyamnesia) or the [documentation](https://pyamnesia.readthedocs.io) of the project.
