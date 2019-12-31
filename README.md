# CodeSLAM

PyTorch implementation of [CodeSLAM - Learning a Compact, Optimisable Representation for Dense Visual SLAM](https://arxiv.org/pdf/1804.00874.pdf).

## Summary

### Problems it tries to tackle/solve
- Representation of geometry in real 3D perception systems.
	- <u>Dense</u> representations, possibly augmented with semantic labels are high dimensional and unsuitable for probabilistic inference.
	- <u>Sparse</u> representations, which avoid these problems but capture only partial scene information.

### The new approach/solution
- New compact but dense representation of scene geometry, conditioned on the intensity data from a single image and generated from a code consisting of a small number of parameters.
- Each keyframe can produce a <u>depth map</u>, but the code can be optimised jointly with pose variables and with the codes of overlapping keyframes, for global consistency.

### Introduction
- As the uncertainty propagation quickly becomes intractable for large degrees of freedom, the approaches on SLAM are split into 2 categories:
	- <u>sparse</u> SLAM, representing geometry by a sparse set of features
	- <u>dense</u> SLAM, that attempts to retrieve a more complete description of the environment.
- The geometry of natural scenes exhibits a high degree of order, so we may not need a <u>large number of params</U> to represent it.
- Besides that, a scene could be <u>decomposed into a set of semantic objects</u> (e.g a chair) together with some <u>internal params</u> (e.g. size of chair, no of legs) and a pose. Other more general scene elements, which exhibit simple regularity, can be recognised and parametrised within SLAM systems.
- A straightforward AE might oversimplify the reconstruction of natural scenes, the **novelty** is to <u>condition the training on intensity images</u>.
- A **scene map** consists of a set of selected and estimated <U>historical camera poses</u>  together with the <u>corresponding captured images</U> and <u>supplementary local information</u> such as depth estimates. The intensity images are usually required for additional tasks.
- **Depth map estimate** becomes a function of <u>corresponding intensity image</u> and an unknown compact representation (referred to as **code**).
- We can think of the image providing <u>local details</u> and the code supplying more <u>global shape params</u> and can be seen as a step towards enabling optimisation in general semantic space.
- The **2 key contributions** of this paper are:
	- The derivation of a compact and optimisable representation of dense geometry by conditioning a depth autoencoder on intensity images.
	- The implementation of the first real-time targeted monocular system that achieves such a tight joint optimisation of motion and dense geometry.

## Usage

## Results

## Requirements
- Python 3.4+
- PyTorch 1.0+
- Torchvision 0.4.0+