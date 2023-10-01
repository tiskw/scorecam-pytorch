PyTorch implementation of Score-CAM
================================================================================

This repository provides an unofficial PyTorch implementation of Score-CAM [1].
Score-CAM is a CAM (class activation mapping) based visual explanation method
like Grad-CAM and Grad-CAM++, but Score-CAM does not depend on gradients and
can provide stable visual explanations.

The features of this implementation are:

* **Versatile**: The code of this repository is applicable to many types of
  neural networks, not only for the models provided by `torchvision` module
  but also for custom CNN models.
* **Portable**: This repository is easily transplanted to user projects.
  At this moment what the users need to do is just copy a single Python file
  to the user's projects.
* **Less dependent**: The core module of this repository has fewer dependencies
  for easier transplantation to user's projects. The current implementation
  depends only on `numpy` and `torch` module.

<div align="center">
  <img src="./resources/scorecam_sketch.jpg" width="960" alt="Sketch of Score-CAM" />
</div>


Installation
--------------------------------------------------------------------------------

The core module of this repository, `scorecam`, requires only NumPy and PyTorch.

```console
pip3 install numpy torch
```

Additionally, the example code `examples.py` requires OpenCV, Matplotlib
and Torchvision.

```console
pip3 install opencv-python matplotlib torchvision
```


Usage
--------------------------------------------------------------------------------

### Minimal example

```python
import numpy as np
import cv2 as cv
import torchvision

# Import ScoreCAM class.
from scorecam import ScoreCAM

# Load NN model.
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

# Load input image.
image = cv.imread("resources/sample_image_01.jpg", cv.IMREAD_COLOR)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)

# Normalize the image.
IMAGENET1K_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET1K_STD  = np.array([0.229, 0.224, 0.225])
x = (image / 255.0 - IMAGENET1K_MEAN) / IMAGENET1K_STD

# Create Score-CAM instance.
scorecam = ScoreCAM(model, actmap="layer4")

# Compute visual explanation.
# The argument 'coi' means 'class of interest' and the number 242
# is the index of the label 'boxer' (breed of dog) in ImageNet.
L = scorecam.compute(x, coi=242)
print(L)
```

### Custom scoring function

If your CNN model is not a classification network, the class of interest
does not make sense and you need a custom function for scoring in Score-CAM.
In such case, you can specify a Python function to the argument `coi`.

For example, imagine that your CNN is YOLO and outputs a tensor with shape
(B, 5 + C, H, W) where B is a batch size, C is a number of class, and
(H, W) is an output resolution. If you want to analyze the detection result
of class `C = c` at `H = h` and `W = w`, the custom function can be written
like the following:

```python
# Define a custom scoring function.
coi_fn = lambda output: output[:, c, h, w]

L = scorecam.compute(x, coi=coi_fn)
```

Note that the following code

```python
L = scorecam.compute(x, coi=target_index)
```

is equevarent with

```python
# Define a scoring function.
coi_fn = lambda output: output[:, target_index]

L = scorecam.compute(x, coi=coi_fn)
```

where `target_index` is an integer.


References
--------------------------------------------------------------------------------

[1] H. Wang, Z. Wang, M. Du, F. Yang, Z. Zhang, S. Ding, P. Mardziel, and X. Hu,
    "Score-CAM: Score-weighted visual explanations for convolutional neural networks",
    CVPR, 2020. [PDF](https://arxiv.org/abs/1910.01279)

[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra,
    "Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization",
    ICCV, 2017. [PDF](https://arxiv.org/abs/1610.02391)

[3] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. Balasubramanian,
    "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks",
    WACV, 2018. [PDF](https://arxiv.org/abs/1710.11063)

