#!/usr/bin/env python3

"""
Usage example of the ScoreCAM calss proviede in this repository.
"""

# Import third-party packages.
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision

# Import our original packages.
from scorecam import ScoreCAM


IMAGENET1K_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET1K_STD  = np.array([0.229, 0.224, 0.225])


# [Note]
#   The following is excerption of the ImageNet labels that
#   likely to be related to 'resources/sample_image_01.jpg'.
#     * 182 = border terrier
#     * 242 = boxer
#     * 243 = bull mastiff
#     * 281 = tabby
#     * 282 = tigar cat
#     * 539 = doormat


def example01():
    """
    Example usage of ScoreCAM calss.
    """
    print("Running example01...")

    # Load a trained NN model.
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    # Prepare input image.
    image_opencv = cv.imread("resources/sample_image_01.jpg", cv.IMREAD_COLOR)
    image_opencv = cv.cvtColor(image_opencv, cv.COLOR_BGR2RGB)
    image_opencv = cv.resize(image_opencv, (224, 224), interpolation=cv.INTER_CUBIC)

    # Normalize the input image.
    # This process is required becase the torchvision models used in this example
    # assumed to be input the normalized image.
    x = (image_opencv / 255.0 - IMAGENET1K_MEAN) / IMAGENET1K_STD

    # Define the class of interest in Score-CAM.
    # (where 242 is the index of label "boxer")
    coi = 242

    # Create Score-CAM instance.
    # In this example, we use the output of "layer4" as activation maps.
    scorecam = ScoreCAM(model, actmap="layer4", device="cuda")

    # Compute visual explanation.
    L = scorecam.compute(x, coi=coi)

    # Overlay the visual explanation to the original image.
    image_overlay = scorecam.overlay(image_opencv, L)

    # Show the overlay image.
    plt.figure()
    plt.title("Example 01")
    plt.imshow(image_overlay)
    plt.show()


def example02():
    """
    Example usage of ScoreCAM calss.
    """
    print("Running example02...")

    # Load a trained NN model.
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    # Prepare input image.
    image_opencv = cv.imread("resources/sample_image_01.jpg", cv.IMREAD_COLOR)
    image_opencv = cv.cvtColor(image_opencv, cv.COLOR_BGR2RGB)
    image_opencv = cv.resize(image_opencv, (224, 224), interpolation=cv.INTER_CUBIC)

    # Normalize the input image.
    x = (image_opencv / 255.0 - IMAGENET1K_MEAN) / IMAGENET1K_STD

    # Define a function and specify it as class of interest in Score-CAM.
    # The input argment of the function is the output of the NN model
    # you've specified, and the output is the score of the model.
    # See README.md for more details.
    # (where 281 is the index of label "tabby")
    coi_fn = lambda output: output[:, 281]

    # Create Score-CAM instance.
    scorecam = ScoreCAM(model, actmap="layer4", device="cuda")

    # Compute visual explanation.
    L = scorecam.compute(x, coi=coi_fn)

    # Overlay the visual explanation to the original image.
    image_overlay = scorecam.overlay(image_opencv, L)

    # Show the overlay image.
    plt.figure()
    plt.title("Example 02")
    plt.imshow(image_overlay)
    plt.show()

    # If you want to access to the normalized activation maps and
    # masked input image, access `scorecam.A_normalized` and `scorecam.M`
    # after calling `scorecam.compute` function.
    print("A.shape =", scorecam.A_normalized.shape)
    print("M.shape =", scorecam.M.shape)


if __name__ == "__main__":
    example01()
    example02()


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
