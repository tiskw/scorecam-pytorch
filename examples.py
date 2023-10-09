#!/usr/bin/env python3

"""
Usage example of the ScoreCAM calss proviede in this repository.
"""

# Import standard libraries.
import time

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


class Timer:
    """
    Class to measure processing time.
    """
    def __init__(self, n_trial):
        """
        """
        self.n_trial = n_trial

    def __enter__(self):
        """
        A member function called when entering `with` sentense.
        """
        # Get start time.
        self.t_start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        """
        A member function called when exiting `with` sentense.
        """
        # Get end time.
        self.t_end = time.time()

        # Compute elapsed time.
        self.elapsed_time  = self.t_end - self.t_start

        print("Elapsed time: ")


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


def example03():
    """
    Measure computation time of ScoreCAM.
    """
    print("Running example03...")

    # Load a trained NN model.
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    # Prepare input image.
    image_opencv = cv.imread("resources/sample_image_01.jpg", cv.IMREAD_COLOR)
    image_opencv = cv.cvtColor(image_opencv, cv.COLOR_BGR2RGB)
    image_opencv = cv.resize(image_opencv, (224, 224), interpolation=cv.INTER_CUBIC)

    # Normalize the input image.
    x = (image_opencv / 255.0 - IMAGENET1K_MEAN) / IMAGENET1K_STD

    # Define the number of trials.
    n_trial = 10

    # Initialize output variable.
    outputs = dict()

    for device in ["cpu", "cuda"]:
        for enable_cskip in [False, True]:

            # Initialize a list of elapsed times.
            times = list()

            # Create Score-CAM instance.
            # In this example, we use the output of "layer4" as activation maps.
            scorecam = ScoreCAM(model, actmap="layer4", device=device)

            # Compute ScoreCAM without CSkip optimization.
            for _ in range(n_trial):

                # Get start time.
                t_start = time.time()

                # Compute visual explanation.
                L = scorecam.compute(x, coi=242, cskip=enable_cskip, cskip_out=16)

                # Compute elapsed time and append it to the list.
                times.append(time.time() - t_start)

            # Store the visual explanation to the output variable.
            outputs[(device, enable_cskip)] = scorecam.overlay(image_opencv, L)

            # Print the elapsed time.
            t_mean = np.mean(times)
            t_std  = np.std(times)
            print(f"{t_mean:.3f} [sec] (std={t_std:.3f}), CSKIP={enable_cskip:d}, device={device}")

    # Show the overlay images.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input image")
    plt.imshow(image_opencv)
    plt.subplot(1, 3, 2)
    plt.title("ScoreCAM of CoI 242 (boxer) without CSKIP")
    plt.imshow(outputs[("cuda", False)])
    plt.subplot(1, 3, 3)
    plt.title("ScoreCAM of CoI 242 (boxer) with CSKIP")
    plt.imshow(outputs[("cuda", True)])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example01()
    example02()
    example03()


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
