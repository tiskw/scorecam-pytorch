"""
PyTorch implementation of Score-CAM.
"""

# Import standard libraries.
import copy

# Import third-party packages.
import numpy as np
import torch


class ScoreCAM:
    """
    PyTorch implementation of Score-CAM [1].

    Example:
    >>> from torchvision.models import resnet18
    >>> from scorecam import ScoreCAM
    >>> 
    >>> # Load NN model.
    >>> model = resnet18(weights="IMAGENET1K_V1")
    >>> 
    >>> # Create Score-CAM instance.
    >>> scorecam = ScoreCAM(model, actmap="layer4")
    >>> 
    >>> # Compute visual explanation.
    >>> L = scorecam.compute(x, coi=242)
    >>> print(L)

    References:
        [1] H. Wang, Z. Wang, M. Du, F. Yang, Z. Zhang, S. Ding, P. Mardziel, and X. Hu,
            "Score-CAM: Score-weighted visual explanations for convolutional neural networks",
            CVPR, 2020.
    """
    def __init__(self, model, actmap, device="cpu"):
        """
        Constructor.

        Args:
            model  (torch.nn.Module): Target NN model.
            actmap (str)            : Name of layer to extract activation maps.
            device (str)            : Device name ("cpu" or "cuda").
        """
        # Copy the target NN model and prepare it.
        self.model = copy.deepcopy(model.to("cpu"))
        self.model.to(device)
        self.model.eval()

        # Register a hook function to extract activation maps.
        getattr(self.model, actmap).register_forward_hook(self.hook)

    def hook(self, module, x_in, x_out):
        """
        Hook function to extract activation maps.
        This function is assumed to be registered to NN layer.

        Args:
            module (torch.nn.Module): Target layer.
            x_in   (torch.Tensor)   : Input tensor of the layer.
            x_out  (torch.Tensor)   : Output tensor of the layer.
        """
        self.activation_map = x_out.detach().to("cpu")

    def compute(self, X, coi, batch_size=128, cskip=False, cskip_out=16):
        """
        Compute visual explanation.

        Args:
            X          (np.ndarray or torch.Tensor): Input image.
            coi        (int or Callable)           : Class of interest.
            batch_size (int)                       : Batch size.
            cskip      (bool)                      : Enable CSKIP optimization.
            cskip_out  (int)                       : Output channels of the CSKIP.

        Returns:
            (torch.Tensor): 2D array of visual explanation.
        """
        # Define a scoring function.
        if isinstance(coi, int):
            self.scoring_fn = lambda output: output[:, coi]
        elif hasattr(coi, "__call__"):
            self.scoring_fn = coi

        # Verify the data type of the input array.
        if (type(X) != np.ndarray) and (type(X) != torch.Tensor):
            raise TypeError("input array should be NumPy or PyTorch array.")

        # Get device.
        device = next(self.model.parameters()).device

        # Convert the input tensor to torch.Tensor on CPU.
        X = torch.Tensor(X).to("cpu")

        # Reshape the input tensor to (B, C, H, W).
        X, (B, C, H, W) = reshape_input_tensor(X)

        # Run inference and get activation maps. The activation maps are
        # acquired by the hook function registered in the __init__ function.
        with torch.no_grad():
            p = self.model(X.to(device))

        # Get the reference score.
        s_ref = self.scoring_fn(p.to("cpu"))

        # Apply CSKIP if specified.
        if cskip: A = channel_skipping(self.activation_map, cskip_out)
        else    : A = self.activation_map

        # Get the channel number of the activation maps.
        # Note that the activation maps are always on CPU.
        K = A.shape[1]

        # Upsample the activation maps and change the dimension order
        # from [1, K, H, W] to [K, 1, H, W].
        A = torch.nn.functional.interpolate(A, (H, W), mode="bicubic")
        A = torch.permute(A, [1, 0, 2, 3])

        # Normalize the activation maps.
        self.A_normalized = normalize_activation_map(A)

        # Compute the masked images.
        self.M = X * self.A_normalized

        # Initialize the list of predictions.
        batch_p = list()

        for batch_idx_bgn in range(0, K, batch_size):

            # Get input batch.
            M_batch = self.M[batch_idx_bgn:batch_idx_bgn+batch_size, :, :, :]

            # Run inference to get the predictions.
            with torch.no_grad():
                p = self.model(M_batch.to(device))

            # Add predictions to the list.
            batch_p.append(p.to("cpu"))

        p = torch.concat(batch_p, dim=0)
        A = A.to("cpu")
        X = X.to("cpu")

        # Compute the CIC score for the activation maps.
        s = self.scoring_fn(p) - s_ref
        a = torch.nn.functional.softmax(s, dim=0)
        a = a.reshape([-1, 1, 1])

        # The tensor A should have the shape [C, 1, H, W] at this moment.
        # Change the shape to [C, H, W] for the following summation.
        A = torch.squeeze(A, dim=1)

        # Compute the visual explanation.
        L = torch.nn.functional.relu(torch.sum(a * A.reshape([K, H, W]), dim=0))

        # Returns as NumPy array.
        return L.numpy()

    @staticmethod
    def to_colormap(X, normalize=True):
        """
        Convert input 2D array to a color heat map.

        Args:
            X         (np.ndarray): Input array.
            normalize (bool)      : Normalize the input array if True.

        Returns:
            (np.ndarray): Color heat map.
        """
        # Normalize if specified.
        if normalize:
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
            X = np.clip(255 * X, 0, 255).astype(np.uint8)

        # Create JET color map.
        CMAP = np.zeros([256, 3], dtype=np.uint8)
        for i in range(256):
            if   i <  32: j = i -   0; CMAP[i, :] = (      0,       0, 127+4*j)
            elif i <  96: j = i -  32; CMAP[i, :] = (      0,     4*j,     255)
            elif i < 160: j = i -  96; CMAP[i, :] = (    4*j,     255, 255-4*j)
            elif i < 224: j = i - 160; CMAP[i, :] = (    255, 255-4*j,       0)
            else        : j = i - 224; CMAP[i, :] = (255-4*j,       0,       0)

        # Apply the JET color map.
        Y = np.stack([CMAP[X, 0], CMAP[X, 1], CMAP[X, 2]], axis=2)

        return Y

    @staticmethod
    def overlay(image, L):
        """
        Overlay the given visual explanation to the given image.

        Args:
            image (np.ndarray or torch.Tensor): Input image with shape (H, W, C).
            L     (np.ndarray or torch.Tensor): Visual explanation with shape (H, W).

        Returns:
            (np.ndarray): Overlay image with shape (H, W, C).
        """
        # Convert the image and the visual explanation to NumPy array.
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(L, torch.Tensor):
            L = L.numpy()

        M = ScoreCAM.to_colormap(L)
        return np.clip(0.5 * image + 0.5 * M, 0, 255).astype(np.uint8)


def reshape_input_tensor(X):
    """
    Reshape input tensor to shape (B, C, H, W) where B is always 1.

    Args:
        X (torch.Tensor): Input array.

    Returns:
        (tuple): A tuple of (reshaped array, shape of the array).
    """
    # Case 1: X.shape == (H, W)
    if len(X.shape) == 2:

        # Get array shape.
        H, W = X.shape

        # Reshape to (B, C, H, W).
        X = X.reshape((1, 1, H, W))

        return (X, (1, 1, H, W))

    # Case 2: X.shape == (H, W, C)
    elif len(X.shape) == 3:

        # Get array shape.
        H, W, C = X.shape

        # Reshape to (B, C, H, W).
        X = torch.permute(X, [2, 0, 1])
        X = X.reshape((1, C, H, W))

        return (X, (1, C, H, W))

    # Case 3: X.shape == (1, C, H, W)
    elif len(X.shape) == 4:

        # Verify the batch size is one.
        if X.shape[0] != 1:
            raise ValueError("Batch size should be 1")

        return (X, X.shape)

    else:
        raise ValueError("Unexpected input shape: %s" % str(X.shape))


def normalize_activation_map(A, eps=1.0E-10):
    """
    Normalize the given activation map.

    Args:
        A   (np.ndarray): Activation map with shape (K, 1, H, W).
        eps (float)     : Small value to prevent zero division.

    Returns:
        (np.ndarray): Normalized activation map.
    """
    # Compute min/max of each channel.
    A_min, _ = torch.min(torch.flatten(A, start_dim=2), dim=2)
    A_max, _ = torch.max(torch.flatten(A, start_dim=2), dim=2)

    # Reshape the min/max array to the shape (K, 1, 1, 1).
    A_min = A_min.reshape([-1, 1, 1, 1])
    A_max = A_max.reshape([-1, 1, 1, 1])

    # Normalize the activation map.
    return (A - A_min) / (A_max - A_min + eps)


def channel_skipping(A, cskip_out=16):
    """
    CSKIP optimization.

    Args:
        A         (torch.Tensor): Activation maps.
        cskip_out (int)         : Number of output channels.
    """
    # Compute max value of each channel.
    A_max, _ = torch.max(torch.flatten(A, start_dim=2), dim=2)

    # Sort channels by the max value.
    idx_max = torch.argsort(A_max.reshape([-1]), descending=True)

    return A[:, idx_max[:cskip_out], :, :]


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
