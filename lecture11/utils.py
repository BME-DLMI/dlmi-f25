import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def standardize_orientation(im):
    """
    Standardize the orientation of a medical image
    by flipping and permuting the axis to
    achieve an identity direction cosine matrix.

    The input image must be 3D.

    Inputs:
    - im (sitk.Image): input 3D image

    Returns:
    - stand_im (sitk.Image): standardized image
    """
    ### First flip
    cosdir = np.array(im.GetDirection()).reshape(3, 3)
    amax = np.argmax(np.abs(cosdir), axis=0)
    nonzero_col = cosdir[amax, [0, 1, 2]]
    flips = (nonzero_col < 0).tolist()
    flip_im = sitk.Flip(im, flips)

    ### Next permute
    flip_im_dir = np.array(flip_im.GetDirection()).reshape(3, 3)
    p = np.argmax(flip_im_dir, axis=1)
    p = [int(i) for i in p]
    stand_im = sitk.PermuteAxes(flip_im, p)

    return stand_im


def myshow_multiview(im, slice_inds, min_intensity, max_intensity):
    """
    Displays three inline images corresponding to the anatomical planes froma 3D image.

    The input image must be 3D.

    Assumes an identity direction cosine matrix!

    Inputs:
    - im (sitk.Image): SimpleITK image to display
    - slice_inds (int, int, int): Tuple with indices of slices to display: (sagittal_ind, coronal_ind, axial_ind)
    - min_intensity (int): Minimum display intensity (mapped to black)
    - max_intensity (int): Maximum display intensity (mapped to white)

    Returns:
    """

    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Solution
    spacing = im.GetSpacing()
    size = im.GetSize()

    # Convert SimpleITK image to Numpy Array
    # solution
    imnp = sitk.GetArrayFromImage(im)

    # Display each slice
    # solution
    # extent left right bottom top
    ax[0].imshow(
        imnp[:, :, slice_inds[0]],
        vmin=min_intensity,
        vmax=max_intensity,
        cmap="gray",
        extent=[0, spacing[1] * size[1], spacing[2] * size[2], 0],
        origin="lower",
    )  # Sagittal slice
    ax[1].imshow(
        imnp[:, slice_inds[1], :],
        vmin=min_intensity,
        vmax=max_intensity,
        cmap="gray",
        extent=[0, spacing[0] * size[0], spacing[2] * size[2], 0],
        origin="lower",
    )  # Coronal slice
    ax[2].imshow(
        imnp[slice_inds[2], :, :],
        vmin=min_intensity,
        vmax=max_intensity,
        cmap="gray",
        extent=[0, spacing[0] * size[0], spacing[1] * size[1], 0],
    )  # Axial slice

    # Add title to each view
    planes = ["Sagittal", "Coronal", "Axial"]
    for i, p in enumerate(planes):
        ax[i].set_title(p)
        ax[i].axis("off")


def mse(im1, im2):
    """
    Calculates mean squared error between two numpy arrays

    Inputs:
    - im1 (np.array): Input 1
    - im2 (np.array): Input 2

    Returns:
    - error (double): mean squared error between im1 and im2
    """

    error = np.mean((im1 - im2) ** 2)
    return error
