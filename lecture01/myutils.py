import SimpleITK as sitk
import matplotlib.pyplot as plt


def myshow(im, slice_index=100, min_intensity=-2000, max_intensity=0):
    """
    Displays an inline image of a slice from a 3D image.

    The input image must be 3D.

    Inputs:
    - im (sitk.Image): SimpleITK image to display
    - slice_index (int): Index of slice to display
    - min_intensity (int): Minimum display intensity (mapped to black)
    - max_intensity (int): Maximum display intensity (mapped to white)

    Returns:
    """
    spacing = im.GetSpacing()
    size = im.GetSize()

    if slice_index < 0 or slice_index > size[2]:
        raise ValueError("Slice index invalid")

    imnp = sitk.GetArrayFromImage(im)
    fig, ax = plt.subplots()

    disp = ax.imshow(
        imnp[slice_index, :, :],
        vmin=min_intensity,
        vmax=max_intensity,
        cmap="gray",
        extent=[0, spacing[0] * size[0], spacing[1] * size[1], 0],
    )

    ax.axis("off")
    plt.colorbar(disp, label="Intensity")
