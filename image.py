import numpy as np
import numba as nb


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:, :]), nopython=True, fastmath=True)
def colour_distance_jit(color: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Calculate perseptive color distance between a color and a set of other colors.
    Algorithm from compuphase.com/cmetric.htm modified to work on an entire palette of colors.

    :param color: RGB color to calculate the distance from
    :param palette: Array of RGB colors to calculate the distance to
    :return: Array of distances
    """
    rmeans = 0.5 * (color[0] + palette[:, 0])
    rmeans = rmeans.reshape(len(rmeans), 1)
    t = rmeans * np.array([1, 0, -1]) + np.array([2, 4, 3])
    delta_rgb = (color - palette) ** 2
    return np.sqrt(np.sum(t * delta_rgb, 1))


@nb.jit(nb.typeof(None)(nb.float64[:, :, :], nb.float64[:], nb.float64[:], nb.int32, nb.int32),
        nopython=True, fastmath=True)
def dither(arr: np.ndarray, old_pixel, new_pixel, x: int, y: int):
    """
    Floyd-Steinberg dithering for a pixel's 4 direct neighbouring pixels.

    :param arr: Image float array of shape (w x h x 3)
    :param old_pixel:
    :param new_pixel:
    :param x: Current pixel's x coordinate
    :param y: Current pixel's y coordinate
    """
    quant_error = old_pixel - new_pixel
    if x + 1 < arr.shape[1]:
        arr[y, x + 1] = np.clip(arr[y, x + 1] + quant_error * 7 / 16, 0, 1)
    if x > 0 and y + 1 < arr.shape[0]:
        arr[y + 1, x - 1] = np.clip(arr[y + 1, x - 1] + quant_error * 3 / 16, 0, 1)
    if y + 1 < arr.shape[0]:
        arr[y + 1, x] = np.clip(arr[y + 1, x] + quant_error * 5 / 16, 0, 1)
    if x + 1 < arr.shape[1] and y + 1 < arr.shape[0]:
        arr[y + 1, x + 1] = np.clip(arr[y + 1, x + 1] + quant_error * 1 / 16, 0, 1)


@nb.jit(nb.typeof(None)(nb.float64[:, :, :], nb.float64[:, :], nb.boolean), nopython=True, fastmath=True)
def reduce_color_space(arr: np.ndarray, palette: np.ndarray, use_dither: bool = True):
    """
    Reduce the color space of an image to a given palette of colors. Color values should be [0..1]

    :param arr: Image float array of shape (w x h x 3).
    :param palette: Float array of colors of shape (n x 3) that will be used in the resulting image
    :param use_dither: Apply a dithering algorithm to the resulting image
    """
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            old_pixel = np.copy(arr[y][x])
            palette_arg = np.argmin(colour_distance_jit(old_pixel, palette))
            new_pixel = palette[palette_arg]

            arr[y][x] = new_pixel

            if not use_dither:
                continue

            dither(arr, old_pixel, new_pixel, x, y)


def _demonstrate():
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open("./data/pineapple.jpg", "r")
    img = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)))
    arr = np.asarray(img) / 255

    palette = np.array([
        np.array([255, 255, 255]) / 255,
        np.array([26, 219, 235]) / 255,
        np.array([224, 154, 22]) / 255,
        np.array([37, 41, 77]) / 255,
        np.array([24, 240, 81]) / 255,
        np.array([189, 66, 0]) / 255
    ])

    reduce_color_space(arr, palette, use_dither=True)
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [10, 1, 10]})
    axs[0].imshow(np.asarray(img) / 255)
    axs[1].imshow([palette])
    axs[1].yaxis.set_visible(False)
    axs[2].imshow(arr)
    plt.show()


if __name__ == '__main__':
    _demonstrate()
