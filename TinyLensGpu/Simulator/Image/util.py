import numpy as np
import jax.numpy as jnp
import scipy.signal as signal
import functools
from jax import jit 
import numba as nb


def make_grid_2d(npix, dpix, nsub=1):
    npix_eff = npix*nsub
    dpix_eff = dpix/float(nsub)
    a = np.arange(npix_eff) * dpix_eff
    x_grid, y_grid = np.meshgrid(a, a)
    shift = np.mean(x_grid)
    return x_grid - shift, y_grid - shift


@functools.partial(jit, static_argnums=(1,))
def bin_image_general(arr, sub_grid=1):
    """
    General binning function that works for both single images and batches of images.
    
    Args:
        arr: Input array. For single image: shape [n_sub_pix, n_sub_pix]
             For batch: shape [n_sub_pix, n_sub_pix, n_batch]
        sub_grid: Subgrid factor to bin by
    
    Returns:
        Binned array with reduced dimensions
    """
    if arr.ndim == 2:  # Single image case
        new_shape = (arr.shape[0] // sub_grid, arr.shape[1] // sub_grid)
        shape = (new_shape[0], sub_grid, new_shape[1], sub_grid)
        arr_reshaped = jnp.reshape(arr, shape)
        return jnp.mean(arr_reshaped, axis=(1, 3))
    else:  # Batch case
        new_shape = (arr.shape[0] // sub_grid, arr.shape[1] // sub_grid)
        shape = (new_shape[0], sub_grid, new_shape[1], sub_grid, arr.shape[2])
        arr_reshaped = jnp.reshape(arr, shape)
        return jnp.mean(arr_reshaped, axis=(1, 3))


@nb.njit
def submask_from_mask_numba(mask, nsub=2):
    npix = len(mask)
    submask = np.zeros((npix*nsub, npix*nsub), dtype=np.bool_)
    for i in range(npix):
        for j in range(npix):
            for x in range(nsub):
                for y in range(nsub):
                    submask[i*nsub+x, j*nsub+y] = mask[i, j]
    return submask


@nb.njit
def binning_matrix_from_numba(mask: np.ndarray, nsub: int = 2) -> np.ndarray:
    """
    Generate a binning matrix from a mask array.

    Args:
        mask: The input mask array.
        nsub: The number of subpixels per pixel. Default is 2.

    Returns:
        The binning matrix.

    """
    submask = submask_from_mask_numba(mask, nsub)
    indices_0_msk, indices_1_msk = np.nonzero(~mask)
    indices_0_submsk, indices_1_submsk = np.nonzero(~submask)
    npix = len(indices_0_msk)
    npix_sub = len(indices_0_submsk)

    bin_mat = np.zeros((npix, npix_sub))
    for i in range(npix):
        idx_0_msk = indices_0_msk[i]
        idx_1_msk = indices_1_msk[i]
        for x in range(nsub):
            for y in range(nsub):
                idx_0_submsk = idx_0_msk * nsub + x
                idx_1_submsk = idx_1_msk * nsub + y
                for j in range(npix_sub):
                    if indices_0_submsk[j] == idx_0_submsk and indices_1_submsk[j] == idx_1_submsk:
                        bin_mat[i, j] = 1.0 / nsub ** 2

    return bin_mat
                

def psf_matrix(psf_kernel, image_shape, mask):
    """
    psf_kernel: two array represent the psf kernel
    image_shape: the shape of image
    indice_0: the indices of unmasked (feature) region along axis-0
    indice_1: the indices of unmasked (feature) region along axis-1
    """
    indice_0, indices_1 = np.nonzero(~mask)
    n_unmasked_pix = len(indice_0)
    psf_mat = np.zeros((n_unmasked_pix, n_unmasked_pix))
    for ii in range(n_unmasked_pix):
        image_unit = np.zeros(image_shape, dtype='float')
        image_unit[indice_0[ii], indices_1[ii]] = 1.0
        image_unit = signal.fftconvolve(image_unit, psf_kernel, mode='same')
        psf_mat[:, ii] = image_unit[indice_0, indices_1]
    return psf_mat


@nb.njit
def psf_matrix_numba(psf_kernel, image_shape, mask):
    """
    psf_kernel: two array represent the psf kernel
    image_shape: the shape of image
    mask: the mask that defines the image-fitting region
    """
    psf_hw = int(psf_kernel.shape[0]/2)
    if psf_hw*2+1 != psf_kernel.shape[0]:
        raise Exception(f"The psf kernel size is: {psf_kernel.shape[0]}, not an odd number!")
    
    if not np.isclose(np.sum(psf_kernel), 1.0):
        print("The psf has not been normalized")
        print("The summed value of psf kernel is:", np.sum(psf_kernel))
        print("Normalize the psf kernel now...")
        psf_kernel = psf_kernel / np.sum(psf_kernel)

    mask_ext = np.ones((mask.shape[0]+psf_hw*2, mask.shape[1]+psf_hw*2), dtype='bool')
    mask_ext[psf_hw:-psf_hw, psf_hw:-psf_hw] = mask
    image_ext_shape = (image_shape[0]+psf_hw*2, image_shape[1]+psf_hw*2)

    indice_0, indice_1 = np.nonzero(~mask_ext)
    n_unmasked_pix = len(indice_0)
    psf_mat = np.zeros((n_unmasked_pix, n_unmasked_pix), dtype='float')

    for ii in range(n_unmasked_pix):
        image_unit = np.zeros(image_ext_shape, dtype='float')
        image_unit[indice_0[ii]-psf_hw:indice_0[ii]+psf_hw+1, indice_1[ii]-psf_hw:indice_1[ii]+psf_hw+1] = psf_kernel[:, :]
        # psf_mat[:, ii] = image_unit[indice_0, indice_1] ##numba give a error with this index scheme
        for jj in range(n_unmasked_pix):
            psf_mat[jj, ii] = image_unit[indice_0[jj], indice_1[jj]]

    return psf_mat