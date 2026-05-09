import numpy as np
from skimage.morphology import dilation, closing
from skimage import measure

def arc_mask_from(snr_map, threshold=3.0, ignor_size=25, ext_size=5, close_size=3):
    """
    Generate a clean arc mask from an SNR map using robust morphological operations.

    Steps:
    1. Threshold the SNR map to create a initial binary mask.
    2. Remove small isolated islands (noise) using connected component analysis.
    3. Apply morphological closing to fill small internal holes and bridge tiny gaps.
    4. Dilate the mask to ensure full coverage of the lensed features.

    Parameters
    ----------
    snr_map : np.ndarray
        2D array of signal-to-noise ratio.
    threshold : float
        SNR threshold for masking.
    ignor_size : int
        Minimum number of connected pixels to keep.
    ext_size : int
        Size of the dilation kernel.
    close_size : int
        Size of the footprint for morphological closing (hole filling).

    Returns
    -------
    mask : np.ndarray
        The resulting boolean mask, where True indicates pixels to be EXCLUDED
        (i.e., pixels that do NOT belong to the arc).
    """
    # Step 1: Binary thresholding
    bool_map = (snr_map > threshold)

    # Step 2: Remove small islands
    labels = measure.label(bool_map)
    label_sizes = np.bincount(labels.ravel())
    
    signal_mask = np.copy(bool_map)
    for label_idx, size in enumerate(label_sizes):
        if label_idx > 0 and size < ignor_size:
            signal_mask[labels == label_idx] = False

    # Step 3: Morphological Closing
    # Fills small holes and bridges tiny gaps in curved arcs.
    # Uses a square footprint of size 'close_size'.
    if close_size > 0:
        signal_mask = closing(signal_mask, footprint=np.ones((close_size, close_size)))

    # Step 4: Dilation
    # Final expansion to ensure the mask safely covers all relevant signal.
    if ext_size > 0:
        signal_mask = dilation(signal_mask, footprint=np.ones((ext_size, ext_size)))
    
    # Invert the mask: TinyLensGpu convention uses True for EXCLUDED pixels.
    # The signal_mask identifies the arc (signal), so we return its inverse.
    return ~signal_mask