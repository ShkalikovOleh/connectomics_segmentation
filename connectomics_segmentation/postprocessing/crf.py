import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    create_pairwise_bilateral,
    create_pairwise_gaussian,
    unary_from_softmax,
)


def apply_crf(
    proba: np.ndarray,
    intensities: np.ndarray,
    pos_weights: list[float] | tuple[float],
    inten_weight: float,
    compat_position: float | np.ndarray,
    compat_bilateral: float | np.ndarray,
    num_steps: int = 10,
) -> np.ndarray:
    """Apply dense fully-connected CRF to the given volume

    Explanation:
    https://proceedings.neurips.cc/paper/2011/file/beda24c1e1b46055dff2c39c98fd6fc1-Paper.pdf

    Args:
        proba (np.ndarray): probabilities of voxels to be asigned to labels

        intensities (np.ndarray): raw volume data

        pos_weight (list[float] | tuple[float]): corresponds to the theta alpha and
        beta parameter of the CRF model

        inten_weight (float): corresponds to the theta gamma parameter of
        the CRF model

        compat_position (float | np.ndarray): weight of the position term (w1) or
        compatability matrix

        compat_bilateral (float | np.ndarray): weight of the bilateral term (w2) or
        compatability matrix

    Returns:
        np.ndarray: MAP labels for the given volume
    """
    n_labels, H, W, D = proba.shape
    crf = dcrf.DenseCRF(H * W * D, n_labels)

    U = unary_from_softmax(proba)
    crf.setUnaryEnergy(U)

    position_feats = create_pairwise_gaussian(pos_weights, (H, W, D))
    crf.addPairwiseEnergy(
        position_feats,
        compat=compat_position,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    bilateral_feats = create_pairwise_bilateral(
        pos_weights, (inten_weight), img=intensities, chdim=0
    )
    crf.addPairwiseEnergy(
        bilateral_feats,
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = crf.inference(num_steps)
    labels = np.argmax(Q, axis=0)

    return labels
