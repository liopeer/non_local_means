import ctypes
from jaxtyping import Float32, jaxtyped
from numpy import ndarray
import numpy as np
from beartype import beartype as typechecker

lib = ctypes.CDLL("./non_local_means.so")

lib.interface_nlm.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

lib.interface_nlm.restype = ctypes.c_void_p

@jaxtyped
@typechecker
def non_local_means(
    img: Float32[ndarray, "h w"],
    radius: int
) -> Float32[ndarray, "h w"]:
    
    h, w = img.shape
    out = np.zeros_like(img).astype(np.float32)
    lib.interface_nlm(img, out, h, w, radius)
    return out