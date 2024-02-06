from typing import Tuple
from numba import prange
import numpy as np
import numba


@numba.njit(fastmath=True)
def img_to_pow_2_size(img: np.ndarray) -> np.ndarray:
    if img.ndim < 2:
        raise RuntimeError("img_to_pow_2_size:: image has to be 2-dimensional, but 1-dimensional was given...")
    rows, cols = img.shape[0], img.shape[1]
    rows2, cols2 = 2 ** int(np.log2(rows)), 2 ** int(np.log2(cols))
    if rows == rows2 and cols2 == cols:
        return img
    return img_crop(img, ((rows - rows2) >> 1, (rows + rows2) >> 1),
                    ((cols - cols2) >> 1, (cols + cols2) >> 1))


@numba.njit(fastmath=True)
def img_crop(img: np.ndarray, rows_bound: Tuple[float, float], cols_bound: Tuple[float, float]) -> np.ndarray:
    if img.ndim < 2:
        raise RuntimeError("img_crop:: image has to be 2-dimensional, but 1-dimensional was given...")
    rows, cols, _ = img.shape
    x_min = max(cols_bound[0], 0)
    x_max = max(cols_bound[0], cols)
    y_min = max(rows_bound[1], 0)
    y_max = max(rows_bound[1], rows)
    return img[y_min: y_max, x_min: x_max, :]


@numba.njit(fastmath=True)
def _fast_fourier_transform(sequence: np.ndarray) -> None:
    _n = sequence.size
    _k = _n
    theta_t = 3.14159265358979323846264338328 / _n
    phi_t = complex(np.cos(theta_t), -np.sin(theta_t))
    while _k > 1:
        _t_n = _k
        _k >>= 1
        phi_t = phi_t * phi_t
        _t = 1.0
        for _l in range(0, _k):
            for a in range(_l, _n, _t_n):
                b = a + _k
                t = sequence[a] - sequence[b]
                sequence[a] += sequence[b]
                sequence[b] = t * _t
            _t *= phi_t
    _m = int(np.log2(_n))
    for a in range(_n):
        b = a
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1))
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2))
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4))
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8))
        b = ((b >> 16) | (b << 16)) >> (32 - _m)
        if b > a:
            t = sequence[a]
            sequence[a] = sequence[b]
            sequence[b] = t


@numba.njit(fastmath=True)
def fft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x.copy() if do_copy else x
    _fast_fourier_transform(_x)
    return _x


@numba.njit(fastmath=True)
def ifft(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    _x = x.copy() if do_copy else x
    _x = _x.conjugate()
    _fast_fourier_transform(_x)
    _x = _x.conjugate()
    _x /= _x.size
    return _x


@numba.njit(parallel=True)
def fft_2d(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = img_to_pow_2_size(x.copy()) if do_copy else img_to_pow_2_size(x)
    for i in prange(_x.shape[0]):
        _x[i, :] = fft(_x[i, :])
    for i in prange(_x.shape[1]):
        _x[:, i] = fft(_x[:, i])
    return _x


@numba.njit(parallel=True)
def ifft_2d(x: np.ndarray, do_copy: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("fft2 :: x.ndim != 2")
    _x = x.copy() if do_copy else x
    for i in prange(_x.shape[0]):
        _x[i, :] = ifft(_x[i, :])
    for i in prange(_x.shape[1]):
        _x[:, i] = ifft(_x[:, i])
    return _x
