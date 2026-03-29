"""
rustymachine_api.models — High-level Python API for rusty-machine.

Scikit-learn-compatible classes wrapping GPU-accelerated Rust/CUDA functions.
All heavy computation runs on the GPU; predictions can run on GPU or CPU.
"""

import numpy as np
import cupy as cp
import rusty_machine

def _safe_ptr(arr):
    """
    Ensure the array is contiguous and C-order so the pointer is safe for FFI.
    Returns (pointer as u64, array) to keep the array object alive.
    """
    arr = cp.ascontiguousarray(arr, dtype=np.float32)
    return arr.data.ptr, arr

def _to_gpu_fast(arr):
    """
    Safely stream host arrays to the GPU using Pinned Memory asynchronously.
    """
    if isinstance(arr, cp.ndarray):
        return arr
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    pinned_mem = cp.cuda.alloc_pinned_memory(arr.nbytes)
    pinned_host = np.frombuffer(pinned_mem, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
    pinned_host[...] = arr
    gpu_arr = cp.empty_like(arr)
    gpu_arr.set(pinned_host)
    return gpu_arr

def _check_gpu_memory(required_bytes):
    """
    Check if the requested allocation exceeds available VRAM.
    """
    try:
        mem_info = cp.cuda.runtime.memGetInfo()
        free_bytes = mem_info[0]
        if required_bytes > free_bytes:
            raise MemoryError(
                f"GPU VRAM Exceeded. Required: {required_bytes / 1e9:.2f} GB, "
                f"Available: {free_bytes / 1e9:.2f} GB. "
                "Consider using a smaller dataset or out-of-core training."
            )
    except cp.cuda.runtime.CUDARuntimeError:
        pass

def _pad_for_tensor_cores(X_b):
    """Pads feature dimensions to a multiple of 16 to trigger TF32 hardware."""
    samples, features = X_b.shape
    alignment = 16
    pad_size = (alignment - (features % alignment)) % alignment
    if pad_size > 0:
        X_b = cp.pad(X_b, ((0, 0), (0, pad_size)), mode='constant')
    return X_b, features + pad_size

class LinearRegression:
    """
    GPU-accelerated Ridge Regression (L2) via Cholesky-based Normal Equation.

    θ = (XᵀX + αI)⁻¹ Xᵀy  — solved via cuSOLVER Cholesky factorization.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization strength. 0.0 = standard OLS (tiny epsilon added
        internally to ensure numerical stability).
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self._coef_gpu = None

    def fit(self, X, y):
        """Fit the model on GPU using Cholesky-based Normal Equation."""
        # Move directly to GPU using fast pinned-memory streams
        X_gpu_raw = _to_gpu_fast(X)
        y_gpu_raw = _to_gpu_fast(y).ravel()

        # Augment with intercept column on GPU
        ones = cp.ones((X_gpu_raw.shape[0], 1), dtype=cp.float32)
        X_b = cp.ascontiguousarray(cp.hstack([X_gpu_raw, ones]))

        X_b_padded, features_padded = _pad_for_tensor_cores(X_b)
        samples = X_b_padded.shape[0]
        
        # Guard: check memory before moving to GPU
        required_mem = samples * features_padded * 4 + samples * 4 + features_padded * 4
        _check_gpu_memory(required_mem)

        X_gpu_ptr, X_gpu = _safe_ptr(X_b_padded)
        y_gpu_ptr, y_gpu = _safe_ptr(y_gpu_raw)
        theta_gpu_ptr, theta_gpu = _safe_ptr(cp.empty(features_padded, dtype=cp.float32))

        rusty_machine.solve_normal_equation_device(
            X_gpu_ptr, y_gpu_ptr, theta_gpu_ptr,
            samples, features_padded, self.alpha,
        )

        theta_host = theta_gpu.get()
        features_orig = X_b.shape[1]
        theta_host_unpadded = theta_host[:features_orig]

        self.intercept_ = float(theta_host_unpadded[-1])
        self.coef_ = theta_host_unpadded[:-1]
        self._coef_gpu = cp.asarray(self.coef_)
        return self

    def predict(self, X):
        """Predict class values on GPU."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_gpu_raw = _to_gpu_fast(X)
        samples, features = X_gpu_raw.shape
        out_gpu = cp.empty(samples, dtype=cp.float32)

        X_gpu_ptr, _X_guard = _safe_ptr(X_gpu_raw)
        coef_gpu_ptr, _coef_guard = _safe_ptr(self._coef_gpu)
        out_gpu_ptr, _out_guard = _safe_ptr(out_gpu)

        rusty_machine.gpu_predict(
            X_gpu_ptr, coef_gpu_ptr, out_gpu_ptr,
            samples, features, float(self.intercept_), False
        )
        return cp.asnumpy(out_gpu)


class LogisticRegression:
    """
    GPU-accelerated Logistic Regression with L1/L2 regularization and momentum.

    Trained via mini-batch SGD with optional Nesterov-style momentum.
    All training ops run on GPU via custom CUDA kernels and cuBLAS.

    Parameters
    ----------
    epochs : int
        Number of passes over the training data.
    lr : float
        Learning rate for gradient descent.
    batch_size : int
        Samples per gradient update.
    penalty : {'l1', 'l2'}
        Regularization type.
    alpha : float
        Regularization strength (≥ 0).
    momentum : float
        SGD momentum factor (0.0 = standard SGD, 0.9 = typical momentum).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, epochs=1000, lr=0.01, batch_size=256,
                 penalty='l2', alpha=0.0, momentum=0.0, random_state=None):
        if penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.momentum = momentum
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self._coef_gpu = None

    def fit(self, X, y):
        """Fit the model on GPU using mini-batch SGD with momentum."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            cp.random.seed(self.random_state)

        # Immediately transfer to GPU via Pinned memory
        X_gpu_raw = _to_gpu_fast(X)
        y_gpu_raw = _to_gpu_fast(y).ravel()

        # Augment with intercept column on GPU natively
        ones = cp.ones((X_gpu_raw.shape[0], 1), dtype=cp.float32)
        X_b = cp.hstack([X_gpu_raw, ones])

        X_b_padded, features_padded = _pad_for_tensor_cores(X_b)
        samples = X_b_padded.shape[0]

        # Shuffle data entirely on GPU VRAM using incredibly high bandwidth
        perm = cp.random.permutation(samples)
        X_shuffled = cp.ascontiguousarray(X_b_padded[perm])
        y_shuffled = cp.ascontiguousarray(y_gpu_raw[perm])

        # Guard: check memory before moving to GPU
        required_mem = samples * features_padded * 4 + samples * 4 + features_padded * 4 * 2 # X, y, theta, velocity
        _check_gpu_memory(required_mem)

        # We already have contiguous GPU arrays
        X_gpu_ptr, X_gpu = _safe_ptr(X_shuffled)
        y_gpu_ptr, y_gpu = _safe_ptr(y_shuffled)
        theta_gpu_ptr, theta_gpu = _safe_ptr(cp.zeros(features_padded, dtype=cp.float32))
        velocity_gpu_ptr, velocity_gpu = _safe_ptr(cp.zeros(features_padded, dtype=cp.float32))

        penalty_type = 1 if self.penalty == 'l1' else 2

        rusty_machine.train_logistic_minibatch_gpu(
            X_gpu_ptr, y_gpu_ptr,
            theta_gpu_ptr, velocity_gpu_ptr,
            samples, features_padded, self.epochs,
            self.lr, self.batch_size, self.alpha,
            penalty_type, self.momentum,
        )

        theta_host = theta_gpu.get()
        features_orig = X_b.shape[1]
        theta_host_unpadded = theta_host[:features_orig]

        self.intercept_ = float(theta_host_unpadded[-1])
        self.coef_ = theta_host_unpadded[:-1]
        self._coef_gpu = cp.asarray(self.coef_)
        return self

    def _forward_gpu(self, X, apply_sigmoid):
        """Internal helper to evaluate the linear forward pass cleanly."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_gpu_raw = _to_gpu_fast(X)
        samples, features = X_gpu_raw.shape
        out_gpu = cp.empty(samples, dtype=cp.float32)

        X_gpu_ptr, _X_guard = _safe_ptr(X_gpu_raw)
        coef_gpu_ptr, _coef_guard = _safe_ptr(self._coef_gpu)
        out_gpu_ptr, _out_guard = _safe_ptr(out_gpu)

        rusty_machine.gpu_predict(
            X_gpu_ptr, coef_gpu_ptr, out_gpu_ptr,
            samples, features, float(self.intercept_), apply_sigmoid
        )
        return cp.asnumpy(out_gpu)

    def predict_proba(self, X):
        """Predict class probabilities on GPU. Returns (n, 2) array."""
        probs = self._forward_gpu(X, True)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        """Predict class labels (0/1) directly on GPU without sigmoid overhead."""
        z = self._forward_gpu(X, False)
        return (z > 0).astype(np.int32)