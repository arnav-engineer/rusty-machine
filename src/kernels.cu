// kernels.cu — Custom CUDA kernels for rusty-machine
// Optimized for high-throughput GPU computation

#define TILE_DIM 32
#define BLOCK_ROWS 8

/**
 * Shared-memory tiled matrix transpose with bank-conflict avoidance.
 * Uses TILE_DIM+1 padding to prevent shared memory bank conflicts.
 * Each thread block handles a TILE_DIM x TILE_DIM tile, with each
 * thread processing TILE_DIM/BLOCK_ROWS elements.
 * Block dims: (32, 8). Grid dims: (ceil(cols/32), ceil(rows/32)).
 */
extern "C" __global__ void transpose(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory into shared memory tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_in < cols && (y_in + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y_in + j) * cols + x_in];
        }
    }

    __syncthreads();

    // Coalesced write from shared memory tile to global memory (transposed)
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_out < rows && (y_out + j) < cols) {
            output[(y_out + j) * rows + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * Fused sigmoid activation and label subtraction.
 * error = sigmoid(logit) - y_true
 * Clamping prevents exp() overflow/underflow.
 */
extern "C" __global__ void fused_sigmoid_sub(
    const float* __restrict__ logits,
    const float* __restrict__ y_true,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = fmaxf(-20.0f, fminf(20.0f, logits[idx]));
        // Use fast-math intrinsic __expf and __fdividef
        float sigmoid = __fdividef(1.0f, 1.0f + __expf(-z));
        output[idx] = sigmoid - y_true[idx];
    }
}

/**
 * Adds L2 regularization (alpha) to the diagonal of a square matrix.
 * Skips the last diagonal element (bias/intercept term).
 */
extern "C" __global__ void add_regularization_term(
    float* __restrict__ matrix,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        matrix[idx * n + idx] += alpha;
    }
}

/**
 * L2 (Ridge) regularized gradient update with momentum.
 *   v = momentum * v_old + grad/bs + (alpha/bs) * theta   (for non-bias)
 *   theta -= lr * v
 * When momentum=0, degrades to standard SGD.
 */
extern "C" __global__ void l2_regularized_update(
    float* __restrict__ theta,
    float* __restrict__ velocity,
    const float* __restrict__ grad,
    float lr,
    float alpha_reg,
    float momentum,
    int features,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        float inv_batch_size = __fdividef(1.0f, (float)batch_size);
        float grad_component = grad[idx] * inv_batch_size;

        float reg_component = 0.0f;
        if (idx < features - 1) {
            reg_component = alpha_reg * inv_batch_size * theta[idx];
        }

        float v = momentum * velocity[idx] + grad_component + reg_component;
        velocity[idx] = v;
        theta[idx] -= lr * v;
    }
}

/**
 * L1 (Lasso) proximal gradient update with momentum.
 *   v = momentum * v_old + grad/bs
 *   theta_temp = theta - lr * v
 *   theta = soft_threshold(theta_temp, lr * alpha / bs)   (for non-bias)
 */
extern "C" __global__ void l1_regularized_update(
    float* __restrict__ theta,
    float* __restrict__ velocity,
    const float* __restrict__ grad,
    float lr,
    float alpha_reg,
    float momentum,
    int features,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        float inv_batch_size = __fdividef(1.0f, (float)batch_size);
        float grad_component = grad[idx] * inv_batch_size;

        float v = momentum * velocity[idx] + grad_component;
        velocity[idx] = v;

        float theta_new = theta[idx] - lr * v;

        if (idx < features - 1) {
            float threshold = lr * alpha_reg * inv_batch_size;
            if (theta_new > threshold) {
                theta[idx] = theta_new - threshold;
            } else if (theta_new < -threshold) {
                theta[idx] = theta_new + threshold;
            } else {
                theta[idx] = 0.0f;
            }
        } else {
            theta[idx] = theta_new;
        }
    }
}

/**
 * Element-wise sigmoid transformation (in-place).
 * Used for GPU-side logistic prediction.
 */
extern "C" __global__ void sigmoid_transform(
    float* __restrict__ data,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = fmaxf(-20.0f, fminf(20.0f, data[idx]));
        data[idx] = __fdividef(1.0f, 1.0f + __expf(-z));
    }
}

/**
 * Element-wise bias (intercept) addition.
 */
extern "C" __global__ void bias_add(
    float* __restrict__ output,
    float bias,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] += bias;
    }
}