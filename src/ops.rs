use cust::prelude::*;
use cust::memory::{DeviceBuffer, DevicePointer, DeviceMemory};
use pyo3::prelude::*;
use std::os::raw::{c_int, c_void};

use crate::ffi;
use crate::context::{get_gpu_context, to_py_err};

pub fn device_ptr_from_u64(ptr: u64) -> DevicePointer<f32> {
    DevicePointer::from_raw(ptr)
}

/// Shared-memory tiled matrix transpose.
/// Block: (32,8), Grid: (ceil(cols/32), ceil(rows/32)).
#[pyfunction]
pub fn gpu_transpose(in_ptr: u64, out_ptr: u64, m: usize, n: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let func = ctx.module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 31) / 32, (m as u32 + 31) / 32, 1);
    let block_dims = (32u32, 8u32, 1u32);
    let stream = &ctx.stream;
    unsafe {
        cust::launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            device_ptr_from_u64(in_ptr), device_ptr_from_u64(out_ptr), m as i32, n as i32
        )).map_err(to_py_err)?;
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// Matrix inversion via LU decomposition (utility function).
#[pyfunction]
pub fn gpu_inverse(a_ptr: u64, inv_ptr: u64, n: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let mut identity = vec![0f32; n * n];
    for i in 0..n { identity[i * n + i] = 1.0f32; }
    let inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;

    unsafe {
        let handle = ctx.cusolver_handle.0;
        let ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        let a_dev_ptr = a_ptr as *mut f32;

        crate::cuda_check!(ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork), "Sgetrf_bufferSize");
        let work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;

        crate::cuda_check!(ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut i32, info_dev.as_raw_ptr() as *mut i32), "Sgetrf");
        crate::cuda_check!(ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev_ptr as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const i32, inv_dev_temp.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut i32), "Sgetrs");

        let mut inv_out_slice = cust::memory::DeviceSlice::from_raw_parts_mut(device_ptr_from_u64(inv_ptr), n * n);
        inv_dev_temp.copy_to(&mut inv_out_slice).map_err(to_py_err)?;
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// Solve Ridge regression via Cholesky: θ = (XᵀX + αI)⁻¹ Xᵀy.
/// Uses Cholesky factorization (Spotrf/Spotrs)  ~2x faster than LU
/// Falls back to LU when alpha=0 (matrix may not be positive-definite).
#[pyfunction]
pub fn solve_normal_equation_device(
    x_ptr: u64, y_ptr: u64, theta_ptr: u64,
    samples: usize, features: usize, alpha_reg: f32,
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let one = 1.0f32;
    let zero = 0.0f32;
    let handle_blas = ctx.cublas_handle.0;
    let handle_solver = ctx.cusolver_handle.0;
    let f = features as i32;
    let s = samples as i32;
    let stream = &ctx.stream;

    let x_dev = x_ptr as *const c_void;
    let y_dev = y_ptr as *const c_void;
    let theta_dev = theta_ptr as *mut c_void;

    unsafe {
        // 1. XtX = Xᵀ X  (cuBLAS SGEMM)
        let xtx = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let xtx_ptr = xtx.as_raw_ptr() as *mut c_void;
        crate::cuda_check!(ffi::cublasSgemm_v2(
            handle_blas, ffi::CUBLAS_OP_N, ffi::CUBLAS_OP_T,
            f, f, s, &one, x_dev, f, x_dev, f, &zero, xtx_ptr, f
        ), "SGEMM XtX");

        // 2. XtX += αI  (skip bias term)
        let effective_alpha = if alpha_reg > 0.0 { alpha_reg } else { 1e-10f32 };
        let add_reg_func = ctx.module.get_function("add_regularization_term").map_err(to_py_err)?;
        let block = 256u32;
        let grid = (features as u32 + block - 1) / block;
        cust::launch!(add_reg_func<<<grid, block, 0, stream>>>(
            xtx.as_device_ptr(), effective_alpha, f
        )).map_err(to_py_err)?;

        // 3. Xᵀy → write directly into theta output buffer
        crate::cuda_check!(ffi::cublasSgemv_v2(
            handle_blas, ffi::CUBLAS_OP_N, f, s, &one,
            x_dev, f, y_dev, 1, &zero, theta_dev, 1
        ), "SGEMV Xty");

        // 4. Cholesky factorize XtX = LLᵀ
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        crate::cuda_check!(ffi::cusolverDnSpotrf_bufferSize(
            handle_solver, ffi::CUBLAS_FILL_MODE_LOWER,
            f, xtx.as_raw_ptr() as *mut f32, f, &mut lwork
        ), "Spotrf_bufferSize");
        let work = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;

        crate::cuda_check!(ffi::cusolverDnSpotrf(
            handle_solver, ffi::CUBLAS_FILL_MODE_LOWER,
            f, xtx.as_raw_ptr() as *mut f32, f,
            work.as_raw_ptr() as *mut f32, lwork,
            info_dev.as_raw_ptr() as *mut c_int
        ), "Spotrf");

        // 5. Solve XtX · θ = Xᵀy  (in-place on theta)
        crate::cuda_check!(ffi::cusolverDnSpotrs(
            handle_solver, ffi::CUBLAS_FILL_MODE_LOWER,
            f, 1, xtx.as_raw_ptr() as *const f32, f,
            theta_dev as *mut f32, f,
            info_dev.as_raw_ptr() as *mut c_int
        ), "Spotrs");
    }

    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// GPU-accelerated prediction: out = X·coef + intercept, optionally sigmoid.
/// `out` buffer must be pre-allocated on device (size = samples).
#[pyfunction]
pub fn gpu_predict(
    x_ptr: u64, coef_ptr: u64, out_ptr: u64,
    samples: usize, features: usize,
    intercept: f32, apply_sigmoid: bool,
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let stream = &ctx.stream;
    let one = 1.0f32;
    let zero = 0.0f32;

    unsafe {
        // out = Xᵀ_col · coef  (row-major X viewed as col-major Xᵀ)
        crate::cuda_check!(ffi::cublasSgemv_v2(
            ctx.cublas_handle.0, ffi::CUBLAS_OP_T,
            features as i32, samples as i32, &one,
            x_ptr as *const c_void, features as i32,
            coef_ptr as *const c_void, 1,
            &zero, out_ptr as *mut c_void, 1
        ), "SGEMV predict");

        // out += intercept
        let bias_f = ctx.module.get_function("bias_add").map_err(to_py_err)?;
        let block = 256u32;
        let grid = (samples as u32 + block - 1) / block;
        cust::launch!(bias_f<<<grid, block, 0, stream>>>(
            device_ptr_from_u64(out_ptr), intercept, samples as i32
        )).map_err(to_py_err)?;

        // optional sigmoid
        if apply_sigmoid {
            let sig_f = ctx.module.get_function("sigmoid_transform").map_err(to_py_err)?;
            cust::launch!(sig_f<<<grid, block, 0, stream>>>(
                device_ptr_from_u64(out_ptr), samples as i32
            )).map_err(to_py_err)?;
        }
    }

    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// Mini-batch logistic regression training with momentum.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn train_logistic_minibatch_gpu(
    x_ptr: u64, y_ptr: u64, theta_ptr: u64, velocity_ptr: u64,
    samples: usize, features: usize, epochs: usize,
    lr: f32, batch_size: usize, alpha_reg: f32,
    penalty_type: i32, momentum: f32,
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let stream = &ctx.stream;

    let fused_sigmoid_sub_f = ctx.module.get_function("fused_sigmoid_sub").map_err(to_py_err)?;
    let kernel_name = if penalty_type == 1 { "l1_regularized_update" } else { "l2_regularized_update" };
    let reg_update_f = ctx.module.get_function(kernel_name).map_err(to_py_err)?;

    let x_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(x_ptr);
    let y_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(y_ptr);
    let theta_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(theta_ptr);
    let velocity_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(velocity_ptr);

    let (z_dev, error_dev, grad_dev) = unsafe {(
        DeviceBuffer::<f32>::uninitialized(batch_size).map_err(to_py_err)?,
        DeviceBuffer::<f32>::uninitialized(batch_size).map_err(to_py_err)?,
        DeviceBuffer::<f32>::uninitialized(features).map_err(to_py_err)?,
    )};

    let one = 1.0f32;
    let zero = 0.0f32;
    let f = features as i32;
    let num_batches = (samples + batch_size - 1) / batch_size;
    let sig_block = 256u32;
    let upd_block = 256u32;
    let upd_grid = (features as u32 + upd_block - 1) / upd_block;

    let mut graph: *mut c_void = std::ptr::null_mut();
    let mut graph_exec: *mut c_void = std::ptr::null_mut();

    unsafe {
        crate::cuda_check!(ffi::cuStreamBeginCapture_v2(stream.as_inner() as *mut c_void, 0), "cuStreamBeginCapture");

        for i in 0..num_batches {
            let bs = if i == num_batches - 1 { samples - i * batch_size } else { batch_size };
            let bs_i32 = bs as i32;
            let x_batch = x_dev_ptr.offset((i * batch_size * features) as isize);
            let y_batch = y_dev_ptr.offset((i * batch_size) as isize);

            // z = X_batch · θ  using SGEMM to unlock Tensor Cores
            crate::cuda_check!(ffi::cublasSgemm_v2(
                ctx.cublas_handle.0, ffi::CUBLAS_OP_T, ffi::CUBLAS_OP_N,
                bs_i32, 1, f, &one,
                x_batch.as_raw() as *const c_void, f,
                theta_dev_ptr.as_raw() as *const c_void, f,
                &zero, z_dev.as_device_ptr().as_raw() as *mut c_void, bs_i32
            ), "SGEMM z");

            // error = sigmoid(z) - y
            let sig_grid = (bs as u32 + sig_block - 1) / sig_block;
            cust::launch!(fused_sigmoid_sub_f<<<sig_grid, sig_block, 0, stream>>>(
                z_dev.as_device_ptr(), y_batch, error_dev.as_device_ptr(), bs_i32
            )).map_err(to_py_err)?;

            // grad = X_batchᵀ · error  using SGEMM to unlock Tensor Cores
            crate::cuda_check!(ffi::cublasSgemm_v2(
                ctx.cublas_handle.0, ffi::CUBLAS_OP_N, ffi::CUBLAS_OP_N,
                f, 1, bs_i32, &one,
                x_batch.as_raw() as *const c_void, f,
                error_dev.as_device_ptr().as_raw() as *const c_void, bs_i32,
                &zero, grad_dev.as_device_ptr().as_raw() as *mut c_void, f
            ), "SGEMM grad");

            // update θ with gradient + regularization + momentum
            cust::launch!(reg_update_f<<<upd_grid, upd_block, 0, stream>>>(
                theta_dev_ptr, velocity_dev_ptr,
                grad_dev.as_device_ptr(),
                lr, alpha_reg, momentum, f, bs_i32
            )).map_err(to_py_err)?;
        }

        crate::cuda_check!(ffi::cuStreamEndCapture(stream.as_inner() as *mut c_void, &mut graph), "cuStreamEndCapture");
        crate::cuda_check!(ffi::cuGraphInstantiate_v2(&mut graph_exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0), "cuGraphInstantiate");
    }

    // Fire the entire compiled epoch graph sequentially
    for _e in 0..epochs {
        unsafe {
            crate::cuda_check!(ffi::cuGraphLaunch(graph_exec, stream.as_inner() as *mut c_void), "cuGraphLaunch");
        }
    }

    unsafe {
        ffi::cuGraphExecDestroy(graph_exec);
        ffi::cuGraphDestroy(graph);
    }

    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}
