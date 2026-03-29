use cust::prelude::*;
use pyo3::prelude::*;
use std::sync::OnceLock;
use std::os::raw::c_void;

use crate::ffi;

pub fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// Thread-safe handle wrappers
pub struct SyncCublasHandle(pub ffi::CublasHandle);
unsafe impl Send for SyncCublasHandle {}
unsafe impl Sync for SyncCublasHandle {}

pub struct SyncCusolverHandle(pub ffi::CusolverDnHandle);
unsafe impl Send for SyncCusolverHandle {}
unsafe impl Sync for SyncCusolverHandle {}

pub struct GpuContext {
    _ctx: Context,
    pub stream: Stream,
    pub module: Module,
    pub cublas_handle: SyncCublasHandle,
    pub cusolver_handle: SyncCusolverHandle,
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasDestroy_v2(self.cublas_handle.0);
            ffi::cusolverDnDestroy(self.cusolver_handle.0);
        }
    }
}

static GLOBAL_CTX: OnceLock<GpuContext> = OnceLock::new();

pub fn get_gpu_context() -> PyResult<&'static GpuContext> {
    if let Some(ctx) = GLOBAL_CTX.get() {
        return Ok(ctx);
    }
    let created = (|| -> Result<GpuContext, PyErr> {
        cust::init(CudaFlags::empty()).map_err(to_py_err)?;
        let device = Device::get_device(0).map_err(to_py_err)?;
        let ctx = Context::new(device).map_err(to_py_err)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(to_py_err)?;
        let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;

        let mut cublas_handle: ffi::CublasHandle = std::ptr::null_mut();
        unsafe {
            crate::cuda_check!(ffi::cublasCreate_v2(&mut cublas_handle), "cublasCreate");
            crate::cuda_check!(ffi::cublasSetStream_v2(cublas_handle, stream.as_inner() as *mut c_void), "cublasSetStream");
            crate::cuda_check!(ffi::cublasSetMathMode(cublas_handle, 1), "cublasSetMathMode");
        }

        let mut cusolver_handle: ffi::CusolverDnHandle = std::ptr::null_mut();
        unsafe {
            crate::cuda_check!(ffi::cusolverDnCreate(&mut cusolver_handle), "cusolverDnCreate");
            crate::cuda_check!(ffi::cusolverDnSetStream(cusolver_handle, stream.as_inner() as *mut c_void), "cusolverDnSetStream");
        }

        Ok(GpuContext {
            _ctx: ctx, stream, module,
            cublas_handle: SyncCublasHandle(cublas_handle),
            cusolver_handle: SyncCusolverHandle(cusolver_handle),
        })
    })();
    match created {
        Ok(ctx_val) => {
            let _ = GLOBAL_CTX.set(ctx_val);
            Ok(GLOBAL_CTX.get().expect("GLOBAL_CTX set but not present"))
        }
        Err(e) => Err(e),
    }
}
