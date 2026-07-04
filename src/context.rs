use cust::prelude::*;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::os::raw::c_void;

use crate::ffi;

pub fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

pub struct GpuContext {
    _ctx: Context,
    pub stream: Stream,
    pub module: Module,
    pub cublas_handle: ffi::CublasHandle,
    pub cusolver_handle: ffi::CusolverDnHandle,
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasDestroy_v2(self.cublas_handle);
            ffi::cusolverDnDestroy(self.cusolver_handle);
        }
    }
}

thread_local! {
    static TLS_CTX: RefCell<Option<GpuContext>> = RefCell::new(None);
}

pub fn with_gpu_context<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&GpuContext) -> PyResult<R>,
{
    TLS_CTX.with(|tls| {
        let mut opt_ctx = tls.borrow_mut();
        if opt_ctx.is_none() {
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
                    cublas_handle,
                    cusolver_handle,
                })
            })();
            *opt_ctx = Some(created?);
        }
        f(opt_ctx.as_ref().unwrap())
    })
}
