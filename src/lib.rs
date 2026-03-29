//! Rusty Machine — GPU-accelerated ML via Rust, CUDA, cuBLAS, cuSOLVER.
//! Provides Python bindings through PyO3.

pub mod ffi;
pub mod context;
pub mod ops;

/// Status-checking macro for cuBLAS/cuSOLVER FFI calls.
#[macro_export]
macro_rules! cuda_check {
    ($call:expr, $name:expr) => {{
        let status = $call;
        if status != 0 {
            return Err(crate::context::to_py_err(format!("{} failed with status {}", $name, status)));
        }
    }};
}

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ops::*;

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------
#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_normal_equation_device, m)?)?;
    m.add_function(wrap_pyfunction!(train_logistic_minibatch_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_predict, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_inverse, m)?)?;
    Ok(())
}