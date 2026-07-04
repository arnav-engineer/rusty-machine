use std::os::raw::{c_int, c_void};

#[repr(C)]
pub struct CublasContext { _private: [u8; 0] }
pub type CublasHandle = *mut CublasContext;

#[repr(C)]
pub struct CusolverDnContext { _private: [u8; 0] }
pub type CusolverDnHandle = *mut CusolverDnContext;

pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;
pub const CUBLAS_FILL_MODE_LOWER: c_int = 0;

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut CublasHandle) -> c_int;
    pub fn cublasDestroy_v2(handle: CublasHandle) -> c_int;
    pub fn cublasSetStream_v2(handle: CublasHandle, stream: *mut c_void) -> c_int;
    pub fn cublasSetMathMode(handle: CublasHandle, mode: c_int) -> c_int;

    pub fn cublasSgemv_v2(
        handle: CublasHandle, trans: c_int,
        m: c_int, n: c_int, alpha: *const f32,
        A: *const c_void, lda: c_int,
        x: *const c_void, incx: c_int,
        beta: *const f32,
        y: *mut c_void, incy: c_int,
    ) -> c_int;

    pub fn cublasSgemm_v2(
        handle: CublasHandle, transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int, alpha: *const f32,
        A: *const c_void, lda: c_int,
        B: *const c_void, ldb: c_int,
        beta: *const f32,
        C: *mut c_void, ldc: c_int,
    ) -> c_int;
}

#[link(name = "cusolver")]
extern "C" {
    pub fn cusolverDnCreate(handle: *mut CusolverDnHandle) -> c_int;
    pub fn cusolverDnDestroy(handle: CusolverDnHandle) -> c_int;
    pub fn cusolverDnSetStream(handle: CusolverDnHandle, stream: *mut c_void) -> c_int;

    // LU decomposition (fallback for alpha=0)
    pub fn cusolverDnSgetrf_bufferSize(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int) -> c_int;
    pub fn cusolverDnSgetrf(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int) -> c_int;
    pub fn cusolverDnSgetrs(handle: CusolverDnHandle, trans: c_int, n: c_int, nrhs: c_int, A: *const f32, lda: c_int, devIpiv: *const c_int, B: *mut f32, ldb: c_int, devInfo: *mut c_int) -> c_int;

    // Cholesky decomposition (primary solver for Ridge)
    pub fn cusolverDnSpotrf_bufferSize(handle: CusolverDnHandle, uplo: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int) -> c_int;
    pub fn cusolverDnSpotrf(handle: CusolverDnHandle, uplo: c_int, n: c_int, A: *mut f32, lda: c_int, workspace: *mut f32, Lwork: c_int, devInfo: *mut c_int) -> c_int;
    pub fn cusolverDnSpotrs(handle: CusolverDnHandle, uplo: c_int, n: c_int, nrhs: c_int, A: *const f32, lda: c_int, B: *mut f32, ldb: c_int, devInfo: *mut c_int) -> c_int;
}

#[link(name = "cuda")]
extern "C" {
    pub fn cuStreamBeginCapture_v2(hStream: *mut c_void, mode: c_int) -> c_int;
    pub fn cuStreamEndCapture(hStream: *mut c_void, phGraph: *mut *mut c_void) -> c_int;
    pub fn cuGraphInstantiate_v2(phGraphExec: *mut *mut c_void, hGraph: *mut c_void, phErrorNode: *mut c_void, logBuffer: *mut i8, bufferSize: usize) -> c_int;
    pub fn cuGraphLaunch(hGraphExec: *mut c_void, hStream: *mut c_void) -> c_int;
    pub fn cuGraphExecDestroy(hGraphExec: *mut c_void) -> c_int;
    pub fn cuGraphDestroy(hGraph: *mut c_void) -> c_int;
}
