fn main() {
    println!("CUDA ROOT: {:?}", find_cuda_helper::find_cuda_root());
    println!("CUDA LIB DIRS: {:?}", find_cuda_helper::find_cuda_lib_dirs());
}
