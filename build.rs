// Build script: compiles CUDA kernels to PTX and links NVIDIA libraries.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Link NVIDIA CUDA libraries
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-lib=cublas");

    // Search paths — support both /usr/local/cuda and system-wide installs
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");

    // Determine CUDA architecture from env or default based on nvidia-smi
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| {
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
        {
            if output.status.success() {
                if let Ok(cap_str) = String::from_utf8(output.stdout) {
                    if let Some(cap) = cap_str.trim().split('\n').next() {
                        return format!("sm_{}", cap.trim().replace('.', ""));
                    }
                }
            }
        }
        "sm_86".to_string()
    });
    eprintln!("Building CUDA kernels for arch: {}", arch);

    let status = Command::new("nvcc")
        .arg("-O3")
        .arg(format!("-arch={}", arch))
        .arg("-ptx")
        .arg("-o").arg("src/kernels.ptx")
        .arg("src/kernels.cu")
        .status()
        .expect("Failed to execute nvcc. Ensure CUDA toolkit is installed and nvcc is in PATH.");

    if !status.success() {
        panic!("nvcc failed to compile CUDA kernels.");
    }
}