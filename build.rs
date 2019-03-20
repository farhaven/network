fn main() {
    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=pthread");
    // println!("cargo:rustc-link-lib=blas");
}
