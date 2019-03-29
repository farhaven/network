fn main() {
    #[cfg(target_os="openbsd")]
    {
        println!("cargo:rustc-link-search=/usr/local/lib");
        println!("cargo:rustc-link-lib=openblas");
    }
    #[cfg(target_os="macos")]
    {
        println!("cargo:rustc-link-search=/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework");
        println!("cargo:rustc-link-lib=blas");
    }
}
