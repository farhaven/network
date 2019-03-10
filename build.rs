fn main() {
    println!(r"cargo:rustc-link-search=/usr/local/lib");
    println!(r"cargo:rustc-link-lib=blas");
    println!(r"cargo:rustc-link-lib=cblas");
}
