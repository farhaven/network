use blas::dgemm;

pub enum Transpose {
    None,
    Ordinary
}

/// DGEMM
///
/// C <- a A B + b C
/// Sizes:
/// A: [m x k]
/// B: [k x n]
/// C: [m x n]
///
/// Ex: (m, n, k) = (2, 4, 3)
/// A: [2 x 3]
/// B: [3 x 4]
/// C: [2 x 4]
#[allow(non_snake_case)]
pub fn dgemm_s(m: usize, n: usize, k: usize,
                  alpha: f64, A: &Vec<f64>, B: &Vec<f64>, beta: f64, C: &mut Vec<f64>,
                  transpose_a: Transpose, transpose_b: Transpose) {
    assert_eq!(A.len(), m * k);
    assert_eq!(B.len(), k * n);
    assert_eq!(C.len(), m * n);

    /* ldX -> Stride of matrix X */
    let (lda, ta) = match transpose_a {
        Transpose::None => (m, b'N'),
        _ => (k, b'T')
    };
    let (ldb, tb) = match transpose_b {
        Transpose::None => (k, b'N'),
        _ => (n, b'T')
    };
    let ldc = m;

    unsafe {
        dgemm(ta, tb,                              /* 1 2 */
              m as i32, n as i32, k as i32,        /* 3 4 5 */
              alpha, A, lda as i32, B, ldb as i32, /* 6 7 8 9 10 */
              beta, C, ldc as i32);                /* 11 12 13 */
    }
}

#[cfg(test)]
mod test_dgemm {
    use super::*;

    #[test]
    fn test_dgemm() {
        // 4x5
        let a = vec![0.0, 1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0,
                     8.0, 9.0, 0.0, 1.0,
                     2.0, 3.0, 4.0, 5.0,
                     -1.0, -2.0, -3.0, -4.0];
        // 5x4
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0,
                     6.0, 7.0, 8.0, 9.0, 0.0,
                     -1.0, -2.1, -0.5, -3.85, -31.0,
                     2.0, 1.0, 0.0, 3.0, 2.2];
        // 4x4
        let mut c = vec![1.0; 4*4];

        let m = 4;
        let n = 4;
        let k = 5;

        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        dgemm_s(m, n, k,
                1.0, &a, &b,
                0.5, &mut c,
                Transpose::None, Transpose::None);

        println!("A: {:?}, B: {:?}, C: {:?}", a, b, c);

        let target = vec![35.5, 40.5, 15.5, 20.5,
                          110.5, 140.5, 90.5, 120.5,
                          11.399999999999999, 34.95, 63.5, 87.05,
                          8.3, 12.1, 15.899999999999999, 19.7];
        assert_eq!(c, target);
    }
}
