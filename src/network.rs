use cblas::{dgemm, Layout, Transpose};
use rand::Rng;

fn nonlinearity(z: &f64) -> f64 {
    z.tanh()
}

fn nonlinearity_prime(z: &f64) -> f64 {
    1_f64 - z.powf(2_f64)
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
unsafe fn dgemm_s(m: usize, n: usize, k: usize,
                  alpha: f64, A: &Vec<f64>, B: &Vec<f64>, beta: f64, C: &mut Vec<f64>,
                  transpose_a: Transpose, transpose_b: Transpose) {
    assert_eq!(A.len(), m * k);
    assert_eq!(B.len(), k * n);
    assert_eq!(C.len(), m * n);

    /* ldX -> Stride of matrix X */
    let lda = match transpose_a {
        Transpose::None => m,
        _ => k
    };
    let ldb = match transpose_b {
        Transpose::None => k,
        _ => n
    };
    let ldc = m; /* Is this correct? */

    dgemm(Layout::ColumnMajor, transpose_a, transpose_b, /* 0 1 2 */
          m as i32, n as i32, k as i32,                  /* 3 4 5 */
          alpha, A, lda as i32, B, ldb as i32,           /* 6 7 8 9 10 */
          beta, C, ldc as i32);                          /* 11 12 13 */
}

#[cfg(test)]
mod test_dgemm {
    use super::*;

    #[test]
    fn test_dgemm() {
        let a = vec![0_f64, 1_f64, 2_f64];
        let b = vec![1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0];
        let mut c = vec![0.0, 0.0, 0.0];

        unsafe {
            dgemm_s(2, 3, 1,
                    1.0, &a, &b, 0.0, &mut c,
                    Transpose::None, Transpose::None);
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    output: Vec<f64>,
    delta: Vec<f64>,
    weights: Vec<f64>,
    shape: (usize, usize) /* Rows x Cols */
}
impl Layer {
    pub fn new(mut inputs: usize, outputs: usize) -> Layer {
        inputs += 1; /* Bias */

        let mut rng = rand::thread_rng();
        let mut weights = Vec::<f64>::with_capacity(inputs * outputs);
        for _ in 0..weights.capacity() {
            let r: f64 = rng.gen();
            weights.push(r - 0.5_f64);
        }

        let mut output = Vec::<f64>::with_capacity(outputs);
        for _ in 0..outputs {
            output.push(0_f64);
        }

        let mut delta = Vec::<f64>::with_capacity(outputs);
        for _ in 0..outputs {
            delta.push(0_f64);
        }

        Layer{
            weights,
            output: output,
            delta: delta,
            shape: (inputs, outputs)
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        /*
         * DGEMM:
         *
         * C <- a AB + b C
         *
         * A: [m x k] -> Inputs  [1 x inputs]
         * B: [k x n] -> Weights [inputs x outputs]
         * C: [n x m] -> Outputs [outputs x 1]
         */

        let m = self.shape.1;
        let n = 1;
        let k = self.shape.0;

        let mut local_inputs = inputs.clone();
        local_inputs.push(1_f64);
        unsafe {
            dgemm_s(m, n, k,
                    1_f64, &self.weights, &local_inputs,
                    0_f64, &mut self.output,
                    Transpose::Ordinary, Transpose::Ordinary);
        }

        self.output = self.output.iter().map(nonlinearity).collect();
        self.output.clone()
    }

    pub fn compute_gradient(&mut self, error: &Vec<f64>) -> Vec<f64> {
        self.delta = Vec::<f64>::with_capacity(self.shape.1);

        for idx in 0..self.delta.capacity() {
            self.delta.push(error[idx] * nonlinearity_prime(&self.output[idx]));
        }

        let m = 1;
        let n = self.shape.0;
        let k = self.shape.1;
        let mut res = Vec::<f64>::with_capacity(self.shape.0);
        unsafe {
            res.set_len(self.shape.0);
            assert_eq!(self.delta.len(), m * k);
            assert_eq!(self.weights.len(), k * n);
            assert_eq!(res.len(), m * n);
            dgemm_s(m, n, k,
                    1_f64, &self.delta, &self.weights,
                    0_f64, &mut res,
                    Transpose::None, Transpose::Ordinary);
        }
        res
    }

    pub fn update_weights(&mut self, input: &Vec<f64>, learning_rate: f64) {
        let mut local_input = input.clone();
        local_input.push(1_f64); /* Bias */

        let alpha = learning_rate;

        let m = self.shape.0;
        let n = self.shape.1;
        let k = 1;

        unsafe {
            dgemm_s(m, n, k,
                    alpha, &local_input, &self.delta,
                    1_f64, &mut self.weights,
                    Transpose::Ordinary, Transpose::None);
        }
    }
}

#[cfg(test)]
mod test_layer {
    use super::*;

    #[test]
    fn test_init() {
        let _layer = Layer::new(3, 2);
    }

    #[test]
    fn test_forward() {
        let input = vec![-1_f64, 0_f64, 1_f64];

        let mut layer = Layer::new(3, 2);
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);
        assert_eq!(output, layer.output);
    }

    #[test]
    fn test_compute_gradient() {
        let input = vec![-1_f64, 0_f64, 1_f64];
        let error = vec![0_f64, 0.3_f64];

        let mut layer = Layer::new(3, 2);
        let output = layer.forward(&input);

        let gradient = layer.compute_gradient(&error);
        println!("Output: {:?}", output);
        println!("Gradient: {:?}", gradient);

        assert_eq!(gradient.len(), 3);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_update_weights() {
        let input = vec![-1_f64, 0_f64, 1_f64];
        let error = vec![0_f64, 0.3_f64];

        let mut layer = Layer::new(3, 2);
        let _ = layer.forward(&input);
        let _ = layer.compute_gradient(&error);
        layer.update_weights(&input, 2_f64);
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    layer_sizes: Vec<usize>
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Network {
        let mut layers = Vec::<Layer>::new();

        for idx in 0..layer_sizes.len() - 1 {
            let inputs = layer_sizes[idx];
            let outputs = layer_sizes[idx + 1];
            layers.push(Layer::new(inputs, outputs));
        }

        Network{
            layers: layers,
            layer_sizes: layer_sizes
        }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn backprop(&mut self, input: &Vec<f64>, error: &Vec<f64>, learning_rate: f64) {
        let mut local_error = error.clone();

        for lidx in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[lidx];
            local_error = layer.compute_gradient(&local_error);
        }

        let mut local_input = input.clone();

        for layer in &mut self.layers {
            layer.update_weights(&local_input, learning_rate);
            local_input = layer.output.clone();
        }
    }

    pub fn error(&self, output: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
        let mut res = Vec::<f64>::with_capacity(output.len());

        for idx in 0..res.capacity() {
            res.push(target[idx] - output[idx]);
        }

        res
    }
}

#[cfg(test)]
mod test_network {
    use super::*;

    #[test]
    fn test_forward() {
        let mut network = Network::new(vec![2, 3, 1]);
        let output = network.forward(&vec![1_f64, 0_f64]);
        println!("Output: {:?}", output);
    }

    #[test]
    fn test_backprop() {
        let mut network = Network::new(vec![2, 3, 1]);

        let input = vec![0_f64, 1_f64];
        let target = vec![1_f64];
        let output1 = network.forward(&input);
        let error1 = network.error(&output1, &target);

        println!("In: {:?}, Target: {:?}, Output: {:?}, Error: {:?}", input, target, output1, error1);

        network.backprop(&input, &error1, 1_f64);

        let output2 = network.forward(&input);
        let error2 = network.error(&output2, &target);

        println!("In: {:?}, Target: {:?}, Output: {:?}, Error: {:?}", input, target, output2, error2);

        /* Calculate squared error before and after training to assert some gradual improvement */
        let se1 = error1.iter().fold(0_f64, |acc, x| acc + x.powf(2_f64));
        let se2 = error2.iter().fold(0_f64, |acc, x| acc + x.powf(2_f64));

        assert!(se2 <= se1); /* Assert _some_ improvement */
    }

    #[test]
    fn test_xor() {
        let mut network = Network::new(vec![2, 2, 1]);
        let samples = vec![(vec![0_f64, 0_f64], vec![0_f64]),
                           (vec![0_f64, 1_f64], vec![1_f64]),
                           (vec![1_f64, 0_f64], vec![1_f64]),
                           (vec![1_f64, 1_f64], vec![0_f64])];

        let target_mse = 0.005;
        let mut learning_rate = 1_f64;
        let mut errors: Vec<f64> = vec![];

        let mut iter = 0;
        loop {
            iter += 1;

            for (input, target) in &samples {
                let output = network.forward(input);
                let error = network.error(&output, target);
                network.backprop(input, &error, learning_rate);
                errors.push(error.iter().fold(0_f64, |acc, x| acc + x.powf(2_f64)) / (error.len() as f64));
            }

            /* Report MSE every 100 iterations */
            if iter % 100 == 0 {
                let mse = errors.iter().fold(0_f64, |acc, x| acc + x) / (errors.len() as f64);
                println!("Iter: {}, MSE: {}", iter, mse);
                if mse <= target_mse {
                    println!("Reached target MSE after {} iterations", iter);
                    break;
                }
                learning_rate = mse.sqrt();
            }
        }

        for (input, target) in &samples {
            let output = network.forward(input);
            let rounded_output: Vec<f64> = output.iter().map(|&x| x.round()).collect();
            println!("In: {:?}, Out: {:?}, Target: {:?}, Round: {:?}", input, output, target, rounded_output);
            assert_eq!(&rounded_output, target);
        }
    }
}
