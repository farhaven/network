use crate::matrixmath::{dgemm_s, Transpose};
use crate::nonlinearity::Nonlinearity;

use rand::distributions::{Normal, Distribution};

#[derive(Debug)]
pub struct Neuronal {
    pub output: Vec<f64>,
    delta: Vec<f64>,
    weights: Vec<f64>,
    shape: (usize, usize), /* Rows x Cols */
    nonlinearity: Nonlinearity
}
impl Neuronal {
    pub fn new(mut inputs: usize, outputs: usize, nonlinearity: Nonlinearity) -> Neuronal {
        inputs += 1;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0);

        let mut weights = Vec::<f64>::with_capacity(inputs * outputs);
        for _ in 0..weights.capacity() {
            let r: f64 = dist.sample(&mut rng);
            // let w = r * (2.0 / (inputs + outputs) as f64).sqrt();
            let w = r * (1.0 / (inputs + outputs) as f64).sqrt();
            weights.push(w);
        }

        let mut output = Vec::<f64>::with_capacity(outputs);
        for _ in 0..outputs {
            output.push(0_f64);
        }

        let mut delta = Vec::<f64>::with_capacity(outputs);
        for _ in 0..outputs {
            delta.push(0_f64);
        }

        Neuronal{
            weights,
            output: output,
            delta: delta,
            shape: (inputs, outputs),
            nonlinearity
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let m = self.shape.1;
        let n = 1;
        let k = self.shape.0;

        let mut local_input = inputs.clone();
        local_input.push(1_f64);

        dgemm_s(m, n, k,
                1_f64, &self.weights, &local_input,
                0_f64, &mut self.output,
                Transpose::Ordinary, Transpose::Ordinary);

        self.output = self.output.iter().map(|x| self.nonlinearity.forward(x)).collect();
        self.output.clone()
    }

    pub fn compute_gradient(&mut self, error: &Vec<f64>) -> Vec<f64> {
        self.delta = Vec::<f64>::with_capacity(self.shape.1);

        for idx in 0..self.delta.capacity() {
            self.delta.push(error[idx] * self.nonlinearity.backward(&self.output[idx]));
        }

        let m = 1;
        let n = self.shape.0;
        let k = self.shape.1;
        let mut res = Vec::<f64>::with_capacity(self.shape.0);
        unsafe { res.set_len(self.shape.0); }
        assert_eq!(self.delta.len(), m * k);
        assert_eq!(self.weights.len(), k * n);
        assert_eq!(res.len(), m * n);
        dgemm_s(m, n, k,
                1_f64, &self.delta, &self.weights,
                0_f64, &mut res,
                Transpose::None, Transpose::Ordinary);
        res
    }

    pub fn update_weights(&mut self, input: &Vec<f64>, learning_rate: f64) {
        let alpha = learning_rate;

        let m = self.shape.0;
        let n = self.shape.1;
        let k = 1;

        let mut local_input = input.clone();
        local_input.push(1_f64);

        dgemm_s(m, n, k,
                alpha, &local_input, &self.delta,
                1_f64, &mut self.weights,
                Transpose::Ordinary, Transpose::None);
    }
}

#[cfg(test)]
mod test_neuronal {
    use super::*;

    #[test]
    fn test_init() {
        let _layer = Neuronal::new(3, 2, Nonlinearity::Tanh);
    }

    #[test]
    fn test_forward() {
        let input = vec![-1_f64, 0_f64, 1_f64];

        let mut layer = Neuronal::new(3, 2, Nonlinearity::Tanh);
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);
        assert_eq!(output, layer.output);
    }

    #[test]
    fn test_compute_gradient() {
        let input = vec![-1_f64, 0_f64, 1_f64];
        let error = vec![0_f64, 0.3_f64];

        let mut layer = Neuronal::new(3, 2, Nonlinearity::Tanh);
        let output = layer.forward(&input);

        let gradient = layer.compute_gradient(&error);
        println!("Output: {:?}", output);
        println!("Gradient: {:?}", gradient);

        assert_eq!(gradient.len(), 4);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_update_weights() {
        let input = vec![-1_f64, 0_f64, 1_f64];
        let error = vec![0_f64, 0.3_f64];

        let mut layer = Neuronal::new(3, 2, Nonlinearity::Tanh);
        let _ = layer.forward(&input);
        let _ = layer.compute_gradient(&error);
        layer.update_weights(&input, 2_f64);
    }
}

