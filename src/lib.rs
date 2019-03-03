use ndarray::Array2;
use rand::Rng;

#[derive(Debug)]
pub struct Layer {
    output: Array2<f64>,
    delta: Array2<f64>,
    weights: Array2<f64>
}
impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let mut rng = rand::thread_rng();
        let weights = Array2::<f64>::from_shape_fn((inputs, outputs), |_| {
            let r: f64 = rng.gen();
            (r * 2.0) - 1.0
        });

        Layer{
            weights,
            output: Array2::<f64>::zeros((1, outputs)),
            delta: Array2::<f64>::zeros((1, outputs))
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let output = self.weights.t().dot(&inputs.t()).map(sigmoid);
        self.output = output.clone();
        output
    }

    pub fn compute_gradient(&mut self, error: Array2<f64>) -> Array2<f64> {
        self.delta = error * self.output.t().map(sigmoid_prime);
        self.delta.dot(&self.weights.t())
    }

    pub fn update_weights(&mut self, input: Array2<f64>, learning_rate: f64) {
        let delta = input.t().dot(&self.delta) * learning_rate;
        self.weights += &delta;
    }

    pub fn shape<'a>(&'a self) -> &'a[usize] {
        self.weights.shape()
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
        let input = Array2::<f64>::from_shape_vec((1, 3), vec![-1_f64, 0_f64, 1_f64]).unwrap();

        let mut layer = Layer::new(3, 2);
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);
        assert_eq!(output, layer.output);
    }

    #[test]
    fn test_compute_gradient() {
        let input = Array2::<f64>::from_shape_vec((1, 3), vec![-1_f64, 0_f64, 1_f64]).unwrap();
        let error = Array2::<f64>::from_shape_vec((1, 2), vec![0_f64, 0.3_f64]).unwrap();

        let mut layer = Layer::new(3, 2);
        let output = layer.forward(&input);

        let gradient = layer.compute_gradient(error);
        println!("Output: {}", output);
        println!("Gradient: {}", gradient);

        assert_eq!(gradient.len(), 3);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_update_weights() {
        let input = Array2::<f64>::from_shape_vec((1, 3), vec![-1_f64, 0_f64, 1_f64]).unwrap();
        let error = Array2::<f64>::from_shape_vec((1, 2), vec![0_f64, 0.3_f64]).unwrap();

        let mut layer = Layer::new(3, 2);
        let _ = layer.forward(&input);
        let _ = layer.compute_gradient(error);
        layer.update_weights(input, 2_f64);
    }
}

fn sigmoid(z: &f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: &f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
