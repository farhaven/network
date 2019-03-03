use ndarray::{Array1, Array2, arr1};
use rand::Rng;

#[derive(Debug)]
pub struct Layer {
    output: Array1<f64>,
    delta: Array1<f64>,
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
            output: Array1::<f64>::zeros(outputs),
            delta: Array1::<f64>::zeros(outputs)
        }
    }

    pub fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let output = self.weights.t().dot(inputs).map(sigmoid);
        self.output = output.clone();
        output
    }

    pub fn compute_gradient(&mut self, error: Array1<f64>) -> Array1<f64> {
        println!("Computing gradients for {:?}, error: {:?}", self, error);
        self.delta = error * self.output.map(sigmoid_prime);
        println!("Delta done: {:?}", self.delta);
        self.delta.dot(&self.weights.t())
    }

    pub fn update_weights(&mut self, input: Array1<f64>, learning_rate: f64) {
        println!("Updating weights for {:?} (input: {:?})", self, input);
        // let update = input.t().dot(&self.delta);
        // let update = input.t().dot(&self.delta);
        let update = &input.t() * &self.delta;
        println!("Update: {:?}", update);
        // self.weights += update * learning_rate;
    }

    pub fn shape<'a>(&'a self) -> &'a[usize] {
        self.weights.shape()
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    layer_sizes: Vec<usize>,
}

fn sigmoid(z: &f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: &f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Network {
        let mut layers = Vec::<Layer>::new();

        for idx in 0..layer_sizes.len() - 1 {
            let inputs  = layer_sizes[idx];
            let outputs = layer_sizes[idx + 1];
            layers.push(Layer::new(inputs, outputs));
        }

        Network{
            layers: layers,
            layer_sizes: layer_sizes,
        }
    }

    /*
    pub fn generate_gradient(&mut self, layer: &Array2<f64>, error: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        /* Returns: delta, gradient */
        let delta = error.dot(
    }
    */

    pub fn backprop(&mut self, input: &Vec<f64>, target: &Vec<f64>) {
        /* Get delta for each layer */
        let learning_rate = 3.0;
        let mut a_input = arr1(input.as_slice());
        let output = arr1(self.forward(input).as_slice());
        let mut error = output - arr1(target);
        println!("Error: {:?}", error);

        for layer in self.layers.iter_mut().rev() {
            error = layer.compute_gradient(error);
        }

        println!("Got gradients");

        for layer in &mut self.layers {
            layer.update_weights(a_input, learning_rate);
            a_input = layer.output.clone();
        }
    }

    pub fn error(&mut self, input: &Vec<f64>, targets: &Vec<f64>) -> f64 {
        /* Squared error */
        let output = self.forward(input);

        let mut res = 0.0;
        for idx in 0..output.len() {
            res += (output[idx] - targets[idx]).powi(2);
        }
        res
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = Array1::<f64>::from_vec(input.to_vec());
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        (*(output.as_slice().unwrap())).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new () {
        let n = Network::new(vec![2, 4, 1]);
        assert_eq!(n.layers.len(), 2);
    }

    #[test]
    fn test_forward() {
        let mut n = Network::new(vec![2, 4, 1]);
        let input = vec![1.0, 0.0];
        let output = n.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_error() {
        let mut n = Network::new(vec![2, 3, 1]);
        let e = n.error(&vec![1.0, 0.0], &vec![1.0]);
        assert!(e >= 0.0);
    }

    #[test]
    fn test_xor_backprop() {
        /* Teach a 2x3x1 network to do XOR with backprop */
        let samples = [[vec![0.0, 0.0], vec![0.0]],
                       [vec![0.0, 1.0], vec![1.0]],
                       [vec![1.0, 0.0], vec![1.0]],
                       [vec![1.0, 1.0], vec![0.0]]];
        let mut net = Network::new(vec![2, 3, 1]);

        for i in 0..100 {
            if i % 10 == 0 {
                println!("Training iteration: {}", i)
            }

            for s in &samples {
                net.backprop(&s[0], &s[1]);
            }
        }

        for s in &samples {
            let pred = net.forward(&s[0]);
            println!("Sample: {:?}, Prediction: {:?}", s, pred);
            assert!(pred[0].round() == s[1][0]);
            println!("S: {:?}, net: {:?}", s, pred);
        }
    }
}
