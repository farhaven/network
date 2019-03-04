use ndarray::{Array1, Array2};
use rand::Rng;

fn nonlinearity(z: &f64) -> f64 {
    z.tanh()
}

fn nonlinearity_prime(z: &f64) -> f64 {
    1_f64 - z.powf(2_f64)
}

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
        let output = self.weights.t().dot(&inputs.t()).map(nonlinearity);
        self.output = output.t().to_owned().clone();
        output.t().to_owned()
    }

    pub fn compute_gradient(&mut self, error: Array2<f64>) -> Array2<f64> {
        self.delta = error * self.output.map(nonlinearity_prime);
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
        let mut output = Array2::<f64>::from_shape_vec((1, input.len()), input.to_vec()).unwrap();

        for layer in &mut self.layers {
            output = layer.forward(&output);
        }

        let mut res = Vec::<f64>::new();
        res.extend_from_slice(output.column(0).into_slice().unwrap());
        res
    }

    pub fn backprop(&mut self, input_param: &Vec<f64>, error_param: &Vec<f64>, learning_rate: f64) {
        let mut error = Array2::<f64>::from_shape_vec((1, error_param.len()), error_param.to_vec()).unwrap();

        for lidx in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[lidx];
            error = layer.compute_gradient(error);
        }

        let mut input = Array2::<f64>::from_shape_vec((1, input_param.len()), input_param.to_vec()).unwrap();
        for layer in &mut self.layers {
            layer.update_weights(input, learning_rate);
            input = layer.output.clone();
        }
    }

    pub fn error(&self, output: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
        (Array1::from_vec(target.to_vec()) - Array1::from_vec(output.to_vec())).iter()
                                                                               .map(|&x| x)
                                                                               .collect()
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
        let mut network = Network::new(vec![2, 3, 1]);
        let samples = vec![(vec![0_f64, 0_f64], vec![0_f64]),
                           (vec![0_f64, 1_f64], vec![1_f64]),
                           (vec![1_f64, 0_f64], vec![1_f64]),
                           (vec![1_f64, 1_f64], vec![0_f64])];

        let target_mse = 0.01;
        let mut learning_rate = 1_f64;
        let mut errors: Vec<f64> = vec![];

        let mut iter = 0;
        while iter < 10000 {
            iter += 1;

            for (input, target) in &samples {
                let output = network.forward(input);
                let error = network.error(&output, target);
                network.backprop(input, &error, learning_rate);
                errors.push(error.iter().fold(0_f64, |acc, x| acc + x.powf(2_f64)));
            }

            /* Report MSE every 100 iterations */
            if iter % 100 == 0 {
                let mse = errors.iter().fold(0_f64, |acc, x| acc + x).sqrt() / (errors.len() as f64);
                println!("Iter: {}, MSE: {}", iter, mse);
                if mse <= target_mse {
                    println!("Reached target MSE after {} iterations", iter);
                    break;
                }
                learning_rate = mse;
            }
        }

        assert!(iter <= 1000);

        for (input, target) in &samples {
            let output = network.forward(input);
            let rounded_output: Vec<f64> = output.iter().map(|&x| x.round()).collect();
            println!("In: {:?}, Out: {:?}, Target: {:?}, Round: {:?}", input, output, target, rounded_output);
            assert_eq!(&rounded_output, target);
        }
    }
}
