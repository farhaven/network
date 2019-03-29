use crate::layer::Neuronal;
use crate::nonlinearity::Nonlinearity;

#[derive(Debug)]
enum WrappedLayer {
    Neuronal(Neuronal)
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<WrappedLayer>,
    layer_sizes: Vec<usize>
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Network {
        let mut layers = Vec::<WrappedLayer>::new();

        for idx in 0..layer_sizes.len() - 1 {
            let inputs = layer_sizes[idx];
            let outputs = layer_sizes[idx + 1];
            layers.push(WrappedLayer::Neuronal(Neuronal::new(inputs, outputs, Nonlinearity::Tanh)));
        }

        Network{
            layers: layers,
            layer_sizes: layer_sizes
        }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = input.clone();

        for wrapper in &mut self.layers {
            let layer = match wrapper {
                WrappedLayer::Neuronal(x) => x
            };

            output = layer.forward(&output);
        }

        output
    }

    pub fn backprop(&mut self, input: &Vec<f64>, error: &Vec<f64>, learning_rate: f64) {
        let mut local_error = error.clone();

        for lidx in (0..self.layers.len()).rev() {
            let layer = match &mut self.layers[lidx] {
                WrappedLayer::Neuronal(x) => x
            };
            local_error = layer.compute_gradient(&local_error);
        }

        let mut local_input = input.clone();

        for wrapper in &mut self.layers {
            let layer = match wrapper {
                WrappedLayer::Neuronal(x) => x
            };
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
        let mut learning_rate = 0.01_f64;
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
                learning_rate = mse;
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
