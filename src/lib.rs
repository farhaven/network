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

    pub fn breed(&self, other: &Network) -> Network {
        /* Create a new child network from `self` and `other`, selecting weights with equal probability */
        let mut rng = rand::thread_rng();
        let mut child = Network::new(self.layer_sizes.to_vec());
        for idx in 0..self.layers.len() {
            let s = self.layers[idx].shape();
            let map = Array2::<f64>::from_shape_fn((s[0], s[1]), |_| rng.gen()).mapv(|a| if a < 0.5 {0.0} else {1.0});
            let new_weights = (&map * &self.layers[idx].weights) +
                            ((Array2::<f64>::ones((s[0], s[1])) - &map) * &other.layers[idx].weights);
            let mut new_layer = Layer::new(s[0], s[1]);
            new_layer.weights = new_weights;

            child.layers[idx] = new_layer;
        }
        child
    }

    pub fn mutate(&mut self, rate: f64) {
        let mut rng = rand::thread_rng();
        let mut new_layers: Vec<Layer> = vec![];

        for idx in 0..self.layers.len() {
            let s = self.layers[idx].shape();
            let mutation = Array2::<f64>::from_shape_fn((s[0], s[1]), |_| {
                let c: f64 = rng.gen();
                if c > rate {
                    0.0
                } else {
                    let m: f64 = rng.gen();
                    (m * 2.0) - 1.0
                }
            });
            let new_weights = &self.layers[idx].weights + &mutation;
            let mut new_layer = Layer::new(s[0], s[1]);
            new_layer.weights = new_weights;
            new_layers.push(new_layer);
        }

        self.layers = new_layers;
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
    fn test_simple_breed() {
        let n1 = Network::new(vec![2, 3, 1]);
        let n2 = Network::new(vec![2, 3, 1]);
        let c = n1.breed(&n2);
        drop(c);
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

    #[test]
    fn test_xor_breed() {
        let limit = 1e-3;
        let mutrate = 0.2;
        let numnets = 25;
        let numparents = 5;
        let geometry = vec![2, 4, 1];

        assert!((numparents as f32) <= (numnets as f32).sqrt());

        /* Train 2x4x1 networks to do XOR by selective breeding */
        let samples = [[vec![0.0, 0.0], vec![0.0]],
                       [vec![0.0, 1.0], vec![1.0]],
                       [vec![1.0, 0.0], vec![1.0]],
                       [vec![1.0, 1.0], vec![0.0]]];
        let mut networks: Vec<Network> = (0..numnets).map(|_| Network::new(geometry.to_vec())).collect();
        let sample_err = |net: &mut Network| -> f64 {
            let mut e = 0.0;
            for s in &samples {
                e += net.error(&s[0], &s[1]);
            }
            e
        };
        let _net_cmp = |a: &mut Network, b: &mut Network| {
            let ae = sample_err(a);
            let be = sample_err(b);
            ae.partial_cmp(&be).unwrap()
        };

        let mut iterations = 0;
        loop {
            break
            iterations += 1;
            // networks.sort_by(net_cmp);
            let mut total_err = 0.0;
            for n in &mut networks {
                total_err += sample_err(n);
            }
            let best_err = sample_err(&mut networks[0]);
            if best_err < limit {
                break;
            }
            if iterations % 50 == 0 {
                println!("Total err: {} best err {}", total_err, best_err);
            }
            /* Take the top `numparents` networks, cross breed them, re-sort by fitness, cut list down to `numnets` networks */
            let mut children: Vec<Network> = vec![];
            for p1idx in 0..numparents {
                for p2idx in 0..numparents {
                    if p1idx == p2idx {
                        continue;
                    }
                    let child = networks[p1idx].breed(&networks[p2idx]);
                    children.push(child);
                }
            }
            networks.append(&mut children);

            for idx in 0..networks.len() {
                networks[idx].mutate(mutrate);
            }

            /* Re-sort network list */
            // networks.sort_by(net_cmp);
            /* Cut off networks list */
            networks.truncate(numnets);
        }

        let best_net = &mut networks[0];
        println!("Took {} iterations to breed a network with target error", iterations);

        for s in &samples {
            let pred = best_net.forward(&s[0]);
            assert!(pred[0].round() == s[1][0]);
            println!("S: {:?}, net: {:?}", s, pred);
        }

        println!("Best network: {:?}", best_net);

        assert!(iterations < 200);
    }
}
