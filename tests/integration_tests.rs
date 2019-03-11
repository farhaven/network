extern crate network;

#[cfg(test)]
mod integration_tests {
    use rand::thread_rng;
    use rand::seq::SliceRandom;

    fn argmax(v: &Vec<f64>) -> usize {
        let mut maximum_seen = std::f64::MIN;
        let mut max_idx = 0;

        for idx in 0..v.len() {
            if v[idx] > maximum_seen {
                max_idx = idx;
                maximum_seen = v[idx];
            }
        }

        max_idx
    }

    #[test]
    fn test_load() {
        let labels = match network::mnist_loader::load_labels("mnist/t10k-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };

        assert_eq!(labels.len(), 10000);

        let images = match network::mnist_loader::load_images("mnist/t10k-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(images.len(), 10000);
    }

    #[test]
    fn test_train_mnist() {
        /* TODO: Split off a validation set from the training data and use that to confirm
         * performance */
        let training_labels = match network::mnist_loader::load_labels("mnist/train-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };

        assert_eq!(training_labels.len(), 60000);

        let training_images = match network::mnist_loader::load_images("mnist/train-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(training_images.len(), 60000);

        let test_labels = match network::mnist_loader::load_labels("mnist/t10k-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };

        assert_eq!(test_labels.len(), 10000);

        let test_images = match network::mnist_loader::load_images("mnist/t10k-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(test_images.len(), 10000);

        // let mut network = network::network::Network::new(vec![784, 80, 10]);
        let mut network = network::network::Network::new(vec![784, 200, 100, 10]);
        let mut epoch = 0;
        let target_mse = 0.01;
        let mut learning_rate = 0.005;
        let mut errors: Vec<f64> = vec![];
        let mut previous_mse = std::f64::MAX;

        loop {
            epoch += 1;

            println!("epoch: {}", epoch);

            let mut indices: Vec<usize> = (0..training_labels.len()).collect();
            indices.shuffle(&mut thread_rng());

            for idx in indices {
                let input = &training_images[idx];
                let target = &training_labels[idx];

                let output = network.forward(input);
                let error = network.error(&output, target);
                network.backprop(input, &error, learning_rate);
                errors.push(error.iter().fold(0_f64, |acc, x| acc + x.powf(2_f64)) / (error.len() as f64));
            }

            let mse = errors.iter().fold(0_f64, |acc, x| acc + x) / (errors.len() as f64);
            let improvement = previous_mse - mse;
            println!("\tMSE: {}", mse);
            if epoch != 1 {
                println!("\tMSE improvement: {}", improvement);
            }
            previous_mse = mse;

            if mse <= target_mse {
                break;
            }

            if epoch % 10 == 0 && improvement <= 0.01 {
                learning_rate = 0.005 * mse.sqrt();
                println!("\tNew learning rate: {}", learning_rate);
            }

            /* Test network with test data set */
            let mut total = 0;
            let mut wrong = 0;
            for idx in 0..test_labels.len() {
                total += 1;

                let input = &test_images[idx];
                let target = &test_labels[idx];

                let output = network.forward(input);

                let label_target = argmax(target);
                let label_output = argmax(&output);

                if label_output != label_target {
                    wrong += 1;
                }
            }

            println!("\tTotal: {} Wrong: {} PCT: {}", total, wrong, (wrong as f64) / (total as f64) * 100.0);
        }
    }
}
