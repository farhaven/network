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
        let mut training_labels = match network::mnist_loader::load_labels("mnist/train-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };

        assert_eq!(training_labels.len(), 60000);

        let mut training_images = match network::mnist_loader::load_images("mnist/train-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(training_images.len(), 60000);

        let test_labels = training_labels.split_off(training_labels.len() - 10000);
        let test_images = training_images.split_off(training_images.len() - 10000);

        let validation_labels = match network::mnist_loader::load_labels("mnist/t10k-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };
        assert_eq!(validation_labels.len(), 10000);

        let validation_images = match network::mnist_loader::load_images("mnist/t10k-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(validation_images.len(), 10000);

        /*
         * Topography 784 x 80 x 10 with target MSE 0.01 gives 2.85% error
         * Topography 784 x 200 x 100 x 10 with target MSE 0.01 gives 2.52% error
         * Topography 784 x 200 x 100 x 10 with target MSE 0.001 gives 1.6% error  // XXX: Not run to completion
         * Topography 784 x 2000 x 1000 x 400 x 10 with target MSE 0.001 gives 1.28% error
         */

        // let mut network = network::network::Network::new(vec![784, 80, 10]); // gives 2.85% error
        let mut network = network::network::Network::new(vec![784, 2000, 1000, 400, 10]);
        let mut epoch = 0;
        let target_mse = 0.001;
        let mut learning_rate = 0.05;
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

            if epoch % 10 == 0 && improvement <= 0.01 {
                learning_rate = mse;
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

            if mse <= target_mse {
                break;
            }
        }

        /* Test network with test data set */
        let mut total = 0;
        let mut wrong = 0;
        for idx in 0..validation_labels.len() {
            total += 1;

            let input = &validation_images[idx];
            let target = &validation_labels[idx];

            let output = network.forward(input);

            let label_target = argmax(target);
            let label_output = argmax(&output);

            if label_output != label_target {
                wrong += 1;
            }
        }

        println!("Test labels: Total: {} Wrong: {} PCT: {}", total, wrong, (wrong as f64) / (total as f64) * 100.0);
    }
}
