extern crate network;

#[cfg(test)]
mod integration_tests {
    use rand::thread_rng;
    use rand::seq::SliceRandom;

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

        let mut network = network::network::Network::new(vec![784, 80, 10]);
        let mut iter = 0;
        let target_mse = 0.1;
        let mut learning_rate = 0.001;
        let mut errors: Vec<f64> = vec![];

        loop {
            iter += 1;

            println!("Iter: {}", iter);

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

            println!("\tMSE:{}", errors.iter().fold(0_f64, |acc, x| acc + x) / (errors.len() as f64));

            if iter % 10 == 0 {
                let mse = errors.iter().fold(0_f64, |acc, x| acc + x) / (errors.len() as f64);
                println!("Iter: {}, MSE: {}", iter, mse);
                if mse <= target_mse {
                    break;
                }
                // learning_rate = mse.sqrt();
                // println!("New learning rate: {}", learning_rate);
            }
        }

        /* Test network with test data set */
        let mut total = 0;
        let mut wrong = 0;
        for idx in 0..test_labels.len() {
            total += 1;

            let input = &test_images[idx];
            let target = &test_labels[idx];

            let output = network.forward(input);

            let mut label_target = 0;
            let mut val_target = 0_f64;
            for idx in 0..target.len() {
                if target[idx] > val_target {
                    label_target = idx;
                    val_target = target[idx];
                }
            }

            let mut label_test = 0;
            let mut val_test = 0_f64;
            for idx in 0..output.len() {
                if output[idx] > val_test {
                    label_test = idx;
                    val_test = output[idx];
                }
            }

            if label_test != label_target {
                wrong += 1;
            }
        }

        println!("Total: {} Wrong: {} PCT: {}", total, wrong, (wrong as f64) / (total as f64));
    }
}
