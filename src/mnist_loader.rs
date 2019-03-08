use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;

#[derive(Debug)]
pub enum MNistLoadError {
    IOError(std::io::Error),
    String(String),
}

pub fn load_labels(path: &str) -> Result<Vec<Vec<f64>>, MNistLoadError> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let magic = match file.read_u32::<BigEndian>() {
        Ok(m) => m,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    if magic != 0x0801 {
        /* Yann LeCuns' MNIST file magic */
        return Err(MNistLoadError::String(format!("Invalid magic: {}", magic)))
    }

    let numitems = match file.read_u32::<BigEndian>() {
        Ok(n) => n,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let mut res: Vec<Vec<f64>> = vec![];
    loop {
        let val = match file.read_u8() {
            Ok(v) => v,
            Err(_) => break
        };

        if val > 10 {
            return Err(MNistLoadError::String(format!("Unexpected label {}", val)));
        }

        let mut onehot: Vec<f64> = Vec::with_capacity(10);

        for idx in 0..10 {
            onehot.push(if idx == val { 1_f64 } else { 0_f64 });
        }

        res.push(onehot);
    }

    if res.len() != numitems as usize {
        return Err(MNistLoadError::String(format!("Unexpected number of items: {}, expected {}", res.len(), numitems)));
    }

    Ok(res)
}

pub fn load_images(path: &str) -> Result<Vec<Vec<f64>>, MNistLoadError> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let magic = match file.read_u32::<BigEndian>() {
        Ok(m) => m,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    if magic != 0x0803 {
        /* Yann LeCuns' MNIST file magic */
        return Err(MNistLoadError::String(format!("Invalid magic: {}", magic)))
    }

    let numitems = match file.read_u32::<BigEndian>() {
        Ok(m) => m,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let numrows = match file.read_u32::<BigEndian>() {
        Ok(n) => n,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let numcols = match file.read_u32::<BigEndian>() {
        Ok(n) => n,
        Err(x) => return Err(MNistLoadError::IOError(x))
    };

    let mut res: Vec<Vec<f64>> = vec![];
    for _i in 0..numitems {
        let numbytes = numcols as usize * numrows as usize;
        let mut img: Vec<u8> = Vec::with_capacity(numbytes);

        unsafe { img.set_len(numbytes); }

        let sz = match file.read(img.as_mut_slice()) {
            Ok(sz) => sz,
            Err(x) => return Err(MNistLoadError::IOError(x))
        };
        if sz == 0 {
            break
        }
        if sz != numbytes {
            return Err(MNistLoadError::String(format!("Read {} bytes, expected {}", sz, numbytes)))
        }

        let val: Vec<f64> = img.iter().map(|&x| x as f64).collect();

        res.push(val);
    }

    if res.len() != numitems as usize {
        return Err(MNistLoadError::String(format!("Got {} images, expected {}", res.len(), numitems)))
    }

    Ok(res)
}

#[cfg(test)]
mod test_mnist_loader {
    use super::*;

    #[test]
    fn test_load_label_file() {
        let labels = match load_labels("mnist/t10k-labels-idx1-ubyte") {
            Ok(l) => l,
            Err(x) => panic!(x)
        };
        assert_eq!(labels.len(), 10000);
    }

    #[test]
    fn test_load_image_file() {
        let images = match load_images("mnist/t10k-images-idx3-ubyte") {
            Ok(l) => l,
            Err(x) => {
                println!("Wat: {:?}", x);
                panic!(x)
            }
        };
        assert_eq!(images.len(), 10000);
    }
}
