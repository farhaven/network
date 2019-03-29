#[derive(Debug)]
pub enum Nonlinearity {
    Tanh,
    Sigmoid,
    LeakyELU
}

impl Nonlinearity {
    pub fn forward(&self, z: &f64) -> f64 {
        match self {
            Nonlinearity::Tanh => z.tanh(),
            Nonlinearity::Sigmoid => (1.0 + z.exp()).ln(),
            Nonlinearity::LeakyELU => if z >= &0.0 {
                *z
            } else {
                let a = 0.1;
                a * (z.exp() - 1.0)
            }
        }
    }

    pub fn backward(&self, z: &f64) -> f64 {
        match self {
            Nonlinearity::Tanh => 1_f64 - z.powf(2_f64),
            Nonlinearity::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Nonlinearity::LeakyELU => if z > &0.0 {
                    1.0
                } else if z == &0.0 {
                    0.5
                } else {
                    0.1
                }
        }
    }
}
