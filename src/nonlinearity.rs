#[derive(Debug)]
pub enum Nonlinearity {
    Tanh,
    Sigmoid,
    LeakyELU(f64) // leak, e.g. 0.1
}

impl Nonlinearity {
    pub fn forward(&self, z: &f64) -> f64 {
        match self {
            Nonlinearity::Tanh => z.tanh(),
            Nonlinearity::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Nonlinearity::LeakyELU(leak) => if z >= &0.0 {
                *z
            } else {
                leak * (z.exp() - 1.0)
            }
        }
    }

    pub fn backward(&self, z: &f64) -> f64 {
        match self {
            Nonlinearity::Tanh => 1_f64 - z.powf(2_f64),
            Nonlinearity::Sigmoid => self.forward(z) * (1.0 - self.forward(z)),
            Nonlinearity::LeakyELU(leak) => if z > &0.0 {
                    1.0
                } else if z == &0.0 {
                    0.5
                } else {
                    *leak
                }
        }
    }
}
