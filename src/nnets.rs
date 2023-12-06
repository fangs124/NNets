#![allow(non_camel_case_types, non_snake_case, dead_code)]
extern crate nalgebra as na;
use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Network<T> {
    pub node_count: Vec<usize>,
    pub input_data: Vec<T>,
    pub alpha: f64,                  // learning rate
    pub gamma: f64,                  // patience factor
    weights: Vec<Box<DMatrix<f64>>>, // row only (col vector)
    biases: Vec<Box<DVector<f64>>>,  // row,col
    layers: Vec<Box<DVector<f64>>>,  // row,col //stores z_j as in a_j = phi(z_j)
    dW: Vec<Box<DMatrix<f64>>>,
    dB: Vec<Box<DVector<f64>>>,
    ty: phi_T, // to select internal activation function used
}

fn phi(ty: &phi_T) -> fn(f64) -> f64 {
    match ty {
        phi_T::sigmoid => sigmoid,
        phi_T::ReLU => ReLU,
        phi_T::LReLU => LReLU,
        phi_T::tanh => tanh,
    }
}

fn Dphi(ty: &phi_T) -> fn(f64) -> f64 {
    match ty {
        phi_T::sigmoid => Dsigmoid,
        phi_T::ReLU => DReLU,
        phi_T::LReLU => DLReLU,
        phi_T::tanh => Dtanh,
    }
}

#[derive(Serialize, Deserialize)]
enum phi_T {
    sigmoid,
    ReLU,
    LReLU,
    tanh,
}

// idk how to implement partial defaults, so document here
// alpha = 0.05, gamma = 0.95

impl<T: InputType> Network<T> {
    //internal node count describes internal layer
    pub fn new_default(internal_nodes: Vec<usize>, input_data: Vec<T>) -> Self {
        let mut weights: Vec<Box<DMatrix<f64>>> = Vec::with_capacity(internal_nodes.len());
        let mut biases: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut layers: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut dW: Vec<Box<DMatrix<f64>>> = Vec::with_capacity(internal_nodes.len() - 1);
        let mut dB: Vec<Box<DVector<f64>>> = Vec::with_capacity(internal_nodes.len());
        let mut j: usize = input_data.len(); //previous node count
        let mut node_count: Vec<usize> = vec![internal_nodes.len()];
        weights.push(Box::new(DMatrix::new_random(j, j)));
        dW.push(Box::new(DMatrix::zeros(j, j))); // M[i]: layer[j] -> layer[i]
        biases.push(Box::new(DVector::new_random(j)));
        dB.push(Box::new(DVector::zeros(j)));
        layers.push(Box::new(DVector::zeros(j)));
        for i in internal_nodes {
            weights.push(Box::new(DMatrix::new_random(i, j))); // M[i]: layer[j] -> layer[i]
            biases.push(Box::new(DVector::new_random(i)));
            layers.push(Box::new(DVector::zeros(i)));
            dW.push(Box::new(DMatrix::zeros(i, j))); // M[i]: layer[j] -> layer[i]
            dB.push(Box::new(DVector::zeros(i)));
            node_count.push(i);
            j = i;
        }
        return Network {
            node_count,
            input_data,
            alpha: 0.5,
            gamma: 0.95,
            weights,
            biases,
            layers,
            dB,
            dW,
            ty: phi_T::sigmoid,
        };
    }

    pub fn set_input(&mut self, input_data: Vec<T>) {
        self.input_data = input_data;
    }

    pub fn forward_prop(&mut self, input_data: &mut Vec<T>) -> Vec<f64> {
        self.compute(input_data);

        self.layers[self.node_count.len() - 1]
            .data
            .as_vec()
            .to_vec()
    }

    pub fn compute(&mut self, input_data: &mut Vec<T>) {
        // turn input data into usable format.
        let input_vector: DVector<f64> =
            DVector::from_iterator(input_data.len(), input_data.iter_mut().map(|x| x.to_f64()));

        // layer 0
        // z_l = W_l * phi(z_{l-1}) + b_l
        *self.layers[0] = &*self.weights[0] * input_vector.map(phi(&self.ty)) + &*self.biases[0];

        // layer l = 1..L-1
        for l in 1..self.node_count.len() {
            // last layer should use softargmax
            if l == self.node_count.len() - 1 {
                // z_l = softargmax(W_l * phi(z_{l-1}) + b_l)
                *self.layers[l] = DVector::from_vec(softmax(
                    (&*self.weights[l] * (&*self.layers[l - 1]).map(phi(&self.ty))
                        + &*self.biases[l])
                        .data
                        .as_vec()
                        .to_vec(),
                ))
            } else {
                // z_l = W_l * phi(z_{l-1}) + b_l
                *self.layers[l] = &*self.weights[l] * (&*self.layers[l - 1]).map(phi(&self.ty))
                    + &*self.biases[l];
            }
        }
    }

    // computes averaged out dpi/dtheta for policy pi_theta iteration.
    pub fn train(&mut self, training_set: Vec<(Vec<T>, Vec<f64>, usize)>, stride: i32) {
        let mut gamma: f64 = 1.0 / (training_set.len() as f64);
        for (mut input_data, dCda, index) in training_set.into_iter().rev() {
            self.back_prop(&mut input_data, dCda, gamma, index);
            gamma = gamma * self.gamma.powi(stride);
        }
    }

    pub fn back_prop(&mut self, input_data: &mut Vec<T>, dCda: Vec<f64>, gamma: f64, index: usize) {
        #[allow(non_snake_case)]
        let mut dCda: DVector<f64> = DVector::from_vec(dCda); //dCda layer L-1
        let output = self.forward_prop(input_data);
        let pi = output[index];
        let input_vector =
            DVector::from_iterator(input_data.len(), input_data.iter().map(|x| x.to_f64()));
        let layer_count = self.layers.len(); // L

        for l in 1..=layer_count {
            // dC/dz = dC/da * da/dz
            let z = &*self.layers[layer_count - l];
            let dCdz = dCda.component_mul(&z.map(Dphi(&self.ty)));

            // dC/db = dC/da * da/dz   * dz/db
            //       = dC/da * phi'(z) * 1
            *self.dB[layer_count - l] =
                &*self.dB[layer_count - l] + (self.alpha * gamma / pi) * &dCdz;

            if l != layer_count {
                // dC/dw = dC/da * da/dz   * dz/dw
                //       = dC/da * phi'(z) * phi(z)
                *self.dW[layer_count - l] = &*self.dW[layer_count - l]
                    + (self.alpha * gamma / pi)
                        * &dCdz
                        * (self.layers[layer_count - (l + 1)].map(phi(&self.ty))).transpose()
                // (AB)^t = B^t A^t
                // (AB^t)^t = B A^t
            } else {
                // dC/dw = dC/da * da/dz   * dz/dw
                //       = dC/da * phi'(z) * phi(z)
                *self.dW[layer_count - l] = &*self.dW[layer_count - l]
                    + (self.alpha * gamma / pi)
                        * input_vector.map(phi(&self.ty))
                        * dCdz.transpose();
            }

            // dC/da' = Sum dC/da  *  da/dz * dz/da'
            //        = Sum dz/da' *  dC/da * phi'(z)
            //        =        [w] * [dC/da * phi'(z)]
            dCda = (self.alpha * gamma / pi) * self.weights[layer_count - l].transpose() * &dCdz;
        }
    }

    // pushes policy interation theta = theta + dpi/dtheta
    pub fn update(&mut self) {
        let layer_count = self.layers.len();
        for i in 0..layer_count {
            self.weights[i] = Box::new(&*self.weights[i] + &*self.dW[i]);
            self.dW[i].fill(0.0);
            self.biases[i] = Box::new(&*self.biases[i] + &*self.dB[i]);
            self.dB[i].fill(0.0);
        }
    }
}

pub trait InputType {
    fn to_f64(&self) -> f64;
}

pub fn ReLU(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

pub fn DReLU(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1 as f64 / (1 as f64 + std::f64::consts::E.powf(-x))
}

pub fn Dsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn LReLU(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        x * 0.01
    }
}

pub fn DLReLU(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.01
    }
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn Dtanh(x: f64) -> f64 {
    1.0 - (x.tanh() * x.tanh())
}

pub fn softmax(xs: Vec<f64>) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    let mut total: f64 = 0.0;
    for x in xs {
        total += std::f64::consts::E.powf(x);
        vec.push(std::f64::consts::E.powf(x));
    }
    vec.iter_mut().map(|x| *x / total).collect()
}
