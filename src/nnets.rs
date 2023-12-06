extern crate nalgebra as na;
use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
#[allow(non_snake_case, dead_code)]
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
    count: u64,
}

#[allow(non_snake_case, dead_code)]
impl<T: InputType + Clone + Debug> Network<T> {
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
            alpha: 0.05,
            gamma: 0.95,
            weights,
            biases,
            layers,
            dB,
            dW,
            count: 0,
        };
    }

    pub fn set_input(&mut self, input_data: Vec<T>) {
        self.input_data = input_data;
    }

    pub fn forward_prop(&mut self, input_data: Vec<T>) -> Vec<f64> {
        let mut input_vec: Vec<f64> = Vec::new();
        for i in 0..input_data.len() {
            input_vec.push(input_data[i].to_f64());
        }

        let input_vector: DVector<f64> = DVector::from_vec(input_vec);
        *self.layers[0] =
            (*self.weights[0].clone() * input_vector.map(sigmoid) + *self.biases[0].clone());

        for l in 1..self.node_count.len() {
            if l == self.node_count.len() {
                *self.layers[l] = DVector::from_vec(softmax(
                    (*self.weights[l].clone() * self.layers[l - 1].clone().map(sigmoid)
                        + *self.biases[l].clone())
                    .data
                    .as_vec()
                    .to_vec(),
                ))
            } else {
                *self.layers[l] = (*self.weights[l].clone()
                    * self.layers[l - 1].clone().map(sigmoid)
                    + *self.biases[l].clone())
                .map(sigmoid);
            }
        }
        self.layers[self.node_count.len() - 1]
            .data
            .as_vec()
            .to_vec()
    }

    pub fn compute(&mut self, input_data: Vec<T>) {
        for i in 0..input_data.len() {
            self.layers[0][i] = input_data[i].to_f64();
        }
        for l in 1..self.node_count.len() {
            if l == self.node_count.len() {
                *self.layers[l] = DVector::from_vec(softmax(
                    (*self.weights[l].clone() * self.layers[l - 1].clone().map(sigmoid)
                        + *self.biases[l].clone())
                    .data
                    .as_vec()
                    .to_vec(),
                ))
            } else {
                *self.layers[l] = (*self.weights[l].clone()
                    * self.layers[l - 1].clone().map(sigmoid)
                    + *self.biases[l].clone())
                .map(sigmoid);
            }
        }
    }

    pub fn train(&mut self, training_set: Vec<(Vec<T>, Vec<f64>)>, stride: i32, moves: Vec<usize>) {
        //println!("training_set: {:?}", training_set);
        //println!("moves: {:?}", moves);
        assert!(training_set.len() == moves.len());
        let mut gamma: f64 = 1.0 / (training_set.len() as f64);
        self.count += 1;
        for ((input_data, dCda), index) in training_set.into_iter().zip(moves.clone()).rev() {
            self.back_prop(input_data, dCda, gamma, index);
            gamma = gamma * self.gamma.powi(stride);
        }
    }

    pub fn back_prop(&mut self, input_data: Vec<T>, dCda: Vec<f64>, gamma: f64, index: usize) {
        #[allow(non_snake_case)]
        let mut dCda_curr = dCda; //dCda layer L-l
        let output = self.forward_prop(input_data.clone());
        let pi = output[index];

        let layer_count = self.layers.len();
        let mut l: usize = 1;
        while l <= layer_count {
            let mut dCda_next: Vec<f64> = Vec::new();
            if l != layer_count {
                dCda_next = vec![0.0; self.layers[layer_count - l - 1].len()]; // dCda layer L-(l+1)
            }
            let mut j: usize = 0;
            while j < self.layers[layer_count - l].len() {
                let z = self.layers[layer_count - l][j];
                self.dB[layer_count - l][j] +=
                    ((self.alpha * dCda_curr[j] * Dsigmoid(z)) * gamma) / pi; // if something breaks, remove the pi's
                let mut k: usize = 0;
                while k < self.weights[layer_count - l].row(j).len() {
                    if l != layer_count {
                        dCda_next[k] +=
                            dCda_curr[j] * Dsigmoid(z) * self.weights[layer_count - l][(j, k)]; //(row,col)

                        self.dW[layer_count - l][(j, k)] += ((self.alpha
                            * dCda_curr[j]
                            * Dsigmoid(z)
                            * self.layers[layer_count - l - 1][k])
                            * gamma)
                            / pi;
                    } else {
                        self.dW[layer_count - l][(j, k)] +=
                            ((self.alpha * dCda_curr[j] * Dsigmoid(z) * input_data[k].to_f64())
                                * gamma)
                                / pi;
                    }
                    k += 1;
                }
                j += 1
            }
            l += 1;
            dCda_curr = dCda_next;
        }
    }

    pub fn update(&mut self) {
        let layer_count = self.layers.len();
        let mut i = 0;
        while i < layer_count {
            self.weights[i] = Box::new(*self.weights[i].clone() + *self.dW[i].clone());
            self.dW[i].fill(0.0);
            self.biases[i] = Box::new(*self.biases[i].clone() + *self.dB[i].clone());
            self.dB[i].fill(0.0);
            i += 1;
        }
    }
}

pub trait InputType {
    fn to_f64(&self) -> f64;
}

#[allow(non_snake_case, dead_code)]
pub fn ReLU(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

#[allow(non_snake_case, dead_code)]
pub fn DReLU(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

#[allow(non_snake_case, dead_code)]
pub fn sigmoid(x: f64) -> f64 {
    (1 as f64 / (1 as f64 + std::f64::consts::E.powf(-x)))
}
#[allow(non_snake_case, dead_code)]
pub fn Dsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[allow(non_snake_case, dead_code)]
pub fn LReLU(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        x * 0.01
    }
}

#[allow(non_snake_case, dead_code)]
pub fn DLReLU(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.01
    }
}

#[allow(non_snake_case, dead_code)]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[allow(non_snake_case, dead_code)]
pub fn Dtanh(x: f64) -> f64 {
    1.0 - (x.tanh() * x.tanh())
}

#[allow(non_snake_case, dead_code)]
pub fn softmax(xs: Vec<f64>) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    let mut total: f64 = 0.0;
    let mut term: f64 = 0.0;
    for x in xs {
        term = std::f64::consts::E.powf(x);
        total += term;
        vec.push(term);
    }
    vec.iter_mut().map(|x| *x / total).collect()
}
