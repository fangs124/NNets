extern crate nalgebra as na;
use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[allow(non_snake_case, dead_code)]
#[derive(Serialize, Deserialize)]
pub struct Network<T> {
    pub node_count: Vec<usize>,
    pub input_data: Vec<T>,
    pub alpha: f32,                  // learning rate
    pub gamma: f32,                  // patience factor
    weights: Vec<Box<DMatrix<f32>>>, // row only (col vector)
    biases: Vec<Box<DVector<f32>>>,  // row,col
    layers: Vec<Box<DVector<f32>>>,  // row,col //stores z_j as in a_j = phi(z_j)
    dW: Vec<Box<DMatrix<f32>>>,
    dB: Vec<Box<DVector<f32>>>,
}

#[allow(non_snake_case, dead_code)]
impl<T: InputType + Clone> Network<T> {
    //internal node count describes internal layer
    pub fn new_default(internal_nodes: Vec<usize>, input_data: Vec<T>) -> Self {
        let mut weights: Vec<Box<DMatrix<f32>>> = Vec::with_capacity(internal_nodes.len());
        let mut biases: Vec<Box<DVector<f32>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut layers: Vec<Box<DVector<f32>>> = Vec::with_capacity(internal_nodes.len() + 1);
        let mut dW: Vec<Box<DMatrix<f32>>> = Vec::with_capacity(internal_nodes.len() - 1);
        let mut dB: Vec<Box<DVector<f32>>> = Vec::with_capacity(internal_nodes.len());
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
            gamma: 0.8,
            weights,
            biases,
            layers,
            dB,
            dW,
        };
    }

    pub fn set_input(&mut self, input_data: Vec<T>) {
        self.input_data = input_data;
    }

    pub fn forward_prop(&mut self, input_data: Vec<T>) -> Vec<f32> {
        let mut input_vec: Vec<f32> = Vec::new();
        for i in 0..input_data.len() {
            input_vec.push(input_data[i].to_f32());
        }
        let input_vector: DVector<f32> = DVector::from_vec(input_vec);
        *self.layers[0] =
            (*self.weights[0].clone() * input_vector + *self.biases[0].clone()).map(sigmoid);

        for l in 1..self.node_count.len() {
            *self.layers[l] = (*self.weights[l].clone() * *self.layers[l - 1].clone()
                + *self.biases[l].clone())
            .map(sigmoid);
        }
        self.layers[self.node_count.len() - 1]
            .data
            .as_vec()
            .to_vec()
    }

    pub fn compute(&mut self, input_data: Vec<T>) {
        for i in 0..input_data.len() {
            self.layers[0][i] = input_data[i].to_f32();
        }
        let l: usize = 0;
        for l in 1..self.node_count.len() {
            // compute z^(l)
            *self.layers[l] = (*self.weights[l].clone() * *self.layers[l - 1].clone()
                + *self.biases[l].clone())
            .map(sigmoid);
        }
    }

    pub fn train(&mut self, training_set: Vec<(Vec<T>, Vec<f32>)>, stride: i32) {
        let mut gamma: f32 = 1.0 / (training_set.len() as f32);
        for (input_data, dCda) in training_set {
            self.back_prop(input_data, dCda, gamma);
            gamma = gamma * self.gamma.powi(stride);
        }
    }

    pub fn back_prop(&mut self, input_data: Vec<T>, dCda: Vec<f32>, gamma: f32) {
        #[allow(non_snake_case)]
        let mut dCda_curr = dCda; //dCda layer L-l
        self.compute(input_data.clone());
        let layer_count = self.layers.len();
        let mut l: usize = 1;
        while l <= layer_count {
            let mut dCda_next: Vec<f32> = Vec::new();
            if l != layer_count {
                dCda_next = vec![0.0; self.layers[layer_count - l - 1].len()]; // dCda layer L-(l+1)
            }
            let mut j: usize = 0;
            while j < self.layers[layer_count - l].len() {
                let z = self.layers[layer_count - l][j];
                self.dB[layer_count - l][j] += (self.alpha * dCda_curr[j] * Dsigmoid(z)) * gamma;
                let mut k: usize = 0;
                while k < self.weights[layer_count - l].row(j).len() {
                    if l != layer_count {
                        dCda_next[k] +=
                            dCda_curr[j] * Dsigmoid(z) * self.weights[layer_count - l][(j, k)]; //(row,col)

                        self.dW[layer_count - l][(j, k)] += (self.alpha
                            * dCda_curr[j]
                            * Dsigmoid(z)
                            * self.layers[layer_count - l - 1][k])
                            * gamma;
                    } else {
                        self.dW[layer_count - l][(j, k)] +=
                            (self.alpha * dCda_curr[j] * Dsigmoid(z) * input_data[k].to_f32())
                                * gamma;
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
    fn to_f32(&self) -> f32;
}

#[allow(non_snake_case, dead_code)]
pub fn ReLU(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

#[allow(non_snake_case, dead_code)]
pub fn DReLU(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

#[allow(non_snake_case, dead_code)]
pub fn sigmoid(x: f32) -> f32 {
    1 as f32 / (1 as f32 + std::f32::consts::E.powf(-x))
}

#[allow(non_snake_case, dead_code)]
pub fn Dsigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
