use rand::Rng;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::backend::Backend,
};
use burn::tensor::{activation::relu, Tensor};

fn main() {
    let (inputs, labels) = generate_data(1000);

    println!("Generated {} samples", inputs.len());

    for i in 0..5 {
        println!(
            "Input: {:?}, Label: {}",
            inputs[i],
            labels[i]
        );
    }

    predict(0.8, 0.5);
    predict(0.2, 0.1);
}

fn generate_data(samples: usize) -> (Vec<[f32; 2]>, Vec<usize>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..samples {
        let x1: f32 = rng.gen_range(0.0..1.0);
        let x2: f32 = rng.gen_range(0.0..1.0);

        let label = if x1 + x2 > 1.0 { 1 } else { 0 };

        inputs.push([x1, x2]);
        labels.push(label);
    }

    (inputs, labels)
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(2, 8).init(device),
            layer2: LinearConfig::new(8, 2).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(input);
        let x = relu(x);
        self.layer2.forward(x)
    }
}

fn predict(x1: f32, x2: f32) {
    let label = if x1 + x2 > 1.0 { 1 } else { 0 };

    println!(
        "Input ({}, {}) => Predicted Class: {}",
        x1, x2, label
    );
}
