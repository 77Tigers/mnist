use mnist::*;
use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution};
use rand_distr::StandardNormal;

use std::io;
use std::io::Write;
// TODO; serde for model loading/saving
use serde::{ Serialize, Deserialize };

trait Memory {
    type O;
    fn get_output(&self) -> Self::O;
}

impl<M1, M2, O1, O2> Memory for (M1, M2) where M1: Memory<O = O1>, M2: Memory<O = O2> {
    type O = O2;

    fn get_output(&self) -> Self::O {
        let (_mem1, mem2) = self;
        mem2.get_output()
    }
}

trait Model {
    type I;
    type GradI;
    type O;
    type GradO;
    type GradW;
    type Mem: Memory<O = Self::O>;

    fn run(&self, input: Self::I) -> Self::Mem;
    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW);
    fn learn_from(&mut self, grad_w: Self::GradW);
}

struct LinearLayer<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    weights: Array2<f32>,
    biases: Array1<f32>,
    iterations: usize,
}

#[derive(Clone)]
struct LinearLayerStorage<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    input: Array1<f32>,
    output: Array1<f32>,
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> LinearLayer<INPUT_DIM, OUTPUT_DIM> {
    fn new() -> Self {
        Self::new_identity(false)
    }

    fn new_identity(id: bool) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn(
            (OUTPUT_DIM, INPUT_DIM),
            |(i, j)| thread_rng().sample::<f32,_>(StandardNormal) * 0.001
            + if i == j && id {1.0} else {0.0}
        );
        let biases = Array1::from_shape_fn(OUTPUT_DIM, |_| 0.1);//rng.gen_range(-0.01..0.01));
        LinearLayer {
            weights,
            biases,
            iterations: 0,
        }
    }
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> Memory
for LinearLayerStorage<INPUT_DIM, OUTPUT_DIM> {
    type O = Array1<f32>;
    fn get_output(&self) -> Array1<f32> {
        self.output.clone()
    }
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> Model for LinearLayer<INPUT_DIM, OUTPUT_DIM> {
    type I = Array1<f32>;
    type O = Array1<f32>;

    type GradI = Array1<f32>;
    type GradO = Array1<f32>;
    type GradW = (Array2<f32>, Array1<f32>);

    type Mem = LinearLayerStorage<INPUT_DIM, OUTPUT_DIM>;

    fn run(&self, input: Self::I) -> Self::Mem {
        let output = self.weights.dot(&input) + &self.biases;
        LinearLayerStorage {
            input,
            output,
        }
    }

    fn learn_from(&mut self, grad_w: Self::GradW) {
        self.iterations += 1;

        // learning rate
        let mut learning_rate = 0.000005;
        if self.iterations < 60_000 {
            learning_rate = 0.0007;
        }

        if self.iterations < 30_000 {
            learning_rate = 0.003;
        }

        if self.iterations < 10_000 {
            learning_rate = 0.02;
        }

        // check if self.iterations is in an array
        if [10_000, 30_000, 60_000].contains(&self.iterations) {
            println!("Iterations: {}", self.iterations);
        }

        let max_grad_value = 0.2;
        self.weights -= &grad_w.0.mapv(|x| x * learning_rate).mapv(|g| g.clamp(-max_grad_value, max_grad_value));
        self.biases -= &grad_w.1.mapv(|x| x * learning_rate * 0.1).mapv(|g| g.clamp(-max_grad_value, max_grad_value));
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        // iterate over the weights and calculate the gradient
        let _output = mem.output;
        let input = mem.input;

        let mut grad_w = Array2::from_shape_fn((OUTPUT_DIM, INPUT_DIM), |(i, j)| {
            grad[i] * input[j]
        });
        // add ridge regression penalty
        grad_w += &self.weights.mapv(|x| x * -0.001);

        let grad_i = self.weights.t().dot(&grad);

        let grad_b = grad;

        // if rand::thread_rng().gen_range(0..5000) == 0 || (self.iterations < 1000 && self.iterations % 100 == 0) {
        //     let num_active_neurons = _output.iter().filter(|&x| *x > 0.0).count();
        //     println!("---------");
        //     println!("Number of active neurons: {}", num_active_neurons);
        //     println!("changed {}, {}", grad_w.clone().fold(0.0, |a, b| a + b.abs()), grad_b.clone().fold(0.0, |a, b| a + b.abs()));
        //     println!("max weight: {}", self.weights.fold(0.0, |a: f32, b| a.max(b.abs())));
        //     println!("max bias: {}", self.biases.fold(0.0, |a: f32, b| a.max(b.abs())));
        //     println!("output: {:?}", _output);
        // }

        (grad_i, (grad_w, grad_b))
    }
}

// mean squared error loss
struct ErrorLoss<M, FL, FD>
    where
        M: Model<O = Array1<f32>, GradO = Array1<f32>>,
        FL: Fn(Array1<f32>, Array1<f32>) -> f32,
        FD: Fn(Array1<f32>, Array1<f32>) -> Array1<f32> {
    model: M,
    loss_fn: FL,
    loss_deriv: FD,
}

impl<M, I, GradI, GradW, Mem, FL, FD> Model
    for ErrorLoss<M, FL, FD>
    where
        M: Model<
            O = Array1<f32>,
            GradO = Array1<f32>,
            I = I,
            GradI = GradI,
            GradW = GradW,
            Mem = Mem
        >,
        Mem: Memory<O = Array1<f32>>,
        Mem: Clone,
        FL: Fn(Array1<f32>, Array1<f32>) -> f32,
        FD: Fn(Array1<f32>, Array1<f32>) -> Array1<f32>
{
    type I = I;
    type GradI = GradI;
    type O = Array1<f32>;
    type GradO = Array1<f32>;
    type GradW = GradW;
    type Mem = Mem;

    fn run(&self, input: Self::I) -> Self::Mem {
        self.model.run(input)
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        // let output = mem.get_output();
        // println!("output is {:?}", output);
        // println!("out grad is {:?}", grad);
        // let grad = (self.loss_deriv)(output.clone(), grad);
        let (grad_i, grad_w) = self.model.backprop(grad, mem.clone());
        (grad_i, grad_w)
    }

    fn learn_from(&mut self, grad_w: Self::GradW) {
        self.model.learn_from(grad_w);
    }
}

struct ErrorLossFactory;

impl ErrorLossFactory {
    fn least_squares<M>(
        model: M
    ) -> ErrorLoss<
            M,
            impl Fn(Array1<f32>, Array1<f32>) -> f32,
            impl Fn(Array1<f32>, Array1<f32>) -> Array1<f32>
        >
        where M: Model<O = Array1<f32>, GradO = Array1<f32>>
    {
        ErrorLoss {
            model,
            loss_fn: |output, target| {
                let mut loss = 0.0;
                for (p, t) in output.iter().zip(target.iter()) {
                    loss += (p - t).powi(2);
                }
                loss
            },
            loss_deriv: |output, target| {
                // derivative of (output - target)^2
                output - target
            },
        }
    }

    fn cross_entropy<M>(
        model: M
    ) -> ErrorLoss<
            M,
            impl Fn(Array1<f32>, Array1<f32>) -> f32,
            impl Fn(Array1<f32>, Array1<f32>) -> Array1<f32>
        >
        where M: Model<O = Array1<f32>, GradO = Array1<f32>>
    {
        ErrorLoss {
            model,
            loss_fn: |output, target| {
                let mut loss = 0.0;
                for (p, t) in output.iter().zip(target.iter()) {
                    loss -= t * p.ln();
                }
                loss
            },
            loss_deriv: |output, target| {
                - target.clone() / output.clone()
                + target.mapv(|x| 1.0 - x) / output.mapv(|x| 1.0 - x)
            },
        }
    }
}

/*
    Below are some activation functions, as models
*/
// RELU
struct ReLU<const DIM: usize>;

impl<const DIM: usize> Model for ReLU<DIM> {
    type I = Array1<f32>;
    type O = Array1<f32>;
    type GradI = Array1<f32>;
    type GradO = Array1<f32>;
    type GradW = ();
    type Mem = LinearLayerStorage<DIM, DIM>;

    fn run(&self, input: Self::I) -> Self::Mem {
        LinearLayerStorage {
            input: input.clone(), // remove clone in later rust version
            output: input.mapv(|x| x.max(0.0)),
        }
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        let grad_i = grad.clone() * &mem.input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        // clip it if it's too big
        (grad_i.mapv(|x| if x.abs() > 100.0 { 100.0 * x.signum() } else { x }), ())
    }

    fn learn_from(&mut self, _grad_w: Self::GradW) {
        // do nothing
    }
}

struct Softmax;

impl Model for Softmax {
    type I = Array1<f32>;
    type O = Array1<f32>;
    type GradI = Array1<f32>;
    type GradO = Array1<f32>;
    type GradW = ();
    type Mem = LinearLayerStorage<{ 28 * 28 }, 10>;

    fn run(&self, input: Self::I) -> Self::Mem {
        // Numerical stability adjustment
        let max_input = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values = input.mapv(|x| ((x - max_input) as f64).exp());
        let total_sum: f64 = exp_values.sum();

        LinearLayerStorage {
            input: input.clone(),
            output: exp_values.mapv(|x| (x / total_sum) as f32),
        }
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        let output = mem.output;
        let mut grad_i = Array1::zeros(10);
        
        for i in 0..10 {
            let mut sum = 0.0;
            for j in 0..10 {
                if i == j {
                    sum += output[i] * (1.0 - output[i]);
                } else {
                    sum += -output[i] * output[j];
                }
            }
            grad_i[i] = grad[i] * sum;
        }

        // Gradient clipping
        let grad_i = grad_i.mapv(|x| if x.abs() > 100.0 { 100.0 * x.signum() } else { x });
        
        (grad_i, ())
    }

    fn learn_from(&mut self, _grad_w: Self::GradW) {
        // No learning required for Softmax layer
    }
}

/* 
    Below is a model that just chains two models together
*/

struct Chain<M1, M2> where M1: Model, M2: Model<I = M1::O, GradI = M1::GradO> {
    model1: M1,
    model2: M2,
}

impl<M1, M2> Chain<M1, M2> where M1: Model, M2: Model<I = M1::O, GradI = M1::GradO> {
    fn new(model1: M1, model2: M2) -> Self {
        Chain {
            model1,
            model2,
        }
    }

    fn then<M>(self, model: M) -> Chain<Self, M> where M: Model<I = M2::O, GradI = M2::GradO> {
        Chain {
            model1: self,
            model2: model,
        }
    }
}

impl<M1, M2> Model for Chain<M1, M2> where M1: Model, M2: Model<I = M1::O, GradI = M1::GradO> {
    type I = M1::I;
    type GradI = M1::GradI;
    type O = M2::O;
    type GradO = M2::GradO;
    type GradW = (M1::GradW, M2::GradW);
    type Mem = (M1::Mem, M2::Mem);

    fn run(&self, input: Self::I) -> Self::Mem {
        let mem1 = self.model1.run(input);
        let mem2 = self.model2.run(mem1.get_output());
        (mem1, mem2)
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        let (mem1, mem2) = mem;
        let (grad2, grad_w2) = self.model2.backprop(grad, mem2);
        let (grad1, grad_w1) = self.model1.backprop(grad2, mem1);
        (grad1, (grad_w1, grad_w2))
    }

    fn learn_from(&mut self, grad_w: Self::GradW) {
        let (grad_w1, grad_w2) = grad_w;
        self.model1.learn_from(grad_w1);
        self.model2.learn_from(grad_w2);
    }
}

fn main() {
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((50_000, 28 * 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| (*x as f32) / 256.0);
    //println!("{:#.1?}\n", train_data.slice(s![image_num, ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array1<u8> = Array1::from_shape_vec(50_000, trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as u8);
    //println!("The first digit is a {:?}", train_labels.slice(s![image_num]));

    let test_data = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| (*x as f32) / 256.0);

    let test_labels: Array1<u8> = Array1::from_shape_vec(10_000, tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as u8);

    // initialize weights to random small values
    let mut rng = rand::thread_rng();

    let mut model = ErrorLoss {
        model: LinearLayer::<{28 * 28}, 10>::new(),
        loss_fn: |output, target| {
            let mut loss = 0.0;
            for (p, t) in output.iter().zip(target.iter()) {
                loss += (p - t).powi(2);
            }
            loss
        },
        loss_deriv: |output, target| {
            output - target
        },
    };

    let mut model = ErrorLossFactory::cross_entropy(
        Chain::new(LinearLayer::<{ 28 * 28 }, 20>::new(), ReLU::<20>)
            .then(LinearLayer::<20, 10>::new_identity(true))
            .then(ReLU::<10>)
            .then(Softmax)
    );

    ///////////////////////////////////////// PROGAM
    // single layer
    let mut model = ErrorLossFactory::least_squares(
        Chain::new(LinearLayer::<{ 28 * 28 }, 60>::new_identity(false), ReLU::<50>)
        .then(LinearLayer::<60, 20>::new_identity(true))
        .then(ReLU::<20>)
        .then(LinearLayer::<20, 10>::new_identity(true))
        .then(ReLU::<10>)
    );

    for epoch in 0..5000 {
        // iterate through train data
        // print "..........." (length 100)
        // let repeated = ".".repeat(100);
        // println!("{}", repeated);
        // iterate through test data

        let mut correct = 0;
        let mut total = 0;

        print!("{}: Accuracy: ", epoch);
        for (i, image) in test_data.outer_iter().enumerate() {
            if i % 2 != 0 {
                continue;
            }

            let label = test_labels[i] as usize;

            // pass image to model
            let input_data = image.to_owned();
            let mem = model.run(input_data);

            // calculate loss
            let output = mem.get_output();

            let max_digit = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;
            if max_digit == label {
                correct += 1;
            }
            total += 1;
        }

        println!("{}", (correct as f32) / (total as f32));

        for (i, image) in train_data.outer_iter().enumerate() {
            // if i % 500 == 0 {
            //     print!(".");
            //     io::stdout().flush().unwrap();
            // }

            // chance to skip
            if rng.gen_range(0..10) < 9 {
                continue;
            }

            let label = train_labels[i] as usize;

            // pass image to model
            let input_data = image.to_owned();
            let mem = model.run(input_data);

            // one hot encoding
            let target = Array1::from_shape_fn(10, |i| if i == label { 1.0 } else { 0.0 });

            // calculate loss
            let output = mem.clone().get_output();
            // let loss = (model.loss_fn)(output.clone(), Array1::from_elem(10, 0.0));
            let grad = (model.loss_deriv)(output.clone(), target);
            model.learn_from(model.backprop(grad, mem).1);
        }
        // println!();

    }
    ///////////////////////////////////////// PROGAM

    // // one layer for testing
    // let mut model = ErrorLossFactory::least_squares(
    //     LinearLayer::<2, 2>::new()
    // );

    // model.model.weights = arr2(&[[-2.0, 0.0], [0.0, -1.0]]);
    // model.model.biases = arr1(&[0.0, 0.0]);

    // for i in 0..20 {
    //     let x = model.run(arr1(&[1.0, 1.0]));
    //     let y = x.output.clone();
    //     let r = (model.loss_deriv)(y.clone(), arr1(&[-2.0, -1.0]));
    //     let t = model.backprop(r.clone(), x);
    //     model.learn_from(t.1.clone());
    //     // println!("output is {:?}: ", y);
    //     // println!("memory is {:?}", r);
    //     // println!("gradients is {:} {:?}", t.0, t.1);
    // }

    // // println!("r is {:?}", r);
    // // println!("t is {:?}", t);

}

// trainer
struct Trainer<M>
    where M: Model<I = Array1<f32>, GradI = Array1<f32>, O = Array1<f32>, GradO = Array1<f32>> {
    model: M,
    training_data: Array2<f32>,
    test_data: Array2<u8>,
    training_labels: Array1<f32>,
    test_labels: Array1<u8>,
    current_epoch: usize,
    current_training_index: usize,
}

impl<M> Trainer<M>
    where M: Model<I = Array1<f32>, GradI = Array1<f32>, O = Array1<f32>, GradO = Array1<f32>> {}

// SAVING AND LOADING