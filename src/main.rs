use mnist::*;
use ndarray::prelude::*;
use rand::Rng;
// TODO; serde for model loading

// impl Model<ModelOutput, ModelResult> for StupidModel {
//     fn predict(&self, input: ArrayView1<f32>) -> ModelOutput {

//         let mut probabilities = self.weights.dot(&input) + &self.biases;

//         // apply softmax
//         let total_exp: f32 = probabilities
//             .iter()
//             .map(|x| x.exp())
//             .sum();

//         probabilities.mapv_inplace(|x| x.exp() / total_exp);
//         return ModelOutput {
//             result: ModelResult {
//                 probabilities,
//             },
//             layer_outputs: [input.to_owned()],
//         };
//     }

//     fn learn_from(&mut self, output: ModelOutput, target: ModelResult) {
//         // find cross entropy loss
//         let mut loss = 0.0;
//         for (i, (p, t)) in output.result.probabilities
//             .iter()
//             .zip(target.probabilities.iter())
//             .enumerate() {
//             loss -= t * p.ln();
//         }

//         // find how much the output layer needs to change (cross entropy)
//         let output_error = (0..10).map(|i| 0.0).collect::<Array1<f32>>();
//     }
// }

trait Memory {
    type O;
    fn get_output(self) -> Self::O;
}

trait Model {
    type I;
    type GradI;
    type O;
    type GradO;
    type GradW;
    type Mem: Memory<O=Self::O>;

    fn run(&self, input: Self::I) -> Self::Mem;
    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW);
    fn learn_from(&mut self, grad_w: Self::GradW);
}

struct LinearLayer<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

struct LinearLayerStorage<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    input: Array1<f32>,
    output: Array1<f32>,
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> LinearLayer<INPUT_DIM, OUTPUT_DIM> {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((OUTPUT_DIM, INPUT_DIM), |_| rng.gen_range(-0.03..0.03));
        let biases = Array1::from_elem(OUTPUT_DIM, 0.0);
        LinearLayer {
            weights,
            biases,
        }
    }
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> Memory
for LinearLayerStorage<INPUT_DIM, OUTPUT_DIM> {
    type O = Array1<f32>;
    fn get_output(self) -> Array1<f32> {
        self.output
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
        // self.weights += &(0.01 * &grad_w.0);
        // self.biases += &(0.01 * &grad_w.1);
        self.weights += &grad_w.0.mapv(|x| x * 0.01);
        self.biases += &grad_w.1.mapv(|x| x * 0.01);
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        // iterate over the weights and calculate the gradient
        let output = mem.output;
        let input = mem.input;

        let grad_w = Array2::from_shape_fn((OUTPUT_DIM, INPUT_DIM), |(i, j)| {
            grad[i] * input[j]
        });

        let grad_i = self.weights.t().dot(&grad);

        let grad_b = grad;

        (grad_i, (grad_w, grad_b))
    }
}

// mean squared error loss
struct ErrorLoss<M, FL, FD> where
    M: Model<O=Array1<f32>, GradO=Array1<f32>>,
    FL: Fn(Self, Array1<f32>, Array1<f32>) -> f32,
    FD: Fn(Self, Array1<f32>, Array1<f32>) -> Array1<f32>,
{
    model: M,
    loss_fn: FL,
    loss_deriv: FD,
}

impl<M, I, GradI, GradW, Mem, FL, FD> Model for ErrorLoss<M, FL, FD> where
    M: Model<
        O = Array1<f32>,
        GradO = Array1<f32>,
        I = I,
        GradI = GradI,
        GradW = GradW,
        Mem = Mem,
    >,
    Mem: Memory<O=Array1<f32>>,
    FL: Fn(Self, Array1<f32>, Array1<f32>) -> f32,
    FD: Fn(Self, Array1<f32>, Array1<f32>) -> Array1<f32>,
{

    type I = I;
    type GradI = GradI;
    type O = Array1<f32>;
    type GradO = Array1<f32>;
    type GradW = GradW;
    type Mem = Mem;

    fn run(&self, input: Self::I) -> Self::Mem {
        todo!()
    }

    fn backprop(&self, grad: Self::GradO, mem: Self::Mem) -> (Self::GradI, Self::GradW) {
        todo!()
    }

    fn learn_from(&mut self, grad_w: Self::GradW) {
        todo!()
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

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((50_000, 28 * 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| (*x as f32) / 256.0);
    //println!("{:#.1?}\n", train_data.slice(s![image_num, ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array1<f32> = Array1::from_shape_vec(50_000, trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    //println!("The first digit is a {:?}", train_labels.slice(s![image_num]));

    let _test_data = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| (*x as f32) / 256.0);

    let _test_labels: Array1<f32> = Array1::from_shape_vec(10_000, tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    // initialize weights to random small values
    let mut rng = rand::thread_rng();

    // let weights = Array1::from_shape_fn(10, |_| Array1::from_shape_fn(28 * 28, |_| rng.gen_range(-0.01..0.01)));
    let weights = Array2::from_shape_fn((10, 28 * 28), |_| rng.gen_range(-0.03..0.03));

    // // create model
    // let mut model = StupidModel {
    //     weights: weights,
    //     biases: Array1::from_elem(10, 0.0),
    // };

    let mut model = 

    // // iterate through train data
    // for (i, image) in train_data.outer_iter().enumerate() {
    //     let label = train_labels[i] as usize;
    //     // pass image to model
    //     let output = model.predict(image);
    //     let target = ModelResult {
    //         probabilities: Array1::from_shape_fn(10, |j| if j == label { 1.0 } else { 0.0 }),
    //     };
    //     let result = output.result.probabilities.clone();
    //     model.learn_from(output, target);

    //     let max_digit = result.iter().enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .unwrap().0;
    //     println!("Predicted: {:?}", max_digit);
    // }
}
