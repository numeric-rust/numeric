//! The random module provides methods of randomizing tensors.
//!
//! Here is an example:
//!
//! ```
//! use numeric::Tensor;
//! use numeric::random::RandomState;
//!
//! // Create a random state with seed 1234 (it has to be mutable)
//! let mut rs = RandomState::new(1234);
//!
//! let t = rs.uniform(0.0, 1.0, &[3, 3]);
//! println!("{}", t);
//! //  0.820987  0.93044 0.507159
//! //  0.603939  0.31157 0.383515
//! //  0.702227 0.346673 0.737954
//! // [Tensor<f64> of shape 3x3]
//! ```
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::uniform::SampleRange;
use num::traits::Float;
use std::f64;

use tensor::{Tensor, AxisIndex};
use traits::NumericTrait;
use math;

pub struct RandomState {
    rng: StdRng,
}

impl RandomState {
    /// Creates a new `RandomState` object with the given seed. The object needs to be captured
    /// as mutable in order to draw samples from it (since its internal state changes).
    pub fn new(seed: usize) -> RandomState {
        let ss: &[_] = &[seed];
        RandomState{rng: SeedableRng::from_seed(ss)}
    }

    /// Generates a tensor by independently drawing samples from a uniform distribution in the 
    /// range [`low`, `high`). This is appropriate for integer types as well.
    pub fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Tensor<T>
            where T: NumericTrait + SampleRange<T> {
        let mut t = Tensor::zeros(shape);
        {
            let n = t.size();
            let mut data = t.slice_mut();
            for i in 0..n {
                data[i] = self.rng.gen_range::<T>(low, high);
            }
        }
        t
    }

    /// Generates a tensor by independently drawing samples from a standard normal.
    pub fn normal<T>(&mut self, shape: &[usize]) -> Tensor<T>
            where T: NumericTrait + SampleRange<T> + Float {
        let u1 = self.uniform(T::zero(), T::one(), shape);
        let u2 = self.uniform(T::zero(), T::one(), shape);

        let minustwo = Tensor::fscalar(-2.0);
        let twopi = Tensor::fscalar(2.0 * f64::consts::PI);

        math::sqrt(math::ln(u1) * &minustwo) * &math::cos(u2 * &twopi)
    }

    /// Shuffle tensor in-place along its first axis. This uses the modern version of the
    /// Fisher-Yates algorithm.
    pub fn shuffle<T>(&mut self, a: &mut Tensor<T>) -> ()
            where T: Copy {
        if a.ndim() == 1 && a.size() > 0 {
            a.canonize_inplace();
            let n = a.dim(0);
            {
                let mut data = a.mem_slice_mut();
                for i in (1..n).rev() {
                    let j = self.rng.gen_range::<usize>(0, i + 1);
                    data.swap(i, j);
                }
            }
        } else if a.ndim() >= 2 && a.size() > 0 {
            a.canonize_inplace();
            let mut row_shape: Vec<usize> = Vec::with_capacity(a.ndim() - 1);
            for i in 1..a.ndim() {
                row_shape.push(a.shape()[i]);
            }
            let n = a.dim(0);
            let mut row1: Tensor<T> = Tensor::empty(&row_shape[..]);
            let mut row2: Tensor<T> = Tensor::empty(&row_shape[..]);
            for i in (1..n).rev() {
                let j: usize = self.rng.gen_range::<usize>(0, i + 1);
                // TODO: Can be made faster.
                // Rust requires us to have two buffers instead of just one, but
                // we can probably make it even faster by shuffling indices first
                // and then move memory around. A specialized swap rows function
                // could also make things faster.
                row1.set(&a.index(&[AxisIndex::Index(i as isize)]));
                row2.set(&a.index(&[AxisIndex::Index(j as isize)]));
                a.index_set(&[AxisIndex::Index(i as isize)], &row2);
                a.index_set(&[AxisIndex::Index(j as isize)], &row1);
            }
        }
    }
}
