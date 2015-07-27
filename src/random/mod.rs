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
use rand::{Rng, SeedableRng, StdRng};
use tensor::Tensor;
use rand::distributions::range::SampleRange;
use num::traits::Float;
use Numeric;
use math;
use std::f64;

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
            where T: Numeric + SampleRange {
        let mut t = Tensor::zeros(shape);
        for i in 0..t.size() {
            t[i] = self.rng.gen_range::<T>(low, high);
        }
        t
    }

    /// Generates a tensor by independently drawing samples from a standard normal
    pub fn normal<T>(&mut self, shape: &[usize]) -> Tensor<T>
            where T: Numeric + SampleRange + Float {
        let u1 = self.uniform(T::zero(), T::one(), shape);
        let u2 = self.uniform(T::zero(), T::one(), shape);

        let minustwo = Tensor::fscalar(-2.0);
        let twopi = Tensor::fscalar(2.0 * f64::consts::PI);

        math::sqrt(math::ln(u1) * &minustwo) * &math::cos(u2 * &twopi)
    }
}
