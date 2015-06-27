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
//! // [[  0.82   0.93   0.51]
//! //  [  0.60   0.31   0.38]
//! //  [  0.70   0.35   0.74]]
//! ```
use rand::{Rng, SeedableRng, StdRng};
use tensor::Tensor;
use rand::distributions::range::SampleRange;
use Numeric;

pub struct RandomState {
    rng: StdRng,
}

impl RandomState {
    pub fn new(seed: usize) -> RandomState {
        let ss: &[_] = &[seed];
        RandomState{rng: SeedableRng::from_seed(ss)}
    }

    /// Generates a tensor by independently drawing samples from a uniform in the range [`low`,
    /// `high`). This is appropriate for integer types as well.
    pub fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Tensor<T>
            where T: Numeric + SampleRange {
        let mut t = Tensor::zeros(shape);
        for i in 0..t.size() {
            t[i] = self.rng.gen_range::<T>(low, high);
        }
        t
    }
}
