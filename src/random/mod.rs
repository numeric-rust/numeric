use rand::{Rng, SeedableRng, StdRng};
use tensor::Tensor;
use num::traits::Num;
use rand::distributions::range::SampleRange;

pub struct RandomState {
    rng: StdRng,
}

impl RandomState {
    pub fn new(seed: usize) -> RandomState {
        let ss: &[_] = &[seed];
        RandomState{rng: SeedableRng::from_seed(ss)}
    }

    /// Generates a tensor by independently drawing samples from a uniform in the range [`low`,
    /// `high`).
    pub fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Tensor<T>
            where T: Copy + Num + PartialOrd + SampleRange {
        let mut t = Tensor::zeros(shape);
        for i in 0..t.size() {
            t[i] = self.rng.gen_range::<T>(low, high);
        }
        t
    }
}
