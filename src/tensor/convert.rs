use tensor::Tensor;
use num::traits::cast;
use traits::NumericTrait;

impl<T: NumericTrait> Tensor<T> {
    /// Returns a new tensor with the elements converted to the selected type.
    ///
    /// ```
    /// # #[macro_use] extern crate numeric; fn main() {
    /// let tdouble = tensor![1.0f64, 2.0, 3.0];
    /// let tsingle = tdouble.convert::<f32>();
    /// # }
    /// ```
    pub fn convert<D: NumericTrait>(&self) -> Tensor<D> {
        let mut t = Tensor::zeros(&self.shape);
        {
            let n = t.size();
            let mut data = t.slice_mut();
            for i in 0..n {
                data[i] = cast(data[i]).unwrap();
            }
        }
        t
    }

    /// Short-hand for `convert::<f32>()`.
    pub fn to_f32(&self) -> Tensor<f32> {
        self.convert::<f32>()
    }

    /// Short-hand for `convert::<f64>()`.
    pub fn to_f64(&self) -> Tensor<f64> {
        self.convert::<f64>()
    }
}
