use tensor::Tensor;
use num::traits::cast;
use Numeric;

impl<T: Numeric> Tensor<T> {
    /// Returns a new tensor with the elements converted to the selected type.
    ///
    /// ```
    /// use numeric::Tensor;
    ///
    /// let tdouble = Tensor::new(vec![1.0f64, 2.0, 3.0]);
    /// let tsingle = tdouble.convert::<f32>();
    /// ```
    pub fn convert<D: Numeric>(&self) -> Tensor<D> {
        let mut t = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            t[i] = cast(self[i]).unwrap();
        }
        t
    }

    /// Short-hand for `convert::<f32>()`.
    pub fn to_f32(&self) -> Tensor<f32> {
        self.convert::<f32>()
    }

    /// Short-hand for `convert::<f64>()`.
    pub fn to_f65(&self) -> Tensor<f64> {
        self.convert::<f64>()
    }
}
