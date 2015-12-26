//use Tensor;
//use tensor;


/// Macro for creating vectors and matrices.
///
/// To use this macro, import Numeric as follows:
///
/// ```text
/// #[macro_use(tensor)]
/// extern crate numeric;
/// ```
///
/// # Examples
///
/// 1D tensor (vector):
///
/// ```
/// # #[macro_use] extern crate numeric; fn main() {
/// let x = tensor![1.0, 2.0, 3.0];
/// # assert!(x == numeric::Tensor::new(vec![1.0, 2.0, 3.0]));
/// # }
/// ```
///
/// 2D tensor (matrix):
///
/// ```
/// # #[macro_use] extern crate numeric; use numeric::tensor; fn main() {
/// let x = tensor![1, 0; 3, 2; 5, 4];
/// # assert!(x == numeric::Tensor::new(vec![1, 0, 3, 2, 5, 4]).reshape(&[3, 2]));
/// # }
/// ```
///
/// 1D tensor filled with a single value:
///
/// ```
/// # #[macro_use] extern crate numeric; use numeric::tensor; fn main() {
/// let x = tensor![2.0; 5];
/// # assert!(x == numeric::Tensor::new(vec![2.0, 2.0, 2.0, 2.0, 2.0]));
/// # }
/// ```
#[macro_export]
macro_rules! tensor {
    ($elem:expr; $n:expr) => (
        numeric::Tensor::filled(&[$n], $elem)
    );
    ($($x:expr),*) => (
        numeric::Tensor::new(vec![$($x),*])
    );
    ($($($x:expr),*);*) => ({
        let mut v = Vec::new();
        let mut n = 0;
        $(
            n += 1;
            $(v.push($x);)*
        )*
        numeric::Tensor::new(v).reshape(&[n, -1])
    });
}
