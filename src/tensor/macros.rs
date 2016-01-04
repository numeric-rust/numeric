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
    (@count) => (0);
    (@count $head:tt $($tail:tt)*) => (1 + tensor!(@count $($tail)*));
    ($elem:expr; $n:expr) => (
        $crate::Tensor::new(vec![$elem; $n])
    );
    ($($x:expr),*) => (
        $crate::Tensor::new(vec![$($x),*])
    );
    ($($x:expr,)*) => (
        tensor![$($x),*]
    );
    ($($($x:expr),*;)*) => (
        $crate::Tensor::new(vec![$($($x),*),*]).reshape(&[tensor!(@count $([$($x),*])*), -1])
    );
    ($($($x:expr),*);*) => (
        tensor![$($($x),*;)*]
    );
}
