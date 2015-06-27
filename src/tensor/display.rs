use tensor::Tensor;
use std::fmt;

// Display for floats
macro_rules! add_impl {
    ($t:ty, $name:tt) => (
        impl fmt::Display for Tensor<$t> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let mv = &self.data[..];
                let mut s = format!("Tensor [{}]:\n", $name);
                if self.ndim() == 1 {
                    s.push_str("[");
                    for i in 0..self.shape[0] {
                        if i > 0 {
                            s.push_str(" ");
                        }
                        s = format!("{}{}", s, mv[i]);
                    }
                    s.push_str("]");
                } else if self.ndim() == 2 {
                    s.push_str("[[");
                    for i in 0..self.shape[0] {
                        if i > 0 {
                            s.push_str(" [");
                        }
                        for j in 0..self.shape[1] {
                            if j > 0 {
                                s.push_str(" ");
                            }
                            s = format!("{}{}", s, self.get(i, j));
                        }
                        if i == self.shape[0] - 1 {
                            s.push_str("]]");
                        } else {
                            s.push_str("]\n");
                        }
                    }
                } else {
                    s = format!("Tensor({:?}, type={})", self.shape, $name);
                }
                write!(f, "{}", s)
            }
        }
    )

}

add_impl!(f32,   "f32");
add_impl!(f64,   "f64");
add_impl!(usize, "usize");
add_impl!(u8,    "u8");
add_impl!(u16,   "u16");
add_impl!(u32,   "u32");
add_impl!(u64,   "u64");
add_impl!(isize, "isize");
add_impl!(i8,    "i8");
add_impl!(i16,   "i16");
add_impl!(i32,   "i32");
add_impl!(i64,   "i64");
add_impl!(bool,  "bool");
