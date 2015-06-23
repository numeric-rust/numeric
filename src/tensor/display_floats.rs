use tensor::Tensor;
use std::fmt;

// Display for floats
macro_rules! add_impl {
    ($t:ty, $name:tt) => (
        impl fmt::Display for Tensor<$t> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let mv = &self.data[..];
                let mut s = "".to_string();
                if self.ndim() == 1 {
                    s.push_str("[");
                    for i in 0..self.shape[0] {
                        if i > 0 {
                            s.push_str(" ");
                        }
                        s = format!("{}{:6.2}", s, mv[i]);
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
                            s = format!("{}{:6.2}", s, self.get(i, j));
                        }
                        if i == self.shape[0] - 1 {
                            s.push_str("]]");
                        } else {
                            s.push_str("]\n");
                        }
                    }
                } else {
                    s = format!("{}Tensor({:?})", $name, self.shape);
                }
                write!(f, "{}", s)
            }
        }
    )

}

add_impl!(f32, "Single");
add_impl!(f64, "Double");
