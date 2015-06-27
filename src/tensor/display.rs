use tensor::Tensor;
use std::fmt;

// Display for floats
macro_rules! add_impl {
    ($t:ty, $name:tt) => (
        impl fmt::Display for Tensor<$t> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let mv = &self.data[..];
                let mut ret = "\n".to_string();
                if self.ndim() <= 2 {
                    // Pre-generate all strings
                    let mut ss: Vec<String> = Vec::with_capacity(self.size());
                    let mut longest: usize = 0;
                    for i in 0..self.size() {
                        let s = format!("{}", mv[i]);
                        ss.push(s.to_string());
                        if s.len() > longest {
                            longest = s.len();
                        }
                    }

                    if self.ndim() == 1 {
                        for i in 0..self.shape[0] {
                            let s = &ss[i];
                            for _ in 0..(1 + longest - s.len()) {
                                ret.push_str(" ");
                            }

                            ret = format!("{}{}", ret, s);
                        }
                    } else if self.ndim() == 2 {
                        let s0 = self.strides()[0];
                        //ret.push_str("[[");
                        for i in 0..self.shape[0] {
                            if i > 0 {
                                //ret.push_str(" [");
                            }
                            for j in 0..self.shape[1] {
                                let s = &ss[i * s0 + j];
                                for _ in 0..(1 + longest - s.len()) {
                                    ret.push_str(" ");
                                }
                                ret = format!("{}{}", ret, s);
                            }
                            if i == self.shape[0] - 1 {
                                //ret.push_str("]]");
                            } else {
                                //ret.push_str("]\n");
                                ret.push_str("\n");
                            }
                        }
                    }
                } else {
                    // For now, higher order are simply omitted
                    ret.push_str("...");
                }

                // Format shape
                // TODO: Is there an implode/join function?
                let mut shape_str = "".to_string();
                for i in 0..self.shape.len() {
                    if i == 0 {
                        shape_str.push_str(&format!("{}", self.shape[i])[..]);
                    } else {
                        shape_str.push_str(&format!("x{}", self.shape[i])[..]);
                    }
                }

                write!(f, "{}\n[Tensor<{}> of shape {}]", ret, $name, shape_str)
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
