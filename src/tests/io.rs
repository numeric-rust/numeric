#[allow(unused_imports)] use tensor::Tensor;
#[allow(unused_imports)] use io;
#[allow(unused_imports)] use std::path::Path;
#[allow(unused_imports)] use std::env;
#[allow(unused_imports)] use std::fs;

#[test]
fn hdf5() {
    // We can't have multiple HDF5 tests, since these functions are not thread-safe! 
    // This is why we have to string all of this together in serial, since otherwise the tests
    // might run concurrently. This is in other words the only test that is allowed to interract
    // with with libhdf5.

    let mut path = env::temp_dir();
    path.push("numeric.h5");
    if path.exists() {
        assert!(fs::remove_file(&path).is_ok());
    }

    {
        let t: Tensor<f64> = Tensor::range(10);

        let res = t.save_hdf5(&path);
        assert!(res.is_ok());
        assert!(path.exists());

        let t2 = io::load_hdf5_as_f64(&path, "/data").unwrap();

        assert!(t == t2);
        assert!(fs::remove_file(&path).is_ok());
    }

    {
        let t: Tensor<i16> = Tensor::range(1000).reshape(&[50, 20]);

        let res = t.save_hdf5(&path);
        assert!(res.is_ok());
        assert!(path.exists());

        let t2 = io::load_hdf5_as_i16(&path, "/data").unwrap();

        assert!(t == t2);
        assert!(fs::remove_file(&path).is_ok());
    }
}

