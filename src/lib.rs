//! Create for sampling data from an autoregressive random process.
//!
//! For more information, please see https://en.wikipedia.org/wiki/Autoregressive_model.
//!
//! ```
//! let mut ar = autoregressive::univariate::Autoregressive::new(0.0, 1.0, &[0.5]);
//!
//! // Sample next value
//! let n = ar.step();
//!
//! // Take 10 values as an iterator
//! let n10 = ar.take(10).collect::<Vec<f32>>();
//! ```

/// Univariate AR model
///
pub mod univariate {
    use num_traits::Float;
    use rand_distr::{Distribution, StandardNormal};

    pub struct Autoregressive<F, const N: usize>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        c: F,
        x: [F; N],
        phi: [F; N],
        noise: rand_distr::Normal<F>,
    }

    impl<F, const N: usize> Autoregressive<F, N>
    where
        F: Float + std::iter::Sum,
        StandardNormal: Distribution<F>,
    {
        /// Create a new instance
        /// * `c`  parameter
        /// * `noise_variance` Variance of the white noise (epsilon)
        /// * `phi` model parameters
        pub fn new(c: F, noise_variance: F, phi: &[F; N]) -> Self {
            let x = [num_traits::identities::zero(); N];
            let noise =
                rand_distr::Normal::new(num_traits::identities::zero(), noise_variance).unwrap();
            Self {
                c,
                phi: *phi,
                x,
                noise,
            }
        }

        /// Next value from the AR model
        pub fn step(&mut self) -> F {
            let mut rng = rand::thread_rng();
            let epsilon: F = self.noise.sample(&mut rng);
            let new_x = self.c
                + self
                    .x
                    .iter()
                    .zip(self.phi.iter())
                    .map(|(x, p)| *x * *p)
                    .sum::<F>()
                + epsilon;
            if !self.x.is_empty() {
                self.x.rotate_right(1);
                self.x[0] = new_x;
            }
            new_x
        }
    }

    impl<F, const N: usize> Iterator for Autoregressive<F, N>
    where
        F: Float + std::iter::Sum,
        StandardNormal: Distribution<F>,
    {
        type Item = F;

        fn next(&mut self) -> Option<Self::Item> {
            Some(self.step())
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn bounded() {
        const NUM: usize = 1_000_000;

        let ar = super::univariate::Autoregressive::new(0.0, 1.0, &[]);
        let avg = ar.take(NUM).sum::<f32>() / (NUM as f32);
        assert!(avg.abs() < 1.0);

        let ar = super::univariate::Autoregressive::new(0.0, 1.0, &[0.3]);
        let avg = ar.take(NUM).sum::<f32>() / (NUM as f32);
        assert!(avg.abs() < 1.0);

        let ar = super::univariate::Autoregressive::new(0.0, 1.0, &[0.9]);
        let avg = ar.take(NUM).sum::<f32>() / (NUM as f32);
        assert!(avg.abs() < 1.0);

        let ar = super::univariate::Autoregressive::new(0.0, 1.0, &[0.3, 0.3]);
        let avg = ar.take(NUM).sum::<f32>() / (NUM as f32);
        assert!(avg.abs() < 1.0);

        let ar = super::univariate::Autoregressive::new(0.0, 1.0, &[0.9, -0.8]);
        let avg = ar.take(NUM).sum::<f32>() / (NUM as f32);
        assert!(avg.abs() < 1.0);
    }
}
