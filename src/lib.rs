/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod basic_converter;
mod default_plan;
mod err;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;

use crate::default_plan::DefaultCepstrumPlan;
pub use err::QuefrencyError;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use pxfm::{f_log, f_logf};
use std::ops::{Add, Mul};
use std::sync::Arc;
use zaft::{C2RFftExecutor, R2CFftExecutor, Zaft};

pub trait CepstrumExecutor<T> {
    fn execute(&self, signal: &mut [T]) -> Result<(), QuefrencyError>;
    fn execute_with_scratch(
        &self,
        signal: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), QuefrencyError>;
    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), QuefrencyError>;
    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), QuefrencyError>;
    fn scratch_size(&self) -> usize;
}

pub type Cepstrum<T> = Arc<dyn CepstrumExecutor<T> + Send + Sync>;

trait FftFactory {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, QuefrencyError>;
    fn make_c2r(len: usize) -> Result<Arc<dyn C2RFftExecutor<Self> + Send + Sync>, QuefrencyError>;
}

impl FftFactory for f32 {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, QuefrencyError> {
        Zaft::make_r2c_fft_f32(len).map_err(|x| QuefrencyError::FftError(x.to_string()))
    }

    fn make_c2r(len: usize) -> Result<Arc<dyn C2RFftExecutor<Self> + Send + Sync>, QuefrencyError> {
        Zaft::make_c2r_fft_f32(len).map_err(|x| QuefrencyError::FftError(x.to_string()))
    }
}

impl FftFactory for f64 {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, QuefrencyError> {
        Zaft::make_r2c_fft_f64(len).map_err(|x| QuefrencyError::FftError(x.to_string()))
    }

    fn make_c2r(len: usize) -> Result<Arc<dyn C2RFftExecutor<Self> + Send + Sync>, QuefrencyError> {
        Zaft::make_c2r_fft_f64(len).map_err(|x| QuefrencyError::FftError(x.to_string()))
    }
}

trait CepstrumConverter<T> {
    fn convert(&self, in_out: &mut [Complex<T>]);
}

trait CepstrumFactory {
    fn cepstrum_converter(norm: Self) -> Arc<dyn CepstrumConverter<Self> + Send + Sync>;
}

impl CepstrumFactory for f32 {
    fn cepstrum_converter(norm: f32) -> Arc<dyn CepstrumConverter<Self> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonCepstrumF32;
            Arc::new(NeonCepstrumF32 { norm })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxCepstrumF32;
                return Arc::new(AvxCepstrumF32 { norm });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::basic_converter::BasicCepstrumConverter;
            Arc::new(BasicCepstrumConverter { norm })
        }
    }
}

impl CepstrumFactory for f64 {
    fn cepstrum_converter(norm: f64) -> Arc<dyn CepstrumConverter<Self> + Send + Sync> {
        use crate::basic_converter::BasicCepstrumConverter;
        Arc::new(BasicCepstrumConverter { norm })
    }
}

trait CepstrumSample:
    Copy
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + MulAdd<Self, Output = Self>
    + 'static
    + Send
    + Sync
    + FftFactory
    + Float
    + CepstrumFactory
{
    fn c_log(&self) -> Self;
}

impl CepstrumSample for f32 {
    #[inline(always)]
    fn c_log(&self) -> Self {
        f_logf(*self)
    }
}

impl CepstrumSample for f64 {
    #[inline(always)]
    fn c_log(&self) -> Self {
        f_log(*self)
    }
}

fn make_cepstrum_impl<T: CepstrumSample>(
    len: usize,
    normalize: bool,
) -> Result<Cepstrum<T>, QuefrencyError>
where
    f64: AsPrimitive<T>,
{
    let fft_forward = T::make_r2c(len)?;
    let fft_backward = T::make_c2r(len)?;
    let converter = T::cepstrum_converter(if normalize {
        (1f64 / (len as f64)).as_()
    } else {
        1f64.as_()
    });
    let inner_scratch_size = fft_backward
        .complex_scratch_length()
        .max(fft_forward.complex_scratch_length());
    let working_scratch_size = len / 2 + 1;
    Ok(Arc::new(DefaultCepstrumPlan {
        fft_forward,
        fft_backward,
        converter,
        inner_scratch_size,
        working_scratch_size,
        execution_length: len,
    }))
}

/// Creates a cepstrum plan for `f32` samples.
pub fn make_cepstrum_f32(len: usize, normalize: bool) -> Result<Cepstrum<f32>, QuefrencyError> {
    make_cepstrum_impl(len, normalize)
}

/// Creates a cepstrum plan for `f64` samples.
pub fn make_cepstrum_f64(len: usize, normalize: bool) -> Result<Cepstrum<f64>, QuefrencyError> {
    make_cepstrum_impl(len, normalize)
}

#[cfg(test)]
mod tests {
    use crate::CepstrumSample;
    use crate::mla::fmla;
    use num_complex::Complex;
    use num_traits::{Float, Zero};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use zaft::Zaft;

    const EPSILON_F32: f32 = 1e-4;

    fn brute_force_cepstrum_f32(arr: &[f32], normalize: bool) -> Vec<f32> {
        let fft_r2c = Zaft::make_r2c_fft_f32(arr.len()).unwrap();
        let fft_c2r = Zaft::make_c2r_fft_f32(arr.len()).unwrap();
        let mut output = vec![Complex::zero(); arr.len() / 2 + 1];
        let mut cepstrum_output = vec![f32::zero(); arr.len()];
        fft_r2c.execute(&arr, &mut output).unwrap();

        if normalize {
            let norm = (1f64 / arr.len() as f64) as f32;
            for dst in output.iter_mut() {
                let q = fmla(dst.re, dst.re, dst.im * dst.im);
                *dst = Complex::new((q + f32::epsilon()).c_log() * norm, f32::zero());
            }
        } else {
            for dst in output.iter_mut() {
                let q = fmla(dst.re, dst.re, dst.im * dst.im);
                *dst = Complex::new((q + f32::epsilon()).c_log(), f32::zero());
            }
        }

        fft_c2r.execute(&output, &mut cepstrum_output).unwrap();
        cepstrum_output
    }

    fn random_signal(rng: &mut StdRng, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    }

    fn random_positive_signal(rng: &mut StdRng, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| rng.random_range(0.01f32..1.0f32))
            .collect()
    }

    #[test]
    fn test_random_signal_unnormalized_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(42);
        let len = 256;
        let input = random_signal(&mut rng, len);

        let expected = brute_force_cepstrum_f32(&input, false);

        let cepstrum = crate::make_cepstrum_f32(len, false).unwrap();
        let mut actual = vec![0.0f32; len];
        cepstrum.execute_into(&input, &mut actual).unwrap();

        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPSILON_F32,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_random_signal_normalized_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(43);
        let len = 256;
        let input = random_signal(&mut rng, len);

        let expected = brute_force_cepstrum_f32(&input, true);

        let cepstrum = crate::make_cepstrum_f32(len, true).unwrap();
        let mut actual = vec![0.0f32; len];
        cepstrum.execute_into(&input, &mut actual).unwrap();

        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPSILON_F32,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_positive_signal_unnormalized_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(99);
        let len = 128;
        let input = random_positive_signal(&mut rng, len);

        let expected = brute_force_cepstrum_f32(&input, false);

        let cepstrum = crate::make_cepstrum_f32(len, false).unwrap();
        let mut actual = vec![0.0f32; len];
        cepstrum.execute_into(&input, &mut actual).unwrap();

        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPSILON_F32,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_positive_signal_normalized_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(100);
        let len = 128;
        let input = random_positive_signal(&mut rng, len);

        let expected = brute_force_cepstrum_f32(&input, true);

        let cepstrum = crate::make_cepstrum_f32(len, true).unwrap();
        let mut actual = vec![0.0f32; len];
        cepstrum.execute_into(&input, &mut actual).unwrap();

        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPSILON_F32,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_positive_signal_normalized_matches_brute_force_in_place() {
        let mut rng = StdRng::seed_from_u64(100);
        let len = 128;
        let mut input = random_positive_signal(&mut rng, len);

        let expected = brute_force_cepstrum_f32(&input, true);

        let cepstrum = crate::make_cepstrum_f32(len, true).unwrap();
        cepstrum.execute(&mut input).unwrap();

        for (i, (a, b)) in input.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPSILON_F32,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_random_length() {
        let mut rng = StdRng::seed_from_u64(100);
        for _ in 0..5 {
            let len = rng.random_range(1..512);
            let mut input = random_positive_signal(&mut rng, len);

            let expected = brute_force_cepstrum_f32(&input, true);

            let cepstrum = crate::make_cepstrum_f32(len, true).unwrap();
            cepstrum.execute(&mut input).unwrap();

            for (i, (a, b)) in input.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < EPSILON_F32,
                    "mismatch at index {}: actual={}, expected={} failed for size {len}",
                    i,
                    a,
                    b
                );
            }
        }
    }
}
