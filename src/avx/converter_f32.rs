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
use crate::avx::lnf::_mm256_ln_ps;
use crate::{CepstrumConverter, CepstrumSample};
use num_complex::Complex;
use num_traits::{Float, Zero};
use std::arch::x86_64::*;

pub(crate) struct AvxCepstrumF32 {
    pub(crate) norm: f32,
}

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn avx_zip(a: __m256, other: __m256) -> [__m256; 2] {
    let r0 = _mm256_unpacklo_ps(a, other);
    let r1 = _mm256_unpackhi_ps(a, other);
    let xy0 = _mm256_permute2f128_ps::<32>(r0, r1);
    let xy1 = _mm256_permute2f128_ps::<49>(r0, r1);
    [xy0, xy1]
}

impl AvxCepstrumF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn convert_impl(&self, in_out: &mut [Complex<f32>]) {
        let eps = _mm256_set1_ps(f32::EPSILON);
        let norm = _mm256_set1_ps(self.norm);
        for chunk_4 in in_out.as_chunks_mut::<8>().0.iter_mut() {
            let q0 = unsafe { _mm256_loadu_ps(chunk_4.as_ptr().cast()) };
            let q1 = unsafe { _mm256_loadu_ps(chunk_4[4..].as_ptr().cast()) };
            let q0_sqr = _mm256_mul_ps(q0, q0);
            let q1_sqr = _mm256_mul_ps(q1, q1);
            let mut lane = _mm256_add_ps(_mm256_hadd_ps(q0_sqr, q1_sqr), eps);
            const MASK: i32 = shuffle(3, 1, 2, 0);
            lane = _mm256_castpd_ps(_mm256_permute4x64_pd::<MASK>(_mm256_castps_pd(lane)));
            let ln_value = _mm256_mul_ps(_mm256_ln_ps(lane), norm);
            let [v0, v1] = avx_zip(ln_value, _mm256_setzero_ps());
            unsafe {
                _mm256_storeu_ps(chunk_4.as_mut_ptr().cast(), v0);
                _mm256_storeu_ps(chunk_4[4..].as_mut_ptr().cast(), v1);
            }
        }
        let rem1 = in_out.as_chunks_mut::<4>().1;
        for chunk_4 in rem1.as_chunks_mut::<4>().0.iter_mut() {
            let q0 = unsafe { _mm256_loadu_ps(chunk_4.as_ptr().cast()) };
            let q0_sqr = _mm256_mul_ps(q0, q0);
            let mut lane = _mm256_add_ps(_mm256_hadd_ps(q0_sqr, _mm256_setzero_ps()), eps);
            const MASK: i32 = shuffle(3, 1, 2, 0);
            lane = _mm256_castpd_ps(_mm256_permute4x64_pd::<MASK>(_mm256_castps_pd(lane)));
            let ln_value = _mm256_mul_ps(_mm256_ln_ps(lane), norm);
            let [v0, _] = avx_zip(ln_value, _mm256_setzero_ps());
            unsafe {
                _mm256_storeu_ps(chunk_4.as_mut_ptr().cast(), v0);
            }
        }
        let rem2 = in_out.as_chunks_mut::<4>().1;
        for dst in rem2.iter_mut() {
            let q = f32::mul_add(dst.re, dst.re, dst.im * dst.im);
            *dst = Complex::new((q + f32::epsilon()).c_log() * self.norm, f32::zero());
        }
    }
}

impl CepstrumConverter<f32> for AvxCepstrumF32 {
    fn convert(&self, in_out: &mut [Complex<f32>]) {
        unsafe {
            self.convert_impl(in_out);
        }
    }
}
