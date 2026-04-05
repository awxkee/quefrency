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
use crate::mla::fmla;
use crate::neon::lnf::vlnq_f32;
use crate::{CepstrumConverter, CepstrumSample};
use num_complex::Complex;
use num_traits::{Float, Zero};
use std::arch::aarch64::*;

pub(crate) struct NeonCepstrumF32 {
    pub(crate) norm: f32,
}

impl NeonCepstrumF32 {
    #[target_feature(enable = "neon")]
    fn convert_impl(&self, in_out: &mut [Complex<f32>]) {
        let eps = vdupq_n_f32(f32::EPSILON);
        let norm = vdupq_n_f32(self.norm);
        for chunk_4 in in_out.as_chunks_mut::<4>().0.iter_mut() {
            let q0 = unsafe { vld1q_f32(chunk_4.as_ptr().cast()) };
            let q1 = unsafe { vld1q_f32(chunk_4[2..].as_ptr().cast()) };
            let q0_sqr = vmulq_f32(q0, q0);
            let q1_sqr = vmulq_f32(q1, q1);
            let lane = vaddq_f32(vpaddq_f32(q0_sqr, q1_sqr), eps);
            let ln_value = vmulq_f32(vlnq_f32(lane), norm);
            let v0 = vzip1q_f32(ln_value, vdupq_n_f32(0.));
            let v1 = vzip2q_f32(ln_value, vdupq_n_f32(0.));
            unsafe {
                vst1q_f32(chunk_4.as_mut_ptr().cast(), v0);
            }
            unsafe {
                vst1q_f32(chunk_4[2..].as_mut_ptr().cast(), v1);
            }
        }
        let rem1 = in_out.as_chunks_mut::<4>().1;
        for chunk_2 in rem1.as_chunks_mut::<2>().0.iter_mut() {
            let q0 = unsafe { vld1q_f32(chunk_2.as_ptr().cast()) };
            let q0_sqr = vmulq_f32(q0, q0);
            let lane = vaddq_f32(vpaddq_f32(q0_sqr, vdupq_n_f32(0.)), eps);
            let ln_value = vmulq_f32(vlnq_f32(lane), norm);
            let v0 = vzip1q_f32(ln_value, vdupq_n_f32(0.));
            unsafe {
                vst1q_f32(chunk_2.as_mut_ptr().cast(), v0);
            }
        }
        let rem2 = in_out.as_chunks_mut::<2>().1;
        for dst in rem2.iter_mut() {
            let q = fmla(dst.re, dst.re, dst.im * dst.im);
            *dst = Complex::new((q + f32::epsilon()).c_log() * self.norm, f32::zero());
        }
    }
}

impl CepstrumConverter<f32> for NeonCepstrumF32 {
    fn convert(&self, in_out: &mut [Complex<f32>]) {
        unsafe {
            self.convert_impl(in_out);
        }
    }
}
