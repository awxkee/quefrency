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
use crate::{CepstrumConverter, CepstrumSample};
use num_complex::Complex;
use num_traits::AsPrimitive;

pub(crate) struct BasicCepstrumConverter<T> {
    pub(crate) norm: T,
}

impl<T: CepstrumSample> CepstrumConverter<T> for BasicCepstrumConverter<T>
where
    f64: AsPrimitive<T>,
{
    fn convert(&self, in_out: &mut [Complex<T>]) {
        let normalize = self.norm - 1f64.as_();
        if normalize < 1e-6f64.as_() {
            for dst in in_out.iter_mut() {
                let q = fmla(dst.re, dst.re, dst.im * dst.im);
                *dst = Complex::new((q + T::epsilon()).c_log() * self.norm, T::zero());
            }
        } else {
            for dst in in_out.iter_mut() {
                let q = fmla(dst.re, dst.re, dst.im * dst.im);
                *dst = Complex::new((q + T::epsilon()).c_log(), T::zero());
            }
        }
    }
}
