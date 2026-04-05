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
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
fn vilogb2kq_f32(d: float32x4_t) -> int32x4_t {
    vsubq_s32(
        vandq_s32(
            vshrq_n_s32::<23>(vreinterpretq_s32_f32(d)),
            vdupq_n_s32(0xff),
        ),
        vdupq_n_s32(0x7f),
    )
}

#[inline]
#[target_feature(enable = "neon")]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
fn vldexp3kq_f32(x: float32x4_t, n: int32x4_t) -> float32x4_t {
    vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32::<23>(n)))
}

#[inline]
#[target_feature(enable = "neon")]
/// Returns true flag if value is Infinity
fn visinfq_f32(d: float32x4_t) -> uint32x4_t {
    vceqq_f32(d, vdupq_n_f32(f32::INFINITY))
}

#[inline]
#[target_feature(enable = "neon")]
/// Returns true flag if value is NaN
fn visnanq_f32(d: float32x4_t) -> uint32x4_t {
    vmvnq_u32(vceqq_f32(d, d))
}

#[inline]
#[target_feature(enable = "neon")]
fn vmlafq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vfmaq_f32(c, b, a)
}

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn vlnq_f32(d: float32x4_t) -> float32x4_t {
    let mut res = vlnq_fast_f32(d);
    // d == 0 -> -Inf
    res = vbslq_f32(vceqzq_f32(d), vdupq_n_f32(f32::NEG_INFINITY), res);
    // d == Inf -> Inf
    res = vbslq_f32(visinfq_f32(d), vdupq_n_f32(f32::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = vbslq_f32(
        vorrq_u32(vcltzq_f32(d), visnanq_f32(d)),
        vdupq_n_f32(f32::NAN),
        res,
    );
    res
}

/// Method that computes ln skipping Inf, Nan checks
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn vlnq_fast_f32(d: float32x4_t) -> float32x4_t {
    const LN_POLY_1_F: f32 = 2f32;
    const LN_POLY_2_F: f32 = 0.6666677f32;
    const LN_POLY_3_F: f32 = 0.40017125f32;
    const LN_POLY_4_F: f32 = 0.28523374f32;
    const LN_POLY_5_F: f32 = 0.23616748f32;
    let n = vilogb2kq_f32(vmulq_n_f32(d, 1f32 / 0.75f32));
    let a = vldexp3kq_f32(d, vnegq_s32(n));
    let ones = vdupq_n_f32(1f32);
    let x = vdivq_f32(vsubq_f32(a, ones), vaddq_f32(a, ones));
    let x2 = vmulq_f32(x, x);
    let mut u = vdupq_n_f32(LN_POLY_5_F);
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_4_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_3_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_2_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_1_F));
    vmlafq_f32(
        vdupq_n_f32(std::f32::consts::LN_2),
        vcvtq_f32_s32(n),
        vmulq_f32(x, u),
    )
}
