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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
fn _mm256_ilogb2kq_ps(d: __m256) -> __m256i {
    _mm256_sub_epi32(
        _mm256_and_si256(
            _mm256_srli_epi32::<23>(_mm256_castps_si256(d)),
            _mm256_set1_epi32(0xff),
        ),
        _mm256_set1_epi32(0x7f),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
fn _mm256_ldexp3kq_ps(x: __m256, n: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_add_epi32(
        _mm256_castps_si256(x),
        _mm256_slli_epi32::<23>(n),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
/// Negates signed 32 bytes integer
fn _mm256_neg_epi32(d: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_setzero_si256(), d)
}

#[inline]
#[target_feature(enable = "avx2")]
/// If mask then `true_vals` otherwise `false_val`
fn _mm256_select_ps(mask: __m256, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
/// Returns flag value is zero
fn _mm256_eqzero_ps(d: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_EQ_OS>(d, _mm256_set1_ps(0.))
}

#[inline]
#[target_feature(enable = "avx2")]
/// Returns flag value is Infinity
fn _mm256_isinf_ps(d: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_EQ_OS>(_mm256_abs_ps(d), _mm256_set1_ps(f32::INFINITY))
}

#[inline]
#[target_feature(enable = "avx2")]
/// Returns true flag if value is NaN
fn _mm256_isnan_ps(d: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_NEQ_OS>(d, d)
}

#[inline]
#[target_feature(enable = "avx2")]
/// Returns flag value is lower than zero
fn _mm256_ltzero_ps(d: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_LT_OS>(d, _mm256_set1_ps(0.))
}

#[inline]
#[target_feature(enable = "avx2")]
/// Modulus operator for f32
fn _mm256_abs_ps(f: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_andnot_si256(
        _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
        _mm256_castps_si256(f),
    ))
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
/// Method that computes ln skipping Inf, Nan checks, error bound *ULP 1.5*
fn _mm256_ln_fast_ps(d: __m256) -> __m256 {
    const LN_POLY_1_F: f32 = 2f32;
    const LN_POLY_2_F: f32 = 0.6666677f32;
    const LN_POLY_3_F: f32 = 0.40017125f32;
    const LN_POLY_4_F: f32 = 0.28523374f32;
    const LN_POLY_5_F: f32 = 0.23616748f32;
    let n = _mm256_ilogb2kq_ps(_mm256_mul_ps(d, _mm256_set1_ps(1f32 / 0.75f32)));
    let a = _mm256_ldexp3kq_ps(d, _mm256_neg_epi32(n));
    let ones = _mm256_set1_ps(1f32);
    let x = _mm256_div_ps(_mm256_sub_ps(a, ones), _mm256_add_ps(a, ones));
    let x2 = _mm256_mul_ps(x, x);
    let mut u = _mm256_set1_ps(LN_POLY_5_F);
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(LN_POLY_4_F));
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(LN_POLY_3_F));
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(LN_POLY_2_F));
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(LN_POLY_1_F));
    _mm256_fmadd_ps(
        _mm256_set1_ps(std::f32::consts::LN_2),
        _mm256_cvtepi32_ps(n),
        _mm256_mul_ps(x, u),
    )
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
/// Computes natural logarithm for an argument *ULP 1.5*
pub(crate) fn _mm256_ln_ps(d: __m256) -> __m256 {
    let mut res = _mm256_ln_fast_ps(d);
    // d == 0 || d == Inf -> Inf
    res = _mm256_select_ps(_mm256_eqzero_ps(d), _mm256_set1_ps(f32::NEG_INFINITY), res);
    res = _mm256_select_ps(_mm256_isinf_ps(d), _mm256_set1_ps(f32::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = _mm256_select_ps(
        _mm256_or_ps(_mm256_ltzero_ps(d), _mm256_isnan_ps(d)),
        _mm256_set1_ps(f32::NAN),
        res,
    );
    res
}
