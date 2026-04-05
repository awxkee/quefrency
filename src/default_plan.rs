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
use crate::err::try_vec;
use crate::{CepstrumConverter, CepstrumExecutor, CepstrumSample, QuefrencyError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;
use zaft::{C2RFftExecutor, R2CFftExecutor};

pub(crate) struct DefaultCepstrumPlan<T> {
    pub(crate) fft_forward: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    pub(crate) fft_backward: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
    pub(crate) converter: Arc<dyn CepstrumConverter<T> + Send + Sync>,
    pub(crate) inner_scratch_size: usize,
    pub(crate) working_scratch_size: usize,
    pub(crate) execution_length: usize,
}

impl<T: CepstrumSample> CepstrumExecutor<T> for DefaultCepstrumPlan<T> {
    fn execute(&self, signal: &mut [T]) -> Result<(), QuefrencyError> {
        let mut scratch = try_vec![Complex::<T>::zero(); self.scratch_size()];
        self.execute_with_scratch(signal, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        signal: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), QuefrencyError> {
        let scratch_size = self.scratch_size();
        if scratch.len() < scratch_size {
            return Err(QuefrencyError::InvalidScratchSize {
                expected: scratch_size,
                actual: scratch.len(),
            });
        }
        if !signal.len().is_multiple_of(self.execution_length) {
            return Err(QuefrencyError::DataIsNotMultipleOfLength {
                expected_multiple_of: self.execution_length,
                actual: signal.len(),
            });
        }
        let (output_scratch, fft_scratch) = scratch.split_at_mut(self.working_scratch_size);

        for signal in signal.chunks_exact_mut(self.execution_length) {
            self.fft_forward
                .execute_with_scratch(signal, output_scratch, fft_scratch)
                .map_err(|x| QuefrencyError::FftError(x.to_string()))?;

            self.converter.convert(output_scratch);

            self.fft_backward
                .execute_with_scratch(output_scratch, signal, fft_scratch)
                .map_err(|x| QuefrencyError::FftError(x.to_string()))?;
        }

        Ok(())
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), QuefrencyError> {
        let mut scratch = try_vec![Complex::<T>::zero(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), QuefrencyError> {
        let scratch_size = self.scratch_size();
        if scratch.len() < scratch_size {
            return Err(QuefrencyError::InvalidScratchSize {
                expected: scratch_size,
                actual: scratch.len(),
            });
        }
        if input.len() != output.len() {
            return Err(QuefrencyError::InputOutputSizesMismatch {
                c1: input.len(),
                c2: output.len(),
            });
        }
        if !output.len().is_multiple_of(self.execution_length) {
            return Err(QuefrencyError::DataIsNotMultipleOfLength {
                expected_multiple_of: self.execution_length,
                actual: output.len(),
            });
        }

        let (output_scratch, fft_scratch) = scratch.split_at_mut(self.working_scratch_size);

        for (input, output) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.fft_forward
                .execute_with_scratch(input, output_scratch, fft_scratch)
                .map_err(|x| QuefrencyError::FftError(x.to_string()))?;

            self.converter.convert(output_scratch);

            self.fft_backward
                .execute_with_scratch(output_scratch, output, fft_scratch)
                .map_err(|x| QuefrencyError::FftError(x.to_string()))?;
        }

        Ok(())
    }

    fn scratch_size(&self) -> usize {
        self.inner_scratch_size + self.working_scratch_size
    }
}
