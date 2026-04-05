#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex;

#[derive(arbitrary::Arbitrary, Debug)]
struct Target {
    size: u16,
    re: f32,
    normalize: bool,
}

fuzz_target!(|data: Target| {
    if data.size == 0 || data.size > 15100 {
        return;
    }
    let executor = quefrency::make_cepstrum_f32(data.size as usize, data.normalize).unwrap();
    let mut chunk = vec![data.re; data.size as usize];
    executor.execute(&mut chunk).unwrap();
    let mut test_target = vec![data.re; data.size as usize];
    executor.execute_into(&chunk, &mut test_target).unwrap();
});
