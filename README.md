# Computing cepstrum in Rust

Adding to project

```bash
cargo add quefrency
```

# Usage

```rust
let signal = vec![0.; 512];
let cepstrum = crate::make_cepstrum_f32(512, true).unwrap();
cepstrum.execute(&mut input).unwrap();
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
