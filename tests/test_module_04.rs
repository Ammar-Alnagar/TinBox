use tinbox::model::loader::dequantize_e4m3_byte;

#[test]
fn test_module_04_e4m3_dequantization_values() {
    // Normal 1.0: sign=0, exp=7 (0111), mant=0 (000) -> 0b00111000
    let one = dequantize_e4m3_byte(0b00111000);
    assert!((one - 1.0).abs() < 1e-6);

    // Normal -1.0: sign=1, exp=7 (0111), mant=0 (000) -> 0b10111000
    let neg_one = dequantize_e4m3_byte(0b10111000);
    assert!((neg_one - (-1.0)).abs() < 1e-6);

    // Subnormal smallest: exp=0, mant=1 -> (1/8) * 2^(-6) = 0.001953125
    let subnormal = dequantize_e4m3_byte(0b00000001);
    assert!((subnormal - 0.001953125).abs() < 1e-8);
}

#[test]
fn test_module_04_e4m3_nan_representation() {
    // NaN in E4M3 is exp=0b1111 (15), mant=0b111 (7) -> 0b01111111
    let nan_val = dequantize_e4m3_byte(0b01111111);
    assert!(nan_val.is_nan());

    let neg_nan_val = dequantize_e4m3_byte(0b11111111);
    assert!(neg_nan_val.is_nan());
}
