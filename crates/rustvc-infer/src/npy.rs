//! A small reader for the float32 `.npy` files the exporter writes.
//!
//! Only little-endian float32 arrays are accepted. That is the bank of
//! content features used for retrieval. NumPy writes the header; we do not
//! link NumPy.

use std::fs;
use std::path::Path;

use crate::dsp::FeatureBank;

pub fn load_feature_bank(path: &Path) -> Result<FeatureBank, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let (rows, cols, data) = parse_f32(&bytes)?;
    FeatureBank::from_rows(cols, data).ok_or_else(|| {
        format!(
            "{} has shape {rows}x{cols}, which is not a feature matrix",
            path.display()
        )
    })
}

fn parse_f32(bytes: &[u8]) -> Result<(usize, usize, Vec<f32>), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a npy file".into());
    }
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (len, 10)
    } else if major == 2 {
        if bytes.len() < 12 {
            return Err("truncated npy header".into());
        }
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (len, 12)
    } else {
        return Err(format!("unsupported npy version {major}"));
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err("truncated npy header".into());
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| "npy header is not utf-8".to_string())?;
    if !header.contains("<f4") && !header.contains("|f4") && !header.contains("f4") {
        return Err("feature bank must be float32".into());
    }
    if header.contains("fortran_order': True") || header.contains("fortran_order\": True") {
        return Err("fortran-order npy is not supported".into());
    }
    let (rows, cols) = parse_shape(header)?;
    let data_bytes = &bytes[header_end..];
    if data_bytes.len() < rows * cols * 4 {
        return Err("npy payload is shorter than its shape".into());
    }
    let mut data = Vec::with_capacity(rows * cols);
    for chunk in data_bytes[..rows * cols * 4].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((rows, cols, data))
}

fn parse_shape(header: &str) -> Result<(usize, usize), String> {
    let start = header.find("shape").ok_or("npy header has no shape")? + "shape".len();
    let tuple = header[start..]
        .split_once('(')
        .ok_or("npy shape is not a tuple")?
        .1;
    let inside = tuple.split_once(')').ok_or("npy shape is not closed")?.0;
    let numbers: Vec<usize> = inside
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect();
    match numbers.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        [cols] => Ok((1, *cols)),
        _ => Err(format!("expected a 1-D or 2-D shape, found {inside}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_two_by_two_matrix() {
        // Written by NumPy as np.save(..., np.array([[1, 2], [3, 4]], np.float32)).
        let bytes = include_bytes!("../../../python/testdata/matrix_f32.npy");
        let (rows, cols, data) = parse_f32(bytes).unwrap();
        assert_eq!((rows, cols), (2, 2));
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
