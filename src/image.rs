//! PNG IO: decode to planar RGB f32 in [0,1], encode 8-bit RGB output, and the
//! CROP that IFAN's preprocessing contract requires.
//!
//! The reference reads with `cv2.imread`, flips BGR to RGB and divides by 255 -
//! the same float pipeline this module implements on the host side of the
//! boundary. The engine has no BGR step: a PNG is RGB already, and the flip in
//! `read_frame` only exists to undo OpenCV's channel order.
//!
//! THE CROP IS NORMATIVE, NOT A CONVENIENCE. Upstream's `refine_image` crops the
//! image to a multiple of 8 rather than padding it, and BOTH `predict.py` and
//! `eval.py` do it, so it is the deployment contract every published number was
//! produced under. An image whose sides are not multiples of 8 is therefore
//! evaluated on a CROP here too, and the engine says so on stderr in the
//! non-quiet case. Silently padding instead would produce a different image
//! from upstream on 6 of every 8 sizes; silently squaring it off would produce
//! one on all of them.
//!
//! What is deliberately NOT reproduced: `predict.py` downscales an image whose
//! longest side exceeds 1920 with `cv2.INTER_AREA` before running. That is a
//! memory accommodation from 2021, not part of the model's contract, and the
//! family's engines deconvolve at full size (the user's explicit call). The
//! divergence is documented in README.md's "differences from upstream" so
//! nobody has to discover it by comparing outputs.
use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// A planar RGB image: `data` is [3][h][w], contiguous, values in [0,1].
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Image {
        Image { w, h, data: vec![0.0; 3 * w * h] }
    }

    #[inline]
    pub fn plane(&self, c: usize) -> &[f32] {
        let hw = self.w * self.h;
        &self.data[c * hw..(c + 1) * hw]
    }

    /// The image as interleaved RGB bytes, clamped and rounded the way the
    /// reference rounds: `(v * 255 + 0.5).floor().min(255)`, which is what
    /// `np.uint8(np.clip(v, 0, 1) * 255 + 0.5)` does for the rounding mode
    /// numpy uses. A plain `as u8` cast truncates and is visibly wrong on
    /// midtones, so the `+ 0.5` is not decoration.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let hw = self.w * self.h;
        let mut out = vec![0u8; hw * 3];
        for i in 0..hw {
            for c in 0..3 {
                let v = self.data[c * hw + i].clamp(0.0, 1.0) * 255.0;
                out[i * 3 + c] = (v + 0.5).floor().min(255.0) as u8;
            }
        }
        out
    }

    /// Crop the right and bottom edges so both sides are multiples of `val`.
    ///
    /// `refine_image`, as upstream writes it: `image[:, :h - h % val, :w - w % val]`.
    /// Returns the cropped dimensions alongside the cropped planes, and reports
    /// `false` when nothing was cropped so the caller can stay silent.
    ///
    /// A side shorter than `val` would crop to ZERO, which upstream does not
    /// guard against and which produces an empty tensor that fails obscurely
    /// three layers down. This returns an error instead: a 4-pixel-wide PNG is
    /// a mistake worth naming.
    pub fn refine(&mut self, val: usize) -> Result<bool, String> {
        if val == 0 {
            return Err("refine value must be non-zero".into());
        }
        let (nh, nw) = (self.h - self.h % val, self.w - self.w % val);
        if nh == 0 || nw == 0 {
            return Err(format!(
                "image {}x{} is smaller than one refine tile ({}): it would crop to {}x{}, \
                 and IFAN has no padding path - use a larger image",
                self.w, self.h, val, nw, nh
            ));
        }
        if nh == self.h && nw == self.w {
            return Ok(false);
        }
        let mut data = vec![0.0f32; 3 * nw * nh];
        for c in 0..3 {
            for y in 0..nh {
                let src = c * self.w * self.h + y * self.w;
                let dst = c * nw * nh + y * nw;
                data[dst..dst + nw].copy_from_slice(&self.data[src..src + nw]);
            }
        }
        self.data = data;
        self.w = nw;
        self.h = nh;
        Ok(true)
    }
}

/// Decode a PNG (8/16-bit, gray/RGB/RGBA) into planar RGB f32 in [0,1].
pub fn load_rgb(path: &str) -> Result<Image, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    load_rgb_stream(file).map_err(|e| format!("{}: {}", path, e))
}

/// The same decode, from any reader - used for `-i -`, where the PNG arrives on
/// stdin. The whole stream is buffered because the decoder needs the header
/// before it can size the frame.
pub fn load_rgb_stream<R: Read>(src: R) -> Result<Image, String> {
    let decoder = png::Decoder::new(src);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("unsupported png colour type {:?}", other)),
    };
    let sixteen = info.bit_depth == png::BitDepth::Sixteen;
    let sample_scale = match info.bit_depth {
        png::BitDepth::Eight => 1.0 / 255.0,
        png::BitDepth::Sixteen => 1.0 / 65535.0,
        other => return Err(format!("unsupported png bit depth {:?}", other)),
    };
    let bytes = &buf[..info.buffer_size()];
    // PNG 16-bit samples are big endian at the byte level, which is what the
    // shift below reproduces; the crate hands them over as raw bytes.
    let sample = |px: usize, c: usize| -> f32 {
        let idx = px * channels + c;
        if sixteen {
            let b0 = bytes[idx * 2] as u16;
            let b1 = bytes[idx * 2 + 1] as u16;
            ((b0 << 8) | b1) as f32 * sample_scale
        } else {
            bytes[idx] as f32 * sample_scale
        }
    };
    let mut img = Image::new(w, h);
    let hw = w * h;
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            let (r, g, b) = match channels {
                1 | 2 => {
                    let v = sample(px, 0);
                    (v, v, v)
                }
                _ => (sample(px, 0), sample(px, 1), sample(px, 2)),
            };
            img.data[px] = r;
            img.data[hw + px] = g;
            img.data[2 * hw + px] = b;
        }
    }
    Ok(img)
}

/// Write 8-bit RGB.
pub fn save_rgb(path: &str, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {}: {}", path, e))?;
    save_rgb_stream(BufWriter::new(file), w, h, rgb).map_err(|e| format!("{}: {}", path, e))
}

/// The same encode, to any writer - used for `-o -`, where the PNG goes to
/// stdout. The writer is flushed here so the bytes are out before the process
/// exits.
pub fn save_rgb_stream<W: Write>(dst: W, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    assert_eq!(rgb.len(), w * h * 3);
    let mut enc = png::Encoder::new(dst, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgb).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}
