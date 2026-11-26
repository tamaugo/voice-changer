//! Visible name and logo. The crate and executable stay `voice-changer`.

use eframe::egui;

pub const APP_NAME: &str = "Lark";
pub const INSTALLER_ASSET: &str = "Lark-Setup.exe";
pub const GITHUB_REPO: &str = "tamaugo/voice-changer";

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub struct Logo {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Logo {
    pub fn load() -> Self {
        let image = image::load_from_memory(include_bytes!("../../../assets/lark.png"))
            .expect("assets/lark.png")
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        Self {
            rgba: image.into_raw(),
            width,
            height,
        }
    }

    pub fn color_image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgba_unmultiplied(
            [self.width as usize, self.height as usize],
            &self.rgba,
        )
    }

    pub fn icon(&self) -> egui::IconData {
        egui::IconData {
            rgba: self.rgba.clone(),
            width: self.width,
            height: self.height,
        }
    }
}
