//! Dark dashboard. Four actions on the home screen, and a page for each of
//! the first three. Close leaves the process.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui;
use rustvc_audio::{
    AudioEngine, DeviceList, EngineConfig, EngineStatus, Passthrough, SampleProcessor,
};
use rustvc_infer::{load_voices, InferDevice, ModelProcessor, VoiceModel, VoiceRecord};
use rustvc_train::{find_script, TrainEvent, TrainHandle, TrainRequest};
use serde::{Deserialize, Serialize};

const BG: egui::Color32 = egui::Color32::from_rgb(16, 17, 20);
const TEXT: egui::Color32 = egui::Color32::from_rgb(232, 228, 220);
const MUTED: egui::Color32 = egui::Color32::from_rgb(154, 150, 140);
const BUTTON: egui::Color32 = egui::Color32::from_rgb(36, 39, 48);
const BUTTON_HOVER: egui::Color32 = egui::Color32::from_rgb(52, 56, 68);
const BUTTON_ACTIVE: egui::Color32 = egui::Color32::from_rgb(68, 74, 90);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(196, 164, 108);
const ERROR: egui::Color32 = egui::Color32::from_rgb(214, 140, 122);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Dashboard,
    Pick,
    Train,
    Settings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct AppConfig {
    input_device: String,
    output_device: String,
    sample_rate: u32,
    block_frames: usize,
    gpu_budget_percent: u8,
    inference: String,
    cuda_device: i32,
    onnx_dll: String,
    rvc_root: String,
    python: String,
    voices_dir: String,
    selected_voice: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            output_device: String::new(),
            sample_rate: 48_000,
            block_frames: 4_800,
            gpu_budget_percent: 5,
            inference: "cuda".to_string(),
            cuda_device: 0,
            onnx_dll: String::new(),
            rvc_root: String::new(),
            python: default_python(),
            voices_dir: "voices".to_string(),
            selected_voice: "Passthrough".to_string(),
        }
    }
}

fn default_python() -> String {
    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

struct TrainForm {
    name: String,
    dataset: String,
    epochs: u32,
    batch_size: u32,
    sample_rate: u32,
    gpu: u32,
    f0: bool,
    dry_run: bool,
}

impl Default for TrainForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            dataset: String::new(),
            epochs: 200,
            batch_size: 8,
            sample_rate: 40_000,
            gpu: 0,
            f0: true,
            dry_run: true,
        }
    }
}

struct VoiceApp {
    screen: Screen,
    config: AppConfig,
    devices: DeviceList,
    voices: Vec<VoiceRecord>,
    engine: AudioEngine,
    form: TrainForm,
    job: Option<TrainHandle>,
    progress: f32,
    progress_message: String,
    log: String,
    train_error: Option<String>,
}

impl VoiceApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let config = load_config();
        let _ = fs::create_dir_all(&config.voices_dir);
        let voices = load_voices(Path::new(&config.voices_dir));
        let devices = rustvc_audio::list_devices();
        let engine = AudioEngine::start(engine_config(&config), processor_for(&config, &voices, 0));
        let mut form = TrainForm::default();
        form.dry_run = config.rvc_root.trim().is_empty();
        form.gpu = config.cuda_device.max(0) as u32;
        Self {
            screen: Screen::Dashboard,
            config,
            devices,
            voices,
            engine,
            form,
            job: None,
            progress: 0.0,
            progress_message: String::new(),
            log: String::new(),
            train_error: None,
        }
    }

    fn reload_voices(&mut self) {
        self.voices = load_voices(Path::new(&self.config.voices_dir));
    }

    fn live_rate(&self) -> u32 {
        let status = self.engine.status();
        if status.input_sample_rate > 0 {
            status.input_sample_rate
        } else {
            self.config.sample_rate
        }
    }

    fn install_voice(&mut self) {
        let rate = self.live_rate();
        self.engine
            .swap_processor(processor_for(&self.config, &self.voices, rate));
        save_config(&self.config);
    }

    fn restart_audio(&mut self) {
        let rate = self.config.sample_rate;
        self.engine = AudioEngine::start(
            engine_config(&self.config),
            processor_for(&self.config, &self.voices, rate),
        );
        save_config(&self.config);
    }

    fn poll_training(&mut self) {
        let Some(job) = self.job.as_mut() else {
            return;
        };
        let events = job.poll();
        let mut finished = false;
        for event in events {
            match event {
                TrainEvent::Progress { fraction, message } => {
                    self.progress = fraction;
                    self.progress_message = message;
                }
                TrainEvent::Log(line) => {
                    self.log.push_str(&line);
                    self.log.push('\n');
                }
                TrainEvent::Voice(path) => {
                    self.log
                        .push_str(&format!("voice manifest: {}\n", path.display()));
                    self.reload_voices();
                }
                TrainEvent::Error(message) => {
                    self.train_error = Some(message.clone());
                    self.log.push_str(&format!("error: {message}\n"));
                }
                TrainEvent::Finished { code } => {
                    finished = true;
                    if code == 0 && self.train_error.is_none() {
                        self.progress = 1.0;
                        self.progress_message = "Finished".to_string();
                    } else if self.train_error.is_none() {
                        self.train_error = Some(format!("trainer exited with status {code}"));
                    }
                    self.reload_voices();
                }
            }
        }
        if finished {
            self.job = None;
        }
    }

    fn start_training(&mut self) {
        self.train_error = None;
        self.log.clear();
        self.progress = 0.0;
        self.progress_message = "Starting".to_string();
        let Some(script) = find_script() else {
            self.train_error =
                Some("python/rvc_train.py was not found next to the source tree".into());
            return;
        };
        let dataset = PathBuf::from(self.form.dataset.trim());
        let rvc_root = {
            let root = self.config.rvc_root.trim();
            if root.is_empty() {
                None
            } else {
                Some(PathBuf::from(root))
            }
        };
        let request = TrainRequest {
            python: self.config.python.clone(),
            script,
            rvc_root,
            name: self.form.name.clone(),
            dataset,
            epochs: self.form.epochs,
            batch_size: self.form.batch_size,
            sample_rate: self.form.sample_rate,
            gpu: self.form.gpu,
            version: "v2".to_string(),
            f0: self.form.f0,
            voices_dir: PathBuf::from(&self.config.voices_dir),
            dry_run: self.form.dry_run,
            workers: std::thread::available_parallelism()
                .map(|count| count.get().clamp(1, 8) as u32)
                .unwrap_or(4),
            sleep_secs: None,
        };
        match TrainHandle::spawn(request) {
            Ok(job) => self.job = Some(job),
            Err(err) => self.train_error = Some(err.to_string()),
        }
    }
}

impl eframe::App for VoiceApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        apply_theme(&ctx);
        self.poll_training();
        if self.job.is_some() || self.engine.status().running {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
        let status = self.engine.status();
        egui::Panel::bottom("status")
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(BG)
                    .inner_margin(egui::Margin::symmetric(16, 10)),
            )
            .show(ui, |ui| {
                status_line(ui, &status, self.config.gpu_budget_percent)
            });
        egui::CentralPanel::default().show(ui, |ui| match self.screen {
            Screen::Dashboard => self.dashboard(ui, &ctx),
            Screen::Pick => self.pick(ui),
            Screen::Train => self.train(ui),
            Screen::Settings => self.settings(ui),
        });
    }
}

impl VoiceApp {
    fn dashboard(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.vertical_centered(|ui| {
            ui.add_space(48.0);
            ui.label(egui::RichText::new("voice-changer").size(40.0).color(TEXT));
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("Real-time voice conversion")
                    .size(16.0)
                    .color(MUTED),
            );
            ui.add_space(8.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(72.0, 2.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 0.0, ACCENT);
            ui.add_space(28.0);
            if action_button(ui, "Pick a voice") {
                self.screen = Screen::Pick;
            }
            ui.add_space(12.0);
            if action_button(ui, "Train a voice") {
                self.screen = Screen::Train;
            }
            ui.add_space(12.0);
            if action_button(ui, "Settings") {
                self.screen = Screen::Settings;
            }
            ui.add_space(12.0);
            if action_button(ui, "Close") {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }

    fn pick(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Pick a voice") {
            self.screen = Screen::Dashboard;
            return;
        }
        ui.label(
            egui::RichText::new("The selected voice is installed in the live audio hook.")
                .color(MUTED),
        );
        ui.add_space(12.0);
        let current = self.config.selected_voice.clone();
        let mut chosen: Option<String> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            if voice_row(
                ui,
                "Passthrough",
                "Built-in. Capture is copied to playback.",
                current == "Passthrough",
            ) {
                chosen = Some("Passthrough".to_string());
            }
            ui.add_space(8.0);
            for voice in &self.voices {
                let detail = voice_detail(voice);
                if voice_row(
                    ui,
                    &voice.manifest.name,
                    &detail,
                    current == voice.manifest.name,
                ) {
                    chosen = Some(voice.manifest.name.clone());
                }
                ui.add_space(8.0);
            }
            if self.voices.is_empty() {
                ui.label(
                    egui::RichText::new("No trained voices yet. Use Train a voice to add one.")
                        .color(MUTED),
                );
            }
        });
        if let Some(name) = chosen {
            self.config.selected_voice = name;
            self.install_voice();
        }
    }

    fn train(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Train a voice") {
            self.screen = Screen::Dashboard;
            return;
        }
        let running = self.job.is_some();
        ui.add_enabled_ui(!running, |ui| {
            egui::Grid::new("train-form")
                .num_columns(2)
                .spacing(egui::vec2(16.0, 10.0))
                .show(ui, |ui| {
                    ui.label("Name");
                    ui.add(egui::TextEdit::singleline(&mut self.form.name).desired_width(360.0));
                    ui.end_row();
                    ui.label("Dataset folder");
                    ui.add(egui::TextEdit::singleline(&mut self.form.dataset).desired_width(360.0));
                    ui.end_row();
                    ui.label("Epochs");
                    ui.add(egui::Slider::new(&mut self.form.epochs, 1..=300));
                    ui.end_row();
                    ui.label("Batch size");
                    ui.add(egui::Slider::new(&mut self.form.batch_size, 1..=32));
                    ui.end_row();
                    ui.label("Sample rate");
                    ui.horizontal(|ui| {
                        for rate in [32_000, 40_000, 48_000] {
                            ui.selectable_value(
                                &mut self.form.sample_rate,
                                rate,
                                format!("{}k", rate / 1000),
                            );
                        }
                    });
                    ui.end_row();
                    ui.label("GPU");
                    ui.add(egui::Slider::new(&mut self.form.gpu, 0..=3));
                    ui.end_row();
                });
            ui.add_space(8.0);
            ui.checkbox(&mut self.form.f0, "Use pitch (F0)");
            ui.checkbox(
                &mut self.form.dry_run,
                "Dry run (skip the GPU and only check the training flow)",
            );
        });
        ui.add_space(12.0);
        let can_start =
            !running && !self.form.name.trim().is_empty() && !self.form.dataset.trim().is_empty();
        if ui
            .add_enabled(can_start, egui::Button::new("Start training"))
            .clicked()
        {
            self.start_training();
        }
        ui.add_space(12.0);
        ui.add(
            egui::ProgressBar::new(self.progress)
                .text(&self.progress_message)
                .desired_width(520.0),
        );
        if let Some(error) = &self.train_error {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(error).color(ERROR));
        }
        ui.add_space(8.0);
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(if self.log.is_empty() {
                        "Training output will show up here."
                    } else {
                        self.log.as_str()
                    })
                    .family(egui::FontFamily::Monospace)
                    .size(13.0)
                    .color(MUTED),
                );
            });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Settings") {
            self.screen = Screen::Dashboard;
            return;
        }
        ui.label(
            egui::RichText::new(
                "Playback should be a virtual cable input, such as \"CABLE Input\". Discord then uses the cable output as its microphone. No kernel driver is installed by this app.",
            )
            .color(MUTED),
        );
        ui.add_space(12.0);
        if ui.button("Refresh devices").clicked() {
            self.devices = rustvc_audio::list_devices();
        }
        ui.add_space(8.0);
        device_picker(
            ui,
            "Input",
            &mut self.config.input_device,
            &self.devices.inputs,
            &self.devices.default_input,
        );
        device_picker(
            ui,
            "Output",
            &mut self.config.output_device,
            &self.devices.outputs,
            &self.devices.default_output,
        );
        if let Some(hint) = rustvc_audio::virtual_output_hint(&self.config.output_device) {
            ui.label(egui::RichText::new(hint).color(ACCENT));
        }
        ui.add_space(8.0);
        egui::Grid::new("settings")
            .num_columns(2)
            .spacing(egui::vec2(16.0, 10.0))
            .show(ui, |ui| {
                ui.label("Sample rate");
                ui.add(
                    egui::Slider::new(&mut self.config.sample_rate, 16_000..=48_000)
                        .step_by(1000.0),
                );
                ui.end_row();
                ui.label("Block frames");
                ui.add(egui::Slider::new(
                    &mut self.config.block_frames,
                    480..=19200,
                ));
                ui.end_row();
                ui.label("GPU budget %");
                ui.add(egui::Slider::new(
                    &mut self.config.gpu_budget_percent,
                    1..=20,
                ));
                ui.end_row();
                ui.label("Inference");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.config.inference, "cpu".to_string(), "CPU");
                    ui.selectable_value(&mut self.config.inference, "cuda".to_string(), "CUDA");
                });
                ui.end_row();
                ui.label("CUDA device");
                ui.add(egui::Slider::new(&mut self.config.cuda_device, 0..=3));
                ui.end_row();
                ui.label("ONNX Runtime library");
                ui.add(egui::TextEdit::singleline(&mut self.config.onnx_dll).desired_width(360.0));
                ui.end_row();
                ui.label("RVC checkout");
                ui.add(egui::TextEdit::singleline(&mut self.config.rvc_root).desired_width(360.0));
                ui.end_row();
                ui.label("Python");
                ui.add(egui::TextEdit::singleline(&mut self.config.python).desired_width(360.0));
                ui.end_row();
                ui.label("Voices folder");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.voices_dir).desired_width(360.0),
                );
                ui.end_row();
            });
        ui.add_space(12.0);
        if ui.button("Save and restart audio").clicked() {
            self.config.gpu_budget_percent = self.config.gpu_budget_percent.clamp(1, 100);
            self.reload_voices();
            self.restart_audio();
        }
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(
                "Live inference is built to stay near the GPU budget by running one short ONNX burst per audio block. Training uses the whole GPU and ignores the budget. CUDA on an RTX 5070 needs ONNX Runtime and PyTorch builds made with CUDA 12.8 or newer (sm_120).",
            )
            .color(MUTED),
        );
    }
}

pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 700.0])
            .with_min_inner_size([860.0, 620.0])
            .with_title("voice-changer"),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "voice-changer",
        options,
        Box::new(|cc| Ok(Box::new(VoiceApp::new(cc)))),
    )
}

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = BG;
    visuals.extreme_bg_color = egui::Color32::from_rgb(12, 13, 16);
    visuals.faint_bg_color = BUTTON;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.widgets.noninteractive.bg_fill = BG;
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.inactive.bg_fill = BUTTON;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.inactive.weak_bg_fill = BUTTON;
    visuals.widgets.hovered.bg_fill = BUTTON_HOVER;
    visuals.widgets.hovered.weak_bg_fill = BUTTON_HOVER;
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.active.bg_fill = BUTTON_ACTIVE;
    visuals.widgets.active.weak_bg_fill = BUTTON_ACTIVE;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    ctx.set_style_of(egui::Theme::Dark, style);
}

fn action_button(ui: &mut egui::Ui, label: &str) -> bool {
    ui.add_sized(
        [420.0, 64.0],
        egui::Button::new(egui::RichText::new(label).size(22.0).color(TEXT)),
    )
    .clicked()
}

fn page_header(ui: &mut egui::Ui, title: &str) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if ui
            .add(egui::Button::new(
                egui::RichText::new("Back").size(16.0).color(TEXT),
            ))
            .clicked()
        {
            back = true;
        }
        ui.add_space(8.0);
        ui.label(egui::RichText::new(title).size(28.0).color(TEXT));
    });
    ui.add_space(14.0);
    back
}

fn voice_row(ui: &mut egui::Ui, title: &str, detail: &str, selected: bool) -> bool {
    let response = ui.add_sized(
        [ui.available_width(), 64.0],
        egui::Button::new(
            egui::RichText::new(format!("{title}\n{detail}"))
                .size(16.0)
                .color(if selected { BG } else { TEXT }),
        )
        .fill(if selected { ACCENT } else { BUTTON }),
    );
    response.clicked()
}

fn voice_detail(voice: &VoiceRecord) -> String {
    let state = if voice.manifest.trained {
        "checkpoint saved"
    } else if voice.manifest.note.is_empty() {
        "not trained"
    } else {
        voice.manifest.note.as_str()
    };
    format!(
        "{} · {} Hz · {}",
        voice.manifest.version, voice.manifest.sample_rate, state
    )
}

fn device_picker(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    devices: &[String],
    default: &Option<String>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let shown = if value.is_empty() {
            default.clone().unwrap_or_else(|| "Default".to_string())
        } else {
            value.clone()
        };
        egui::ComboBox::from_id_salt(label)
            .selected_text(shown)
            .width(420.0)
            .show_ui(ui, |ui| {
                if ui.selectable_label(value.is_empty(), "Default").clicked() {
                    value.clear();
                }
                if devices.is_empty() {
                    ui.label(egui::RichText::new("No devices reported").color(MUTED));
                }
                for device in devices {
                    ui.selectable_value(value, device.clone(), device);
                }
            });
    });
}

fn status_line(ui: &mut egui::Ui, status: &EngineStatus, budget: u8) {
    if let Some(error) = &status.error {
        ui.label(egui::RichText::new(error).color(ERROR));
        return;
    }
    let input = if status.input_device.is_empty() {
        "no input"
    } else {
        status.input_device.as_str()
    };
    let output = if status.output_device.is_empty() {
        "no output"
    } else {
        status.output_device.as_str()
    };
    let rate = if status.input_sample_rate == 0 {
        "device closed".to_string()
    } else if status.input_sample_rate == status.output_sample_rate {
        format!("{} Hz", status.input_sample_rate)
    } else {
        format!(
            "{} Hz -> {} Hz",
            status.input_sample_rate, status.output_sample_rate
        )
    };
    let text = format!(
        "{}  ·  {}  ·  {}  ·  {}  ·  duty {:.1}% / {budget}%",
        status.label, input, output, rate, status.process_duty_percent
    );
    ui.label(egui::RichText::new(text).color(MUTED));
}

fn engine_config(config: &AppConfig) -> EngineConfig {
    EngineConfig {
        input_device: config.input_device.clone(),
        output_device: config.output_device.clone(),
        sample_rate: config.sample_rate,
        block_frames: config.block_frames.max(1),
    }
}

fn processor_for(
    config: &AppConfig,
    voices: &[VoiceRecord],
    device_rate: u32,
) -> Box<dyn SampleProcessor> {
    let rate = if device_rate == 0 {
        config.sample_rate
    } else {
        device_rate
    };
    if config.selected_voice.is_empty() || config.selected_voice == "Passthrough" {
        return Box::new(Passthrough);
    }
    let Some(record) = voices
        .iter()
        .find(|voice| voice.manifest.name == config.selected_voice)
    else {
        return Box::new(Passthrough);
    };
    let dll = optional_path(&config.onnx_dll);
    Box::new(ModelProcessor::new(VoiceModel::load(
        record.manifest.clone(),
        &record.directory,
        rate,
        InferDevice::parse(&config.inference, config.cuda_device),
        dll.as_deref(),
    )))
}

fn optional_path(value: &str) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn config_path() -> PathBuf {
    PathBuf::from("voice-changer.toml")
}

fn load_config() -> AppConfig {
    let path = config_path();
    let Ok(text) = fs::read_to_string(&path) else {
        return AppConfig::default();
    };
    toml::from_str(&text).unwrap_or_default()
}

fn save_config(config: &AppConfig) {
    if let Ok(text) = toml::to_string_pretty(config) {
        let _ = fs::write(config_path(), text);
    }
}
