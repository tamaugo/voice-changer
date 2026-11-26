//! Dark dashboard. Four actions on the home screen, and a page for each of
//! the first three. Close leaves the process.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui;
use rustvc_audio::{
    AudioEngine, DeviceList, EngineConfig, EngineStatus, Passthrough, SampleProcessor,
};
use rustvc_infer::{
    load_voices, resolve_voices_dir, scan_repo_voices, stage_selected, InferDevice, ModelProcessor,
    RepoVoice, VoiceModel, VoiceRecord,
};
use rustvc_train::{find_script, TrainEvent, TrainHandle, TrainRequest};
use serde::{Deserialize, Serialize};

use crate::brand::{self, Logo};
use crate::clips::{self, ClipEvent, ClipHandle};
use crate::setup::{self, SetupEvent, SetupHandle, SetupPaths};
use crate::update::{self, UpdateEvent, UpdateHandle};

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
    Setup,
    Update,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct AppConfig {
    input_device: String,
    output_device: String,
    sample_rate: u32,
    block_frames: usize,
    callback_frames: u32,
    gpu_budget_percent: u8,
    inference: String,
    cuda_device: i32,
    onnx_dll: String,
    rvc_root: String,
    python: String,
    voices_dir: String,
    selected_voice: String,
    setup_done: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            output_device: String::new(),
            sample_rate: 48_000,
            block_frames: 4_800,
            callback_frames: 480,
            gpu_budget_percent: 5,
            inference: "cuda".to_string(),
            cuda_device: 0,
            onnx_dll: String::new(),
            rvc_root: String::new(),
            python: default_python(),
            voices_dir: "voices".to_string(),
            selected_voice: "Passthrough".to_string(),
            setup_done: false,
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
    benchmark: String,
    voices_root: PathBuf,
    repo_voices: Vec<SelectableVoice>,
    selected_repo: Option<String>,
    logo: egui::TextureHandle,
    setup_job: Option<SetupHandle>,
    setup_progress: f32,
    setup_message: String,
    setup_log: String,
    setup_error: Option<String>,
    update_job: Option<UpdateHandle>,
    update_progress: f32,
    update_message: String,
    update_log: String,
    update_error: Option<String>,
    restart: bool,
    clip_job: Option<ClipHandle>,
    clip_progress: f32,
    clip_message: String,
    clip_log: String,
    clip_error: Option<String>,
}

struct SelectableVoice {
    voice: RepoVoice,
    checked: Vec<bool>,
}

impl VoiceApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let logo_image = Logo::load();
        let logo = cc.egui_ctx.load_texture(
            "lark-logo",
            logo_image.color_image(),
            egui::TextureOptions::LINEAR,
        );
        let config = load_config();
        let voices_root = voices_root_from(&config);
        let _ = fs::create_dir_all(&voices_root);
        let voices = load_voices(&voices_root);
        let devices = rustvc_audio::list_devices();
        let engine = AudioEngine::start(engine_config(&config), processor_for(&config, &voices, 0));
        let mut form = TrainForm::default();
        form.dry_run = config.rvc_root.trim().is_empty();
        form.gpu = config.cuda_device.max(0) as u32;
        let mut app = Self {
            screen: if config.setup_done {
                Screen::Dashboard
            } else {
                Screen::Setup
            },
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
            benchmark: String::new(),
            voices_root,
            repo_voices: Vec::new(),
            selected_repo: None,
            logo,
            setup_job: None,
            setup_progress: 0.0,
            setup_message: String::new(),
            setup_log: String::new(),
            setup_error: None,
            update_job: None,
            update_progress: 0.0,
            update_message: String::new(),
            update_log: String::new(),
            update_error: None,
            restart: false,
            clip_job: None,
            clip_progress: 0.0,
            clip_message: String::new(),
            clip_log: String::new(),
            clip_error: None,
        };
        app.refresh_repo_voices();
        app
    }

    fn reload_voices(&mut self) {
        self.voices_root = voices_root_from(&self.config);
        let _ = fs::create_dir_all(&self.voices_root);
        self.voices = load_voices(&self.voices_root);
        self.refresh_repo_voices();
    }

    fn refresh_repo_voices(&mut self) {
        let scanned = scan_repo_voices(&self.voices_root);
        let mut next = Vec::new();
        for voice in scanned {
            let previous = self
                .repo_voices
                .iter()
                .find(|item| item.voice.name == voice.name);
            let checked = voice
                .samples
                .iter()
                .map(|clip| {
                    previous
                        .and_then(|item| {
                            item.voice
                                .samples
                                .iter()
                                .zip(item.checked.iter())
                                .find(|(old, _)| old.path == clip.path)
                                .map(|(_, checked)| *checked)
                        })
                        .unwrap_or(true)
                })
                .collect();
            next.push(SelectableVoice { voice, checked });
        }
        if self
            .selected_repo
            .as_ref()
            .is_none_or(|name| !next.iter().any(|item| item.voice.name == *name))
        {
            self.selected_repo = next
                .iter()
                .find(|item| !item.voice.trained)
                .or_else(|| next.first())
                .map(|item| item.voice.name.clone());
        }
        self.repo_voices = next;
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
        let dataset = PathBuf::from(self.form.dataset.trim());
        let name = self.form.name.clone();
        self.spawn_training(name, dataset);
    }

    fn train_selected(&mut self) {
        let Some(name) = self.selected_repo.clone() else {
            self.train_error = Some("Select a voice from the list.".into());
            return;
        };
        let Some(item) = self.repo_voices.iter().find(|item| item.voice.name == name) else {
            self.train_error = Some("That voice is no longer in the folder.".into());
            return;
        };
        let files: Vec<PathBuf> = item
            .voice
            .samples
            .iter()
            .zip(item.checked.iter())
            .filter(|(_, checked)| **checked)
            .map(|(clip, _)| clip.path.clone())
            .collect();
        if files.is_empty() {
            self.train_error = Some("Check at least one clip.".into());
            return;
        }
        let dest = std::env::temp_dir().join(format!(
            "rustvc-train-{}-{}",
            name.replace(' ', "_"),
            std::process::id()
        ));
        if let Err(err) = stage_selected(&files, &dest) {
            self.train_error = Some(err.to_string());
            return;
        }
        self.form.name = name.clone();
        self.spawn_training(name, dest);
    }

    fn spawn_training(&mut self, name: String, dataset: PathBuf) {
        self.train_error = None;
        self.log.clear();
        self.progress = 0.0;
        self.progress_message = "Starting".to_string();
        let Some(script) = find_script() else {
            self.train_error =
                Some("python/rvc_train.py was not found next to the source tree".into());
            return;
        };
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
            name,
            dataset,
            epochs: self.form.epochs,
            batch_size: self.form.batch_size,
            sample_rate: self.form.sample_rate,
            gpu: self.form.gpu,
            version: "v2".to_string(),
            f0: self.form.f0,
            voices_dir: self.voices_root.clone(),
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

    fn poll_setup(&mut self) {
        let events = {
            let Some(job) = &self.setup_job else {
                return;
            };
            let mut events = Vec::new();
            while let Some(event) = job.poll() {
                events.push(event);
            }
            events
        };
        let mut finished = false;
        for event in events {
            match event {
                SetupEvent::Progress { fraction, message } => {
                    self.setup_progress = fraction;
                    self.setup_message = message;
                }
                SetupEvent::Log(line) => {
                    self.setup_log.push_str(&line);
                    self.setup_log.push('\n');
                }
                SetupEvent::Finished(result) => {
                    finished = true;
                    match result {
                        Ok(done) => {
                            self.setup_progress = 1.0;
                            self.setup_message = "Finished".to_string();
                            self.config.python = done.python;
                            self.config.rvc_root = done.rvc_root;
                            self.config.onnx_dll = done.onnx_dll;
                            self.config.setup_done = true;
                            self.setup_error = None;
                            save_config(&self.config);
                        }
                        Err(err) => self.setup_error = Some(err),
                    }
                }
            }
        }
        if finished {
            self.setup_job = None;
        }
    }

    fn poll_update(&mut self) {
        let events = {
            let Some(job) = &self.update_job else {
                return;
            };
            let mut events = Vec::new();
            while let Some(event) = job.poll() {
                events.push(event);
            }
            events
        };
        let mut finished = false;
        let mut restart = false;
        for event in events {
            match event {
                UpdateEvent::Progress { fraction, message } => {
                    self.update_progress = fraction;
                    self.update_message = message;
                }
                UpdateEvent::Log(line) => {
                    self.update_log.push_str(&line);
                    self.update_log.push('\n');
                }
                UpdateEvent::Voices(dir) => {
                    self.config.voices_dir = dir.display().to_string();
                    save_config(&self.config);
                    self.reload_voices();
                }
                UpdateEvent::Restart => restart = true,
                UpdateEvent::Finished(result) => {
                    finished = true;
                    if let Err(err) = result {
                        self.update_error = Some(err);
                    } else if self.update_error.is_none() {
                        self.update_progress = 1.0;
                        if self.update_message.is_empty() {
                            self.update_message = "Finished".to_string();
                        }
                    }
                }
            }
        }
        if finished {
            self.update_job = None;
        }
        if restart {
            self.restart = true;
        }
    }

    fn poll_clips(&mut self) {
        let events = {
            let Some(job) = &self.clip_job else {
                return;
            };
            let mut events = Vec::new();
            while let Some(event) = job.poll() {
                events.push(event);
            }
            events
        };
        let mut finished = false;
        for event in events {
            match event {
                ClipEvent::Progress { fraction, message } => {
                    self.clip_progress = fraction;
                    self.clip_message = message;
                }
                ClipEvent::Log(line) => {
                    self.clip_log.push_str(&line);
                    self.clip_log.push('\n');
                }
                ClipEvent::Finished(result) => {
                    finished = true;
                    match result {
                        Ok(summary) => {
                            self.clip_progress = 1.0;
                            self.clip_message =
                                format!("{} saved, {} skipped", summary.saved, summary.skipped);
                            self.clip_error = None;
                            self.refresh_repo_voices();
                        }
                        Err(err) => {
                            self.clip_error = Some(err);
                            self.refresh_repo_voices();
                        }
                    }
                }
            }
        }
        if finished {
            self.clip_job = None;
        }
    }

    fn start_clip_download(&mut self, directory: PathBuf) {
        if self.clip_job.is_some() {
            return;
        }
        self.clip_error = None;
        self.clip_log.clear();
        self.clip_progress = 0.0;
        self.clip_message = "Starting".to_string();
        self.clip_job = Some(clips::start(directory));
    }

    fn start_setup(&mut self) {
        if self.setup_job.is_some() {
            return;
        }
        self.setup_error = None;
        self.setup_log.clear();
        self.setup_progress = 0.0;
        self.setup_message = "Starting".to_string();
        let cable = self
            .devices
            .outputs
            .iter()
            .chain(self.devices.inputs.iter())
            .any(|name| name.to_ascii_lowercase().contains("cable"));
        self.setup_job = Some(setup::start(SetupPaths::new(), cable));
    }

    fn start_update(&mut self) {
        if self.update_job.is_some() {
            return;
        }
        self.update_error = None;
        self.update_log.clear();
        self.update_progress = 0.0;
        self.update_message = "Checking GitHub".to_string();
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("voice-changer"));
        self.update_job = Some(update::start(
            brand::version().to_string(),
            self.voices_root.clone(),
            exe,
        ));
    }
}

impl eframe::App for VoiceApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        apply_theme(&ctx);
        self.poll_training();
        self.poll_setup();
        self.poll_update();
        self.poll_clips();
        if self.restart {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if self.job.is_some()
            || self.setup_job.is_some()
            || self.update_job.is_some()
            || self.clip_job.is_some()
            || self.engine.status().running
        {
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
            Screen::Setup => self.setup(ui),
            Screen::Update => self.update_page(ui),
        });
    }
}

impl VoiceApp {
    fn dashboard(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Update").size(14.0).color(MUTED),
                    ))
                    .clicked()
                {
                    self.screen = Screen::Update;
                    self.start_update();
                }
                ui.label(
                    egui::RichText::new(format!("{} {}", brand::APP_NAME, brand::version()))
                        .size(14.0)
                        .color(MUTED),
                );
            });
        });
        ui.vertical_centered(|ui| {
            ui.add_space(24.0);
            ui.add(egui::Image::from_texture(&self.logo).fit_to_exact_size(egui::vec2(84.0, 84.0)));
            ui.add_space(12.0);
            ui.label(egui::RichText::new(brand::APP_NAME).size(40.0).color(TEXT));
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
        egui::ScrollArea::vertical().show(ui, |ui| {
            if self.repo_voices.is_empty() {
                ui.label(egui::RichText::new("No voices/<Name>/samples folders yet.").color(MUTED));
            }
            let mut select: Option<String> = None;
            let mut download: Option<PathBuf> = None;
            for item in &mut self.repo_voices {
                let tag = if item.voice.trained {
                    "trained"
                } else {
                    "untrained"
                };
                let chosen = self.selected_repo.as_deref() == Some(item.voice.name.as_str());
                if ui
                    .selectable_label(chosen, format!("{}  {tag}", item.voice.name))
                    .clicked()
                {
                    select = Some(item.voice.name.clone());
                }
                if chosen && item.voice.samples.is_empty() && item.voice.has_source {
                    if ui
                        .add_enabled(self.clip_job.is_none(), egui::Button::new("Download clips"))
                        .clicked()
                    {
                        download = Some(item.voice.directory.clone());
                    }
                } else if chosen {
                    for (clip, checked) in item.voice.samples.iter().zip(item.checked.iter_mut()) {
                        ui.push_id(&clip.path, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(checked, &clip.file_name);
                                if clip.over_limit {
                                    ui.label(egui::RichText::new("over 50 MB").color(ERROR));
                                }
                            });
                        });
                    }
                }
                ui.add_space(6.0);
            }
            let has_checks = self.repo_voices.iter().any(|item| {
                self.selected_repo.as_deref() == Some(item.voice.name.as_str())
                    && item.checked.iter().any(|checked| *checked)
            });
            if ui
                .add_enabled(!running && has_checks, egui::Button::new("Train"))
                .clicked()
            {
                self.train_selected();
            }
            if let Some(name) = select {
                self.selected_repo = Some(name);
            }
            if let Some(directory) = download {
                self.start_clip_download(directory);
            }
            if self.clip_job.is_some() || !self.clip_message.is_empty() {
                ui.add_space(8.0);
                ui.add(
                    egui::ProgressBar::new(self.clip_progress)
                        .text(if self.clip_message.is_empty() {
                            "Waiting"
                        } else {
                            self.clip_message.as_str()
                        })
                        .desired_width(520.0),
                );
                if let Some(error) = &self.clip_error {
                    ui.label(egui::RichText::new(error).color(ERROR));
                }
                if !self.clip_log.is_empty() {
                    job_log(ui, &self.clip_log);
                }
            }
            ui.add_space(16.0);
            ui.label(egui::RichText::new("Or a folder").color(MUTED));
            ui.add_enabled_ui(!running, |ui| {
                egui::Grid::new("train-form")
                    .num_columns(2)
                    .spacing(egui::vec2(16.0, 10.0))
                    .show(ui, |ui| {
                        ui.label("Name");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.form.name).desired_width(360.0),
                        );
                        ui.end_row();
                        ui.label("Dataset folder");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.form.dataset).desired_width(360.0),
                        );
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
            let can_start = !running
                && !self.form.name.trim().is_empty()
                && !self.form.dataset.trim().is_empty();
            if ui
                .add_enabled(can_start, egui::Button::new("Train folder"))
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
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Settings") {
            self.screen = Screen::Dashboard;
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
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
                ui.label("Callback frames");
                ui.add(egui::Slider::new(
                    &mut self.config.callback_frames,
                    128..=2048,
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
        if ui.button("Set up dependencies").clicked() {
            self.screen = Screen::Setup;
        }
        ui.add_space(8.0);
        if ui.button("Save and restart audio").clicked() {
            self.config.gpu_budget_percent = self.config.gpu_budget_percent.clamp(1, 100);
            self.reload_voices();
            self.restart_audio();
        }
        ui.add_space(12.0);
        if ui
            .button("Benchmark GPU duty")
            .on_hover_text("Times the selected voice over a few blocks. Run this on the RTX 5070.")
            .clicked()
        {
            self.benchmark = run_benchmark(&self.config, &self.voices);
        }
        if !self.benchmark.is_empty() {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(&self.benchmark).color(TEXT));
        }
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(
                "Live inference stays near the GPU budget by running one short ONNX burst per audio block. Callback frames is the WASAPI buffer when the device publishes a size range. Training uses the whole GPU and ignores the budget. CUDA on an RTX 5070 needs ONNX Runtime and PyTorch builds made with CUDA 12.8 or newer (sm_120).",
            )
            .color(MUTED),
        );
        });
    }

    fn setup(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Setup") {
            self.screen = Screen::Dashboard;
            return;
        }
        ui.label(
            egui::RichText::new(
                "Installs Git, Git LFS, Python 3.12, the RVC checkout and model files, ONNX Runtime GPU 1.26, and launches the VB-CABLE installer.",
            )
            .color(MUTED),
        );
        ui.add_space(12.0);
        for step in setup::STEPS {
            ui.label(format!("· {}", step.label));
        }
        ui.add_space(12.0);
        let running = self.setup_job.is_some();
        if ui
            .add_enabled(!running, egui::Button::new("Install"))
            .clicked()
        {
            self.start_setup();
        }
        ui.add_space(8.0);
        if ui.button("Continue").clicked() {
            self.screen = Screen::Dashboard;
        }
        ui.add_space(12.0);
        ui.add(
            egui::ProgressBar::new(self.setup_progress)
                .text(if self.setup_message.is_empty() {
                    "Waiting"
                } else {
                    self.setup_message.as_str()
                })
                .desired_width(520.0),
        );
        if let Some(error) = &self.setup_error {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(error).color(ERROR));
        }
        ui.add_space(8.0);
        job_log(ui, &self.setup_log);
    }

    fn update_page(&mut self, ui: &mut egui::Ui) {
        if page_header(ui, "Update") {
            self.screen = Screen::Dashboard;
            return;
        }
        ui.label(
            egui::RichText::new(format!(
                "{} {} checks GitHub, downloads the installer, updates voices, and restarts.",
                brand::APP_NAME,
                brand::version()
            ))
            .color(MUTED),
        );
        ui.add_space(12.0);
        if self.update_job.is_none() && ui.button("Check again").clicked() {
            self.start_update();
        }
        ui.add_space(12.0);
        ui.add(
            egui::ProgressBar::new(self.update_progress)
                .text(if self.update_message.is_empty() {
                    "Waiting"
                } else {
                    self.update_message.as_str()
                })
                .desired_width(520.0),
        );
        if let Some(error) = &self.update_error {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(error).color(ERROR));
        }
        ui.add_space(8.0);
        job_log(ui, &self.update_log);
    }
}

fn job_log(ui: &mut egui::Ui, log: &str) {
    egui::ScrollArea::vertical()
        .max_height(220.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(if log.is_empty() {
                    "Progress will show up here."
                } else {
                    log
                })
                .family(egui::FontFamily::Monospace)
                .size(13.0)
                .color(MUTED),
            );
        });
}

fn run_benchmark(config: &AppConfig, voices: &[VoiceRecord]) -> String {
    let rate = config.sample_rate.max(1);
    let block = config.block_frames.max(1);
    let mut model = match voices
        .iter()
        .find(|voice| voice.manifest.name == config.selected_voice)
    {
        Some(record) => {
            let dll = optional_path(&config.onnx_dll);
            VoiceModel::load(
                record.manifest.clone(),
                &record.directory,
                rate,
                InferDevice::parse(&config.inference, config.cuda_device),
                dll.as_deref(),
            )
        }
        None => VoiceModel::passthrough(rate),
    };
    let report = rustvc_infer::measure_duty(
        |input| model.process(input),
        rate,
        block,
        12,
        config.gpu_budget_percent,
    );
    format!("{} — {}", model.label(), report.summary())
}

pub fn run() -> eframe::Result<()> {
    let logo = Logo::load();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 720.0])
            .with_min_inner_size([860.0, 640.0])
            .with_title(brand::APP_NAME)
            .with_icon(logo.icon()),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        brand::APP_NAME,
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
        callback_frames: config.callback_frames.max(32),
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

fn voices_root_from(config: &AppConfig) -> PathBuf {
    let configured = PathBuf::from(&config.voices_dir);
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    resolve_voices_dir(&configured, &roots)
}

fn config_path() -> PathBuf {
    PathBuf::from("voice-changer.toml")
}

pub fn benchmark_voice(voice_name: &str, seconds: f32) -> Result<String, String> {
    let config = load_config();
    let voices = load_voices(Path::new(&config.voices_dir));
    let rate = config.sample_rate.max(1);
    let block = config.block_frames.max(1);
    let mut model = if voice_name == "Passthrough" {
        VoiceModel::passthrough(rate)
    } else {
        let record = voices
            .iter()
            .find(|voice| voice.manifest.name == voice_name)
            .ok_or_else(|| format!("no voice named {voice_name} in {}", config.voices_dir))?;
        let dll = optional_path(&config.onnx_dll);
        VoiceModel::load(
            record.manifest.clone(),
            &record.directory,
            rate,
            InferDevice::parse(&config.inference, config.cuda_device),
            dll.as_deref(),
        )
    };
    let blocks = ((seconds * rate as f32) / block as f32).round().max(1.0) as usize;
    let report = rustvc_infer::measure_duty(
        |input| model.process(input),
        rate,
        block,
        blocks,
        config.gpu_budget_percent,
    );
    Ok(format!("{}\n{}", model.label(), report.summary()))
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
