//! Download speech clips from a `source.toml` manifest. Audio is not stored in git.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use serde::Deserialize;

use crate::catalog::caption_is_speech;
use crate::setup::download_file;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SourceManifest {
    pub name: String,
    pub page: String,
    pub file_base: String,
    #[serde(default = "default_min_seconds")]
    pub min_seconds: f32,
    #[serde(default)]
    pub speaker: String,
    #[serde(default)]
    pub skip_sections: Vec<String>,
}

fn default_min_seconds() -> f32 {
    0.3
}

impl SourceManifest {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let path = directory.join("source.toml");
        let text = fs::read_to_string(&path).map_err(|err| err.to_string())?;
        toml::from_str(&text).map_err(|err| err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipChoice {
    pub file_name: String,
    pub url: String,
    pub section: String,
    pub caption: String,
    pub keep: bool,
    pub reason: String,
}

pub fn wikitext_api(page: &str) -> Result<String, String> {
    let page = page.trim();
    let rest = page
        .strip_prefix("https://")
        .or_else(|| page.strip_prefix("http://"))
        .ok_or("voice page URL must start with http:// or https://")?;
    let (host, path) = rest
        .split_once('/')
        .ok_or("voice page URL has no article path")?;
    let mut title = path.trim_start_matches('/');
    if let Some(stripped) = title.strip_prefix("wiki/") {
        title = stripped;
    }
    if title.is_empty() {
        return Err("voice page URL has no article title".into());
    }
    let encoded = encode_title(title);
    Ok(format!(
        "https://{host}/api.php?action=parse&page={encoded}&prop=wikitext&format=json"
    ))
}

fn encode_title(title: &str) -> String {
    let mut out = String::new();
    for byte in title.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'_' | b'-' | b'.' => {
                out.push(byte as char);
            }
            b' ' => out.push('_'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub fn wikitext_from_api(body: &str) -> Result<String, String> {
    let parsed: ParseResponse = serde_json::from_str(body).map_err(|err| err.to_string())?;
    Ok(parsed.parse.wikitext.body)
}

pub fn clip_url(file_base: &str, file_name: &str) -> String {
    let name = file_name.trim().replace(' ', "_");
    format!("{}/{}", file_base.trim_end_matches('/'), name)
}

pub fn wav_name(file_name: &str) -> String {
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    format!("{stem}.wav")
}

/// Parse `{{Audio link|file|caption}}` rows and apply the manifest filters.
pub fn choose_clips(wikitext: &str, manifest: &SourceManifest) -> Vec<ClipChoice> {
    let mut choices = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut h2 = String::new();
    let mut h3 = String::new();
    let mut index = 0;
    while index < wikitext.len() {
        if index == 0 || wikitext.as_bytes()[index] == b'\n' {
            let line_start = if index == 0 { 0 } else { index + 1 };
            if line_start < wikitext.len() {
                let line_end = wikitext[line_start..]
                    .find('\n')
                    .map(|offset| line_start + offset)
                    .unwrap_or(wikitext.len());
                if let Some((level, title)) = parse_heading(&wikitext[line_start..line_end]) {
                    if level <= 2 {
                        h2 = title;
                        h3.clear();
                    } else {
                        h3 = title;
                    }
                }
            }
        }
        if wikitext[index..].starts_with("{{Audio link|") {
            if let Some((next, file_name, caption)) = parse_audio_link(wikitext, index) {
                if seen.insert(file_name.clone()) {
                    let section = if h3.is_empty() {
                        h2.clone()
                    } else {
                        format!("{h2} / {h3}")
                    };
                    let speaker = last_hero_icon(&wikitext[index.saturating_sub(500)..index]);
                    let reason = skip_reason(manifest, &section, &caption, speaker.as_deref());
                    choices.push(ClipChoice {
                        url: clip_url(&manifest.file_base, &file_name),
                        keep: reason.is_empty(),
                        reason,
                        file_name,
                        section,
                        caption,
                    });
                }
                index = next;
                continue;
            }
        }
        index += 1;
    }
    choices
}

fn skip_reason(
    manifest: &SourceManifest,
    section: &str,
    caption: &str,
    speaker: Option<&str>,
) -> String {
    let section_lower = section.to_ascii_lowercase();
    for skip in &manifest.skip_sections {
        if section_lower.contains(&skip.to_ascii_lowercase()) {
            return format!("section {skip}");
        }
    }
    if !manifest.speaker.is_empty() {
        if let Some(name) = speaker {
            if !name.eq_ignore_ascii_case(&manifest.speaker) {
                return format!("other speaker {name}");
            }
        }
    }
    if !caption_is_speech(caption) {
        return "no speech".to_string();
    }
    String::new()
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim();
    let level = trimmed.chars().take_while(|c| *c == '=').count();
    if level < 2 || !trimmed.ends_with('=') {
        return None;
    }
    let title = trimmed.trim_matches('=').trim();
    if title.is_empty() {
        return None;
    }
    Some((level, plain_heading(title)))
}

fn plain_heading(title: &str) -> String {
    let Some(start) = title.find("{{") else {
        return title.to_string();
    };
    let Some(end) = title[start..].find("}}") else {
        return title.to_string();
    };
    let inner = &title[start + 2..start + end];
    inner.split('|').nth(1).unwrap_or(inner).trim().to_string()
}

fn parse_audio_link(text: &str, start: usize) -> Option<(usize, String, String)> {
    const MARK: &str = "{{Audio link|";
    if !text[start..].starts_with(MARK) {
        return None;
    }
    let body_start = start + MARK.len();
    let end = text[body_start..].find("}}")? + body_start;
    let body = &text[body_start..end];
    let (file_name, caption) = body.split_once('|')?;
    let file_name = file_name.trim();
    if file_name.is_empty() {
        return None;
    }
    Some((end + 2, file_name.to_string(), caption.trim().to_string()))
}

fn last_hero_icon(before: &str) -> Option<String> {
    let mut last = None;
    let mut rest = before;
    while let Some(pos) = rest.find("{{HeroIcon|") {
        let after = &rest[pos + "{{HeroIcon|".len()..];
        let name = after.split(['|', '}']).next().unwrap_or("").trim();
        if !name.is_empty() {
            last = Some(name.to_string());
        }
        rest = &rest[pos + "{{HeroIcon|".len()..];
    }
    last
}

pub fn too_short(seconds: f32, min_seconds: f32) -> bool {
    min_seconds > 0.0 && seconds < min_seconds
}

pub struct ClipSummary {
    pub saved: usize,
    pub skipped: usize,
}

pub enum ClipEvent {
    Progress { fraction: f32, message: String },
    Log(String),
    Finished(Result<ClipSummary, String>),
}

pub struct ClipHandle {
    rx: Receiver<ClipEvent>,
}

impl ClipHandle {
    pub fn poll(&self) -> Option<ClipEvent> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }
}

pub fn start(directory: PathBuf) -> ClipHandle {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = download_voice(&directory, &tx);
        let _ = tx.send(ClipEvent::Finished(result));
    });
    ClipHandle { rx }
}

fn download_voice(directory: &Path, tx: &Sender<ClipEvent>) -> Result<ClipSummary, String> {
    let manifest = SourceManifest::load(directory)?;
    send(tx, 0.02, "Reading the voice page");
    let api = wikitext_api(&manifest.page)?;
    let body = http_text(&api)?;
    let wikitext = wikitext_from_api(&body)?;
    let choices = choose_clips(&wikitext, &manifest);
    let kept: Vec<_> = choices.iter().filter(|clip| clip.keep).cloned().collect();
    let mut skipped = choices.len() - kept.len();
    send_log(
        tx,
        &format!(
            "{} clips on the page, {} to download",
            choices.len(),
            kept.len()
        ),
    );
    let samples = directory.join("samples");
    fs::create_dir_all(&samples).map_err(|err| err.to_string())?;
    let mut saved = 0;
    let total = kept.len().max(1) as f32;
    for (index, clip) in kept.iter().enumerate() {
        let fraction = 0.05 + (index as f32 / total) * 0.9;
        send(tx, fraction, &clip.file_name);
        let dest_wav = samples.join(wav_name(&clip.file_name));
        if dest_wav.is_file() {
            saved += 1;
            continue;
        }
        let temp = std::env::temp_dir().join(format!(
            "lark-{}-{}",
            std::process::id(),
            clip.file_name.replace(' ', "_")
        ));
        if let Err(err) = download_file(&clip.url, &temp, |_, _| {}) {
            let _ = fs::remove_file(&temp);
            send_log(tx, &format!("skip {}: {err}", clip.file_name));
            skipped += 1;
            continue;
        }
        match transcode(&temp, &dest_wav) {
            Ok(seconds) if too_short(seconds, manifest.min_seconds) => {
                let _ = fs::remove_file(&dest_wav);
                let _ = fs::remove_file(&temp);
                send_log(
                    tx,
                    &format!(
                        "skip {} ({seconds:.2}s, under {:.1}s)",
                        clip.file_name, manifest.min_seconds
                    ),
                );
                skipped += 1;
            }
            Ok(_) => {
                let _ = fs::remove_file(&temp);
                saved += 1;
            }
            Err(err) => {
                let _ = fs::remove_file(&temp);
                let _ = fs::remove_file(&dest_wav);
                send_log(tx, &format!("skip {}: {err}", clip.file_name));
                skipped += 1;
            }
        }
    }
    send(tx, 1.0, "Finished");
    send_log(tx, &format!("{saved} clips saved, {skipped} skipped"));
    if saved == 0 {
        return Err("No speech clips were saved.".into());
    }
    Ok(ClipSummary { saved, skipped })
}

fn http_text(url: &str) -> Result<String, String> {
    let response = ureq::get(url)
        .set(
            "User-Agent",
            "Mozilla/5.0 (compatible; Lark/0.2; +https://github.com/tamaugo/voice-changer)",
        )
        .set("Accept", "application/json,text/plain,*/*")
        .call()
        .map_err(|err| err.to_string())?;
    response.into_string().map_err(|err| err.to_string())
}

fn transcode(source: &Path, wav_path: &Path) -> Result<f32, String> {
    let ext = source
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "wav" {
        fs::copy(source, wav_path).map_err(|err| err.to_string())?;
        return Ok(1.0);
    }
    decode_to_wav(source, wav_path)
}

fn decode_to_wav(source: &Path, wav_path: &Path) -> Result<f32, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = fs::File::open(source).map_err(|err| err.to_string())?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = source.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| err.to_string())?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no audio track")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let channels = track
        .codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1)
        .max(1);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| err.to_string())?;
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(wav_path, spec).map_err(|err| err.to_string())?;
    let mut frames = 0u64;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(_) => continue,
        };
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        for sample in buffer.samples() {
            let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(scaled).map_err(|err| err.to_string())?;
        }
        frames += buffer.samples().len() as u64 / channels as u64;
    }
    writer.finalize().map_err(|err| err.to_string())?;
    Ok(frames as f32 / rate as f32)
}

fn send(tx: &Sender<ClipEvent>, fraction: f32, message: &str) {
    let _ = tx.send(ClipEvent::Progress {
        fraction,
        message: message.to_string(),
    });
}

fn send_log(tx: &Sender<ClipEvent>, line: &str) {
    let _ = tx.send(ClipEvent::Log(line.to_string()));
}

#[derive(Deserialize)]
struct ParseResponse {
    parse: ParseBody,
}

#[derive(Deserialize)]
struct ParseBody {
    wikitext: WikiText,
}

#[derive(Deserialize)]
struct WikiText {
    #[serde(rename = "*")]
    body: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rem() -> SourceManifest {
        SourceManifest {
            name: "Rem".into(),
            page: "https://deadlock.wiki/Rem/Voice_lines".into(),
            file_base: "https://cdn.deadlock.wiki".into(),
            min_seconds: 0.3,
            speaker: "Rem".into(),
            skip_sections: vec![
                "Efforts".into(),
                "Pain".into(),
                "Other Characters".into(),
                "Enemy Heroes".into(),
                "Patrons".into(),
                "Shopkeeper".into(),
                "Soul Urn".into(),
                "Removed".into(),
            ],
        }
    }

    #[test]
    fn api_fixture_keeps_rem_speech_and_drops_the_rest() {
        let body = include_str!("../fixtures/voice_page.json");
        let wikitext = wikitext_from_api(body).unwrap();
        let choices = choose_clips(&wikitext, &rem());
        let kept: Vec<_> = choices
            .iter()
            .filter(|clip| clip.keep)
            .map(|clip| clip.file_name.as_str())
            .collect();
        assert_eq!(
            kept,
            vec![
                "Familiar select 01.mp3",
                "Familiar unselect 02.mp3",
                "familiar convo 01.mp3",
            ]
        );
        let by_name = |name: &str| choices.iter().find(|clip| clip.file_name == name).unwrap();
        assert_eq!(by_name("Familiar select 05 alt 01.mp3").reason, "no speech");
        assert_eq!(by_name("Familiar unselect 01.mp3").reason, "no speech");
        assert_eq!(by_name("abrams-rem-01-full.mp3").reason, "no speech");
        assert!(by_name("atlas convo 02.mp3")
            .reason
            .contains("other speaker"));
        assert!(by_name("Familiar effort general 01.mp3")
            .reason
            .contains("Efforts"));
        assert!(by_name("Familiar pain small 01.mp3")
            .reason
            .contains("Pain"));
        assert!(
            by_name("apollo kill 01.mp3")
                .reason
                .contains("Enemy Heroes")
                || by_name("apollo kill 01.mp3")
                    .reason
                    .contains("Other Characters")
        );
        assert!(by_name("Familiar select old.mp3")
            .reason
            .contains("Removed"));
        assert_eq!(
            by_name("Familiar select 01.mp3").url,
            "https://cdn.deadlock.wiki/Familiar_select_01.mp3"
        );
        assert_eq!(wav_name("Familiar select 01.mp3"), "Familiar select 01.wav");
        assert!(too_short(0.2, 0.3));
        assert!(!too_short(2.7, 0.3));
    }

    #[test]
    fn page_url_becomes_the_mediawiki_parse_api() {
        let api = wikitext_api("https://deadlock.wiki/Rem/Voice_lines").unwrap();
        assert_eq!(
            api,
            "https://deadlock.wiki/api.php?action=parse&page=Rem/Voice_lines&prop=wikitext&format=json"
        );
    }

    #[test]
    fn real_page_count_when_the_file_is_present() {
        let Ok(path) = std::env::var("REM_WIKITEXT") else {
            return;
        };
        let wikitext = fs::read_to_string(path).unwrap();
        let choices = choose_clips(&wikitext, &rem());
        let kept = choices.iter().filter(|clip| clip.keep).count();
        let mut reasons = std::collections::BTreeMap::<String, usize>::new();
        for clip in &choices {
            if !clip.keep {
                *reasons.entry(clip.reason.clone()).or_default() += 1;
            }
        }
        println!("page clips {} kept {kept}", choices.len());
        for (reason, count) in &reasons {
            println!("  {count} {reason}");
        }
        assert!(kept > 100, "expected speech clips, got {kept}");
    }
}
