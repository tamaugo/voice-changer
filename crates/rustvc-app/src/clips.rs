//! Download speech clips from a `source.toml` manifest. Audio is not stored in git.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use serde::Deserialize;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::catalog::caption_is_speech;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const IMAGEINFO_BATCH: usize = 40;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SourceManifest {
    pub name: String,
    pub page: String,
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

/// Spaces and underscores are the same wiki title character. Case is ignored.
pub fn file_key(name: &str) -> String {
    name.trim()
        .strip_prefix("File:")
        .unwrap_or(name.trim())
        .split(|c: char| c.is_whitespace() || c == '_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn imageinfo_query(origin: &str, file_names: &[String]) -> String {
    let titles = file_names
        .iter()
        .map(|name| encode_query_component(&format!("File:{}", name.replace(' ', "_"))))
        .collect::<Vec<_>>()
        .join("%7C");
    format!(
        "{}/api.php?action=query&format=json&prop=imageinfo&iiprop=url&titles={titles}",
        origin.trim_end_matches('/')
    )
}

fn encode_query_component(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'.' => out.push(byte as char),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub fn urls_from_imageinfo(body: &str) -> Result<HashMap<String, String>, String> {
    let parsed: serde_json::Value = serde_json::from_str(body).map_err(|err| err.to_string())?;
    let pages = parsed
        .get("query")
        .and_then(|query| query.get("pages"))
        .and_then(|pages| pages.as_object())
        .ok_or("imageinfo response has no pages")?;
    let mut urls = HashMap::new();
    for page in pages.values() {
        if page.get("missing").is_some() {
            continue;
        }
        let Some(title) = page.get("title").and_then(|title| title.as_str()) else {
            continue;
        };
        let Some(url) = page
            .get("imageinfo")
            .and_then(|info| info.get(0))
            .and_then(|info| info.get("url"))
            .and_then(|url| url.as_str())
        else {
            continue;
        };
        if !url.is_empty() {
            urls.insert(file_key(title), url.to_string());
        }
    }
    Ok(urls)
}

pub fn clear_downloaded(directory: &Path) -> Result<usize, String> {
    let samples = directory.join("samples");
    if !samples.is_dir() {
        return Ok(0);
    }
    let mut removed = 0;
    for entry in fs::read_dir(&samples).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_file() {
            fs::remove_file(&path).map_err(|err| err.to_string())?;
            removed += 1;
        }
    }
    Ok(removed)
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
    cancel: Arc<AtomicBool>,
}

impl ClipHandle {
    pub fn poll(&self) -> Option<ClipEvent> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub fn start(directory: PathBuf) -> ClipHandle {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    thread::spawn(move || {
        let result = download_voice(&directory, &tx, &flag);
        let _ = tx.send(ClipEvent::Finished(result));
    });
    ClipHandle { rx, cancel }
}

fn download_voice(
    directory: &Path,
    tx: &Sender<ClipEvent>,
    cancel: &AtomicBool,
) -> Result<ClipSummary, String> {
    let manifest = SourceManifest::load(directory)?;
    send(tx, 0.02, "Reading the voice page");
    let api = wikitext_api(&manifest.page)?;
    let body = http_text(&api)?;
    let wikitext = wikitext_from_api(&body)?;
    let choices = choose_clips(&wikitext, &manifest);
    let kept: Vec<_> = choices.into_iter().filter(|clip| clip.keep).collect();
    let mut skipped = 0usize;
    send(tx, 0.08, "Resolving audio URLs");
    let origin = site_origin(&manifest.page)?;
    let names: Vec<String> = kept.iter().map(|clip| clip.file_name.clone()).collect();
    let urls = resolve_urls(&origin, &names, cancel)?;
    if cancel.load(Ordering::Relaxed) {
        let summary = ClipSummary {
            saved: 0,
            skipped: 0,
        };
        send(tx, 1.0, &summary_line(&summary));
        return Ok(summary);
    }
    let samples = directory.join("samples");
    fs::create_dir_all(&samples).map_err(|err| err.to_string())?;
    let mut saved = 0usize;
    let total = kept.len().max(1) as f32;
    for (index, clip) in kept.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let fraction = 0.1 + (index as f32 / total) * 0.85;
        send(tx, fraction, &format!("{} / {}", index + 1, kept.len()));
        let Some(url) = urls.get(&file_key(&clip.file_name)) else {
            skipped += 1;
            continue;
        };
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
        match fetch_to_file(url, &temp) {
            Ok(()) => {}
            Err(FetchError::Missing) => {
                let _ = fs::remove_file(&temp);
                skipped += 1;
                continue;
            }
            Err(FetchError::Other) => {
                let _ = fs::remove_file(&temp);
                skipped += 1;
                continue;
            }
        }
        match transcode(&temp, &dest_wav) {
            Ok(seconds) if too_short(seconds, manifest.min_seconds) => {
                let _ = fs::remove_file(&dest_wav);
                let _ = fs::remove_file(&temp);
                skipped += 1;
            }
            Ok(_) => {
                let _ = fs::remove_file(&temp);
                saved += 1;
            }
            Err(_) => {
                let _ = fs::remove_file(&temp);
                let _ = fs::remove_file(&dest_wav);
                skipped += 1;
            }
        }
    }
    let summary = ClipSummary { saved, skipped };
    let line = summary_line(&summary);
    send(tx, 1.0, &line);
    send_log(tx, &line);
    Ok(summary)
}

fn summary_line(summary: &ClipSummary) -> String {
    format!("{} downloaded, {} skipped", summary.saved, summary.skipped)
}

fn resolve_urls(
    origin: &str,
    names: &[String],
    cancel: &AtomicBool,
) -> Result<HashMap<String, String>, String> {
    let mut urls = HashMap::new();
    for chunk in names.chunks(IMAGEINFO_BATCH) {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let body = http_text(&imageinfo_query(origin, chunk))?;
        urls.extend(urls_from_imageinfo(&body)?);
    }
    Ok(urls)
}

fn site_origin(page: &str) -> Result<String, String> {
    let rest = page
        .trim()
        .strip_prefix("https://")
        .or_else(|| page.trim().strip_prefix("http://"))
        .ok_or("voice page URL must start with http:// or https://")?;
    let host = rest.split('/').next().unwrap_or("");
    if host.is_empty() {
        return Err("voice page URL has no host".into());
    }
    let scheme = if page.trim().starts_with("http://") {
        "http"
    } else {
        "https"
    };
    Ok(format!("{scheme}://{host}"))
}

enum FetchError {
    Missing,
    Other,
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<(), FetchError> {
    let response = agent()
        .get(url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "*/*")
        .call();
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(404 | 410, _)) => return Err(FetchError::Missing),
        Err(_) => return Err(FetchError::Other),
    };
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|_| FetchError::Other)?;
    }
    let mut file = fs::File::create(dest).map_err(|_| FetchError::Other)?;
    std::io::copy(&mut response.into_reader(), &mut file).map_err(|_| FetchError::Other)?;
    Ok(())
}

fn http_text(url: &str) -> Result<String, String> {
    let response = agent()
        .get(url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "application/json,text/plain,*/*")
        .call()
        .map_err(|err| err.to_string())?;
    response.into_string().map_err(|err| err.to_string())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().redirects(8).build()
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
            min_seconds: 0.3,
            speaker: "Rem".into(),
            skip_sections: vec![
                "Lil Helpers".into(),
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
        assert!(by_name("Familiar ping use ability3.mp3")
            .reason
            .contains("Lil Helpers"));
        assert!(kept
            .iter()
            .all(|name| !name.to_ascii_lowercase().contains("ability3")));
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
        let helpers = choices
            .iter()
            .filter(|clip| clip.keep && clip.section.to_ascii_lowercase().contains("helper"));
        assert_eq!(helpers.count(), 0);
    }

    #[test]
    fn imageinfo_uses_the_real_url_and_skips_missing_files() {
        let urls = urls_from_imageinfo(include_str!("../fixtures/imageinfo.json")).unwrap();
        assert_eq!(
            urls.get(&file_key("familiar select 01.mp3"))
                .map(String::as_str),
            Some("https://cdn.deadlock.wiki/Familiar_select_01.mp3")
        );
        assert_eq!(
            urls.get(&file_key(
                "familiar match start familiar atlas convo01 01.mp3"
            ))
            .map(String::as_str),
            Some("https://cdn.deadlock.wiki/Familiar_match_start_familiar_atlas_convo01_01.mp3")
        );
        assert!(!urls.contains_key(&file_key("Missing clip.mp3")));
        let query = imageinfo_query(
            "https://deadlock.wiki",
            &[
                "Familiar select 01.mp3".into(),
                "familiar match start familiar atlas convo01 01.mp3".into(),
            ],
        );
        assert!(query.contains("action=query"));
        assert!(query.contains("prop=imageinfo"));
        assert!(query.contains("iiprop=url"));
        assert!(query.contains("File%3AFamiliar_select_01.mp3"));
        assert!(query.contains("%7C"));
        assert!(!query.contains("cdn.deadlock.wiki"));
    }

    #[test]
    fn print_imageinfo_queries_when_asked() {
        if std::env::var("LARK_QUERIES").is_err() {
            return;
        }
        let wikitext = fs::read_to_string("/tmp/rem-voicelines.wikitext").unwrap();
        let names: Vec<String> = choose_clips(&wikitext, &rem())
            .into_iter()
            .filter(|clip| clip.keep)
            .map(|clip| clip.file_name)
            .collect();
        println!("kept {}", names.len());
        for (index, chunk) in names.chunks(20).enumerate() {
            println!(
                "BATCH {index} {}",
                imageinfo_query("https://deadlock.wiki", chunk)
            );
        }
    }

    #[test]
    fn download_resolved_rem_clips_when_asked() {
        if std::env::var("LARK_FETCH").is_err() {
            return;
        }
        let wikitext = fs::read_to_string("/tmp/rem-voicelines.wikitext").unwrap();
        let kept: Vec<_> = choose_clips(&wikitext, &rem())
            .into_iter()
            .filter(|clip| clip.keep)
            .collect();
        let mut urls = HashMap::new();
        for entry in fs::read_dir("/tmp/lark-ii").unwrap() {
            let path = entry.unwrap().path();
            let body = fs::read_to_string(&path).unwrap();
            urls.extend(urls_from_imageinfo(&body).unwrap());
        }
        let directory = PathBuf::from("/tmp/lark-rem-voice");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(directory.join("samples")).unwrap();
        let mut saved = 0usize;
        let mut skipped = 0usize;
        let mut missing_reason = std::collections::BTreeMap::<String, usize>::new();
        for clip in &kept {
            assert!(!clip.section.to_ascii_lowercase().contains("helper"));
            assert!(!clip.file_name.to_ascii_lowercase().contains("ability3"));
            assert!(!clip.file_name.to_ascii_lowercase().contains("power3"));
            let Some(url) = urls.get(&file_key(&clip.file_name)) else {
                *missing_reason.entry("no imageinfo url".into()).or_default() += 1;
                skipped += 1;
                continue;
            };
            let dest = directory.join("samples").join(wav_name(&clip.file_name));
            let temp = std::env::temp_dir()
                .join(format!("lark-live-{}", clip.file_name.replace(' ', "_")));
            match fetch_to_file(url, &temp) {
                Err(FetchError::Missing) => {
                    *missing_reason.entry("404".into()).or_default() += 1;
                    skipped += 1;
                    continue;
                }
                Err(FetchError::Other) => {
                    *missing_reason.entry("other".into()).or_default() += 1;
                    skipped += 1;
                    continue;
                }
                Ok(()) => {}
            }
            match transcode(&temp, &dest) {
                Ok(seconds) if too_short(seconds, 0.3) => {
                    let _ = fs::remove_file(&dest);
                    *missing_reason.entry("under 0.3s".into()).or_default() += 1;
                    skipped += 1;
                }
                Ok(_) => saved += 1,
                Err(_) => {
                    let _ = fs::remove_file(&dest);
                    *missing_reason.entry("decode".into()).or_default() += 1;
                    skipped += 1;
                }
            }
            let _ = fs::remove_file(&temp);
        }
        println!("{saved} downloaded, {skipped} skipped");
        for (reason, count) in &missing_reason {
            println!("  {count} {reason}");
        }
        let mut names: Vec<_> = fs::read_dir(directory.join("samples"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names.iter().take(8) {
            println!("  {name}");
        }
        assert_eq!(saved + skipped, kept.len());
        assert!(saved > 0);
        assert!(names.iter().all(|name| {
            let lower = name.to_ascii_lowercase();
            !lower.contains("ability3") && !lower.contains("power3") && !lower.contains("helper")
        }));
    }

    #[test]
    fn live_rem_download_when_asked() {
        if std::env::var("LARK_LIVE").is_err() {
            return;
        }
        let wikitext = fs::read_to_string("/tmp/rem-voicelines.wikitext").unwrap();
        let choices = choose_clips(&wikitext, &rem());
        let kept: Vec<_> = choices.into_iter().filter(|clip| clip.keep).collect();
        assert!(kept.iter().all(|clip| {
            !clip.section.to_ascii_lowercase().contains("helper")
                && !clip.file_name.to_ascii_lowercase().contains("ability3")
                && !clip.file_name.to_ascii_lowercase().contains("power3")
        }));
        let out = PathBuf::from("/tmp/lark-rem-samples");
        let _ = fs::remove_dir_all(&out);
        fs::create_dir_all(out.join("samples")).unwrap();
        let manifest_dir = PathBuf::from("/tmp/lark-rem-voice");
        let _ = fs::remove_dir_all(&manifest_dir);
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::copy(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../voices/Rem/source.toml"),
            manifest_dir.join("source.toml"),
        )
        .unwrap();
        let (tx, rx) = mpsc::channel();
        let summary = download_voice(&manifest_dir, &tx, &AtomicBool::new(false)).unwrap();
        println!(
            "live {} downloaded, {} skipped",
            summary.saved, summary.skipped
        );
        let mut names: Vec<_> = fs::read_dir(manifest_dir.join("samples"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        println!("files {}", names.len());
        for name in names.iter().take(8) {
            println!("  {name}");
        }
        assert!(names.iter().all(|name| {
            let lower = name.to_ascii_lowercase();
            !lower.contains("ability3") && !lower.contains("power3") && !lower.contains("helper")
        }));
        assert!(summary.saved > 0, "no clips downloaded");
        let _ = rx;
        let _ = out;
    }
}
