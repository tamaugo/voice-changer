//! Decide which Rem wiki clips are speech.
#![cfg_attr(not(test), allow(dead_code))]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipDecision {
    Keep,
    Skip(&'static str),
}

impl ClipDecision {
    pub fn is_skip(self) -> bool {
        matches!(self, Self::Skip(_))
    }
}

/// `section` is the wiki heading. `name` is the row label or transcript.
/// `duration_secs` is set only when the file itself was measured.
pub fn classify_clip(section: &str, name: &str, duration_secs: Option<f32>) -> ClipDecision {
    let section = section.to_ascii_lowercase();
    if section.contains("removed") {
        return ClipDecision::Skip("removed");
    }
    if [
        "other characters",
        "enemy heroes",
        "patrons",
        "shopkeeper",
        "soul urn",
    ]
    .iter()
    .any(|marker| section.contains(marker))
    {
        return ClipDecision::Skip("not Rem");
    }
    if ["weapon", "reload", "movement"]
        .iter()
        .any(|marker| section.contains(marker))
    {
        return ClipDecision::Skip("weapon, movement, or ability effect");
    }
    if ["effort", "pain"]
        .iter()
        .any(|marker| section.contains(marker))
    {
        return ClipDecision::Skip("effort or pain, not speech");
    }
    if !is_speech(name) {
        return ClipDecision::Skip("no speech");
    }
    if duration_secs.is_some_and(|seconds| seconds < 0.3) {
        return ClipDecision::Skip("shorter than 0.3 seconds");
    }
    ClipDecision::Keep
}

fn is_speech(name: &str) -> bool {
    let spoken = spoken_text(name);
    let spoken = spoken.trim();
    !spoken.is_empty()
        && spoken.chars().any(|c| c.is_ascii_lowercase())
        && spoken.contains(['.', '!', '?'])
}

/// Drop bracketed stage directions and parenthetical alts.
fn spoken_text(name: &str) -> String {
    let mut out = String::new();
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '(' || c == '[' {
            let end = if c == '(' { ')' } else { ']' };
            for next in chars.by_ref() {
                if next == end {
                    break;
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_lines_are_kept() {
        assert_eq!(
            classify_clip("Select", "Being awake is hard!", None),
            ClipDecision::Keep
        );
        assert_eq!(
            classify_clip(
                "Select",
                "My friends and I will help! (Alt: [Chirping])",
                None
            ),
            ClipDecision::Keep
        );
        assert_eq!(
            classify_clip("Abilities", "Pillow's ready!", None),
            ClipDecision::Keep
        );
    }

    #[test]
    fn stage_directions_efforts_and_other_speakers_are_skipped() {
        assert_eq!(
            classify_clip("Unselect", "[Yawn]", None),
            ClipDecision::Skip("no speech")
        );
        assert_eq!(
            classify_clip("Unselect", "[Snores] (Alt lines: [Various Snores])", None),
            ClipDecision::Skip("no speech")
        );
        assert_eq!(
            classify_clip("Efforts", "Effort 01", None),
            ClipDecision::Skip("effort or pain, not speech")
        );
        assert_eq!(
            classify_clip("Pain", "Small Pain 01", None),
            ClipDecision::Skip("effort or pain, not speech")
        );
        assert_eq!(
            classify_clip("Enemy Heroes", "Dealt with Rem!", None),
            ClipDecision::Skip("not Rem")
        );
        assert_eq!(
            classify_clip("Removed/Unused Voice Lines", "I have a pillow.", None),
            ClipDecision::Skip("removed")
        );
        assert_eq!(
            classify_clip("Select", "Hi!", Some(0.2)),
            ClipDecision::Skip("shorter than 0.3 seconds")
        );
    }

    #[test]
    fn sounds_page_is_effects_only() {
        let mut names = Vec::new();
        for index in 1..=6 {
            names.push(("Weapon", format!("Weapon Fire {index:02}")));
        }
        names.push(("Weapon", "Last shot".into()));
        for stage in ["Start", "Mid", "End"] {
            for index in 1..=3 {
                names.push(("Weapon", format!("{stage} {index:02}")));
            }
        }
        for index in 1..=3 {
            names.push(("Abilities", format!("Cast {index:02}")));
            names.push(("Abilities", format!("Impact {index:02}")));
            names.push(("Abilities", format!("Hit {index:02}")));
        }
        names.push(("Abilities", "Projectile Loop".into()));
        names.push(("Abilities", "Cast Delay".into()));
        for index in 1..=3 {
            names.push(("Abilities", format!("Cast {index:02}")));
        }
        names.push(("Abilities", "Travel Loop".into()));
        for index in 1..=3 {
            names.push(("Abilities", format!("Attach Host {index:02}")));
            names.push(("Abilities", format!("Detach Host {index:02}")));
            names.push(("Abilities", format!("Timeout {index:02}")));
        }
        names.push(("Abilities", "Attach Rampup".into()));
        for index in 1..=3 {
            names.push(("Abilities", format!("Select {index:02}")));
            names.push(("Abilities", format!("Deselect {index:02}")));
        }
        names.push(("Abilities", "Select Loop".into()));
        for index in 1..=16 {
            names.push(("Abilities", format!("Helper Emote {index:02}")));
        }
        for index in 1..=3 {
            names.push(("Abilities", format!("Cast {index:02}")));
        }
        for index in 1..=9 {
            names.push(("Abilities", format!("Emote Attack {index:02}")));
        }
        for index in 1..=3 {
            names.push(("Abilities", format!("Heal {index:02}")));
        }
        names.push(("Abilities", "Buff Loop".into()));
        for index in 1..=6 {
            names.push(("Abilities", format!("Helper Shoot {index:02}")));
        }
        names.push(("Abilities", "Cast Delay".into()));
        names.push(("Abilities", "Channel".into()));
        names.push(("Abilities", "Slow Loop".into()));
        names.push(("Abilities", "Cast".into()));
        names.push(("Abilities", "AoE Explode".into()));
        for index in 1..=4 {
            names.push(("Abilities", format!("Wakeup {index:02}")));
        }
        assert_eq!(names.len(), 95);
        assert!(names
            .iter()
            .all(|(section, name)| { classify_clip(section, name, None).is_skip() }));
    }
}
