//! Real Speech-to-Text Live Transcript & Subtitles Pipeline (`src/ai/subtitles_engine.rs`):
//! - Zero hardcoded fake sentences
//! - Triggers real microphone WAV chunk transcription via Google Gemini Multimodal API (`audio/wav`)
//! - Stores time-coded `SubtitleCue` entries and exports `.srt` alongside `recordings/lecture_YYYYMMDD_HHMMSS.wav`

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleCue {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptSegmentEvent {
    pub timestamp_ms: u64,
    pub text: String,
}

pub struct LiveSubtitlesEngine {
    pub enabled: bool,
    pub cues: Vec<SubtitleCue>,
    pub current_banner_text: String,
    pub stt_in_flight: bool,
    last_stt_request_ms: u64,
    speech_energy_frames: u32,
}

impl LiveSubtitlesEngine {
    pub fn new() -> Self {
        Self {
            enabled: true,
            cues: Vec::new(),
            current_banner_text:
                "[STT] Phụ đề trực tiếp sẵn sàng — Nhấn '[Ghi Bài Giảng]' và nói vào Micro để nhận diện giọng nói thật.".to_string(),
            stt_in_flight: false,
            last_stt_request_ms: 0,
            speech_energy_frames: 0,
        }
    }

    pub fn clear(&mut self) {
        self.cues.clear();
        self.last_stt_request_ms = 0;
        self.speech_energy_frames = 0;
        self.stt_in_flight = false;
        self.current_banner_text = "[STT] Đã xóa Live Transcript.".to_string();
    }

    /// Push a real transcribed segment or real board event cue with timestamp.
    pub fn push_cue(&mut self, timestamp_ms: u64, duration_ms: u64, text: String) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        let idx = self.cues.len() + 1;
        self.current_banner_text = trimmed.to_string();
        self.cues.push(SubtitleCue {
            index: idx,
            start_ms: timestamp_ms,
            end_ms: timestamp_ms + duration_ms.max(2000),
            text: trimmed.to_string(),
        });
    }

    /// Check whether real microphone speech activity should trigger a background STT request.
    pub fn should_trigger_live_stt(
        &mut self,
        is_recording: bool,
        elapsed_ms: u64,
        mic_rms: f32,
    ) -> bool {
        if !self.enabled || !is_recording || self.stt_in_flight {
            return false;
        }
        if mic_rms > 0.012 {
            self.speech_energy_frames = self.speech_energy_frames.saturating_add(1);
        }
        // Trigger STT after ~0.5s of real voice energy and at least 5.5s since last request
        if self.speech_energy_frames >= 12
            && elapsed_ms.saturating_sub(self.last_stt_request_ms) >= 5500
        {
            self.speech_energy_frames = 0;
            self.last_stt_request_ms = elapsed_ms;
            self.stt_in_flight = true;
            return true;
        }
        false
    }

    /// Return the full concatenated transcript text for display in the Live Transcript panel
    /// and for multimodal attachment when clicking "Analyze Board & Lecture".
    pub fn full_transcript_text(&self) -> String {
        if self.cues.is_empty() {
            return String::new();
        }
        self.cues
            .iter()
            .map(|c| {
                let secs = c.start_ms / 1000;
                format!("[{:02}:{:02}] {}", secs / 60, secs % 60, c.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Retrieve the subtitle text at a specific playback timestamp during Lecture Replay.
    pub fn subtitle_at_timestamp(&self, timestamp_ms: u64) -> Option<&str> {
        self.cues
            .iter()
            .rev()
            .find(|c| timestamp_ms >= c.start_ms && timestamp_ms <= c.end_ms + 1500)
            .map(|c| c.text.as_str())
    }

    /// Export all recorded subtitle cues to a standard `.srt` file alongside the `.wav` and `.airteach` bundle.
    pub fn export_srt(&self, path: &Path) -> Result<PathBuf, String> {
        let mut out = String::new();
        for cue in &self.cues {
            out.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                cue.index,
                Self::format_srt_time(cue.start_ms),
                Self::format_srt_time(cue.end_ms),
                cue.text
            ));
        }
        std::fs::write(path, out).map_err(|e| format!("Failed to write SRT: {e}"))?;
        Ok(path.to_path_buf())
    }

    fn format_srt_time(ms: u64) -> String {
        let hours = ms / 3_600_000;
        let mins = (ms % 3_600_000) / 60_000;
        let secs = (ms % 60_000) / 1_000;
        let millis = ms % 1_000;
        format!("{hours:02}:{mins:02}:{secs:02},{millis:03}")
    }
}
