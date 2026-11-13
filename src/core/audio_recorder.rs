//! Real Microphone Audio Recording Engine (`src/core/audio_recorder.rs`):
//! - Uses `cpal` (Windows WASAPI default input device) to capture real-time microphone audio
//! - Converts `f32`, `i16`, or `u16` hardware streams into 16-bit PCM samples in `Arc<Mutex<Vec<i16>>>`
//! - Encodes and saves recordings to standard `.wav` files via `hound` alongside `.airteach` bundles

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use crossbeam_channel::{unbounded, Sender};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    Stopped,
    Recording,
    Paused,
}

enum AudioThreadCmd {
    Start,
    Pause,
    StopAndClear,
    Shutdown,
}

pub struct AudioRecorder {
    pcm_buffer: Arc<Mutex<Vec<i16>>>,
    waveform_history: Arc<Mutex<Vec<f32>>>,
    is_recording: Arc<AtomicBool>,
    is_paused: Arc<AtomicBool>,
    has_hardware_device: Arc<AtomicBool>,
    sample_rate: Arc<AtomicU32>,
    channels: Arc<AtomicU32>,
    rms_level_bits: Arc<AtomicU32>,
    device_name: Arc<Mutex<String>>,
    last_saved_wav: Option<PathBuf>,
    cmd_tx: Sender<AudioThreadCmd>,
}

impl AudioRecorder {
    pub fn new() -> Self {
        let pcm_buffer = Arc::new(Mutex::new(Vec::with_capacity(48_000 * 60)));
        let waveform_history = Arc::new(Mutex::new(vec![0.0_f32; 48]));
        let is_recording = Arc::new(AtomicBool::new(false));
        let is_paused = Arc::new(AtomicBool::new(false));
        let has_hardware_device = Arc::new(AtomicBool::new(false));
        let sample_rate = Arc::new(AtomicU32::new(44_100));
        let channels = Arc::new(AtomicU32::new(1));
        let rms_level_bits = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let device_name = Arc::new(Mutex::new("Detecting Microphone...".to_string()));

        let (cmd_tx, cmd_rx) = unbounded::<AudioThreadCmd>();

        let pcm_clone = Arc::clone(&pcm_buffer);
        let wave_clone = Arc::clone(&waveform_history);
        let rec_clone = Arc::clone(&is_recording);
        let pause_clone = Arc::clone(&is_paused);
        let hw_clone = Arc::clone(&has_hardware_device);
        let sr_clone = Arc::clone(&sample_rate);
        let ch_clone = Arc::clone(&channels);
        let rms_clone = Arc::clone(&rms_level_bits);
        let name_clone = Arc::clone(&device_name);

        thread::Builder::new()
            .name("airteach-cpal-audio-worker".to_string())
            .spawn(move || {
                let host = cpal::default_host();
                let device_opt = host.default_input_device();

                let mut active_stream: Option<cpal::Stream> = None;

                if let Some(ref device) = device_opt {
                    let dev_name = device
                        .name()
                        .unwrap_or_else(|_| "Default Audio Input".to_string());
                    *name_clone.lock() = dev_name;

                    if let Ok(supported_cfg) = device.default_input_config() {
                        hw_clone.store(true, Ordering::SeqCst);
                        let sample_format = supported_cfg.sample_format();
                        let config: StreamConfig = supported_cfg.into();
                        sr_clone.store(config.sample_rate.0, Ordering::SeqCst);
                        ch_clone.store(config.channels as u32, Ordering::SeqCst);

                        let err_fn = |err| {
                            tracing::warn!("cpal input stream error: {}", err);
                        };

                        let buf_for_cb = Arc::clone(&pcm_clone);
                        let wave_for_cb = Arc::clone(&wave_clone);
                        let rec_for_cb = Arc::clone(&rec_clone);
                        let pause_for_cb = Arc::clone(&pause_clone);
                        let rms_for_cb = Arc::clone(&rms_clone);

                        let stream_res = match sample_format {
                            SampleFormat::F32 => device.build_input_stream(
                                &config,
                                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                    Self::ingest_f32_samples(
                                        data,
                                        &buf_for_cb,
                                        &wave_for_cb,
                                        &rec_for_cb,
                                        &pause_for_cb,
                                        &rms_for_cb,
                                    );
                                },
                                err_fn,
                                None,
                            ),
                            SampleFormat::I16 => device.build_input_stream(
                                &config,
                                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                    Self::ingest_i16_samples(
                                        data,
                                        &buf_for_cb,
                                        &wave_for_cb,
                                        &rec_for_cb,
                                        &pause_for_cb,
                                        &rms_for_cb,
                                    );
                                },
                                err_fn,
                                None,
                            ),
                            SampleFormat::U16 => device.build_input_stream(
                                &config,
                                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                                    Self::ingest_u16_samples(
                                        data,
                                        &buf_for_cb,
                                        &wave_for_cb,
                                        &rec_for_cb,
                                        &pause_for_cb,
                                        &rms_for_cb,
                                    );
                                },
                                err_fn,
                                None,
                            ),
                            _ => Err(cpal::BuildStreamError::StreamConfigNotSupported),
                        };

                        if let Ok(stream) = stream_res {
                            let _ = stream.play();
                            active_stream = Some(stream);
                        }
                    } else {
                        *name_clone.lock() = "Microphone (Unsupported Config)".to_string();
                    }
                } else {
                    *name_clone.lock() = "No Hardware Microphone Detected".to_string();
                }

                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        AudioThreadCmd::Start => {
                            pause_clone.store(false, Ordering::SeqCst);
                            rec_clone.store(true, Ordering::SeqCst);
                            if let Some(ref s) = active_stream {
                                let _ = s.play();
                            }
                        }
                        AudioThreadCmd::Pause => {
                            pause_clone.store(true, Ordering::SeqCst);
                            rms_clone.store(0.0_f32.to_bits(), Ordering::Relaxed);
                        }
                        AudioThreadCmd::StopAndClear => {
                            rec_clone.store(false, Ordering::SeqCst);
                            pause_clone.store(false, Ordering::SeqCst);
                            rms_clone.store(0.0_f32.to_bits(), Ordering::Relaxed);
                            pcm_clone.lock().clear();
                        }
                        AudioThreadCmd::Shutdown => {
                            drop(active_stream);
                            break;
                        }
                    }
                }
            })
            .expect("Failed to spawn cpal audio worker thread");

        Self {
            pcm_buffer,
            waveform_history,
            is_recording,
            is_paused,
            has_hardware_device,
            sample_rate,
            channels,
            rms_level_bits,
            device_name,
            last_saved_wav: None,
            cmd_tx,
        }
    }

    #[inline]
    fn push_waveform_bar(wave_buf: &Arc<Mutex<Vec<f32>>>, rms: f32) {
        let mut w = wave_buf.lock();
        if w.len() >= 48 {
            w.remove(0);
        }
        w.push(rms);
    }

    #[inline]
    fn ingest_f32_samples(
        data: &[f32],
        pcm_buf: &Arc<Mutex<Vec<i16>>>,
        wave_buf: &Arc<Mutex<Vec<f32>>>,
        is_recording: &Arc<AtomicBool>,
        is_paused: &Arc<AtomicBool>,
        rms_bits: &Arc<AtomicU32>,
    ) {
        if data.is_empty() {
            return;
        }
        let actively_recording =
            is_recording.load(Ordering::Relaxed) && !is_paused.load(Ordering::Relaxed);
        let mut sum_sq = 0.0_f32;
        if actively_recording {
            let mut guard = pcm_buf.lock();
            for &sample in data {
                let clamped = sample.clamp(-1.0, 1.0);
                sum_sq += clamped * clamped;
                let pcm16 = (clamped * i16::MAX as f32) as i16;
                guard.push(pcm16);
            }
        } else {
            for &sample in data {
                let clamped = sample.clamp(-1.0, 1.0);
                sum_sq += clamped * clamped;
            }
        }
        let rms = (sum_sq / data.len() as f32).sqrt().clamp(0.0, 1.0);
        rms_bits.store(rms.to_bits(), Ordering::Relaxed);
        Self::push_waveform_bar(wave_buf, rms);
    }

    #[inline]
    fn ingest_i16_samples(
        data: &[i16],
        pcm_buf: &Arc<Mutex<Vec<i16>>>,
        wave_buf: &Arc<Mutex<Vec<f32>>>,
        is_recording: &Arc<AtomicBool>,
        is_paused: &Arc<AtomicBool>,
        rms_bits: &Arc<AtomicU32>,
    ) {
        if data.is_empty() {
            return;
        }
        if is_recording.load(Ordering::Relaxed) && !is_paused.load(Ordering::Relaxed) {
            let mut guard = pcm_buf.lock();
            guard.extend_from_slice(data);
        }
        let mut sum_sq = 0.0_f32;
        for &s in data {
            let norm = s as f32 / i16::MAX as f32;
            sum_sq += norm * norm;
        }
        let rms = (sum_sq / data.len() as f32).sqrt().clamp(0.0, 1.0);
        rms_bits.store(rms.to_bits(), Ordering::Relaxed);
        Self::push_waveform_bar(wave_buf, rms);
    }

    #[inline]
    fn ingest_u16_samples(
        data: &[u16],
        pcm_buf: &Arc<Mutex<Vec<i16>>>,
        wave_buf: &Arc<Mutex<Vec<f32>>>,
        is_recording: &Arc<AtomicBool>,
        is_paused: &Arc<AtomicBool>,
        rms_bits: &Arc<AtomicU32>,
    ) {
        if data.is_empty() {
            return;
        }
        let actively_recording =
            is_recording.load(Ordering::Relaxed) && !is_paused.load(Ordering::Relaxed);
        let mut sum_sq = 0.0_f32;
        if actively_recording {
            let mut guard = pcm_buf.lock();
            for &u in data {
                let s = (u as i32 - 32768).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                let norm = s as f32 / i16::MAX as f32;
                sum_sq += norm * norm;
                guard.push(s);
            }
        } else {
            for &u in data {
                let s = (u as i32 - 32768).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                let norm = s as f32 / i16::MAX as f32;
                sum_sq += norm * norm;
            }
        }
        let rms = (sum_sq / data.len() as f32).sqrt().clamp(0.0, 1.0);
        rms_bits.store(rms.to_bits(), Ordering::Relaxed);
        Self::push_waveform_bar(wave_buf, rms);
    }

    /// Start or resume microphone capture into the 16-bit PCM buffer.
    pub fn start(&self) {
        let _ = self.cmd_tx.send(AudioThreadCmd::Start);
    }

    /// Pause microphone capture without losing recorded PCM samples.
    pub fn pause(&self) {
        let _ = self.cmd_tx.send(AudioThreadCmd::Pause);
    }

    /// Stop recording and clear the PCM buffer.
    pub fn stop_and_clear(&self) {
        let _ = self.cmd_tx.send(AudioThreadCmd::StopAndClear);
    }

    pub fn state(&self) -> RecordingState {
        if !self.is_recording.load(Ordering::Relaxed) {
            RecordingState::Stopped
        } else if self.is_paused.load(Ordering::Relaxed) {
            RecordingState::Paused
        } else {
            RecordingState::Recording
        }
    }

    pub fn has_hardware_mic(&self) -> bool {
        self.has_hardware_device.load(Ordering::Relaxed)
    }

    pub fn device_name(&self) -> String {
        self.device_name.lock().clone()
    }

    pub fn rms_level(&self) -> f32 {
        f32::from_bits(self.rms_level_bits.load(Ordering::Relaxed))
    }

    /// Real-time microphone RMS level in decibels (dBFS, e.g., -60.0 dB to 0.0 dB).
    pub fn rms_db(&self) -> f32 {
        let rms = self.rms_level().max(1e-4);
        (20.0 * rms.log10()).clamp(-60.0, 0.0)
    }

    pub fn current_vu_db(&self) -> f32 {
        self.rms_db()
    }

    pub fn current_vu_normalized(&self) -> f32 {
        let db = self.rms_db();
        ((db + 60.0) / 60.0).clamp(0.0, 1.0)
    }

    /// Snapshot of the 48-bar live microphone waveform history.
    pub fn waveform_bars(&self) -> Vec<f32> {
        self.waveform_history.lock().clone()
    }

    pub fn sample_count(&self) -> usize {
        self.pcm_buffer.lock().len()
    }

    /// Encode the most recent `max_seconds` of recorded 16-bit PCM audio into an in-memory `.wav` buffer
    /// for real-time multimodal speech-to-text transcription via Gemini API.
    pub fn encode_recent_wav_bytes(&self, max_seconds: f32) -> Result<Vec<u8>, String> {
        let guard = self.pcm_buffer.lock();
        if guard.is_empty() {
            return Err("No audio samples recorded".to_string());
        }
        let sr = self.sample_rate.load(Ordering::Relaxed).max(8_000);
        let ch = self.channels.load(Ordering::Relaxed).clamp(1, 2) as u16;
        let max_samples = ((sr as f32 * ch as f32 * max_seconds) as usize).max(sr as usize);
        let slice = if guard.len() > max_samples {
            &guard[guard.len() - max_samples..]
        } else {
            &guard[..]
        };

        let spec = WavSpec {
            channels: ch,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::with_capacity(slice.len() * 2 + 64));
        {
            let mut writer =
                WavWriter::new(&mut cursor, spec).map_err(|e| format!("WAV mem writer error: {e}"))?;
            for &sample in slice {
                writer
                    .write_sample(sample)
                    .map_err(|e| format!("WAV mem sample error: {e}"))?;
            }
            writer
                .finalize()
                .map_err(|e| format!("WAV mem finalize error: {e}"))?;
        }
        Ok(cursor.into_inner())
    }

    pub fn recorded_duration_secs(&self) -> f32 {
        let len = self.pcm_buffer.lock().len();
        let sr = self.sample_rate.load(Ordering::Relaxed).max(1) as f32;
        let ch = self.channels.load(Ordering::Relaxed).max(1) as f32;
        len as f32 / (sr * ch)
    }

    pub fn last_saved_wav(&self) -> Option<&Path> {
        self.last_saved_wav.as_deref()
    }

    /// Write captured 16-bit PCM audio samples to a standard `.wav` file using `hound`.
    pub fn save_to_wav(&mut self, output_path: &Path) -> Result<PathBuf, String> {
        let samples = self.pcm_buffer.lock().clone();
        let sr = self.sample_rate.load(Ordering::Relaxed).max(8_000);
        let ch = self.channels.load(Ordering::Relaxed).clamp(1, 2) as u16;

        let spec = WavSpec {
            channels: ch,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };

        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        let mut writer =
            WavWriter::create(output_path, spec).map_err(|e| format!("WAV create error: {e}"))?;

        for &sample in &samples {
            writer
                .write_sample(sample)
                .map_err(|e| format!("WAV sample write error: {e}"))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("WAV finalize error: {e}"))?;

        let saved = output_path.to_path_buf();
        self.last_saved_wav = Some(saved.clone());
        Ok(saved)
    }

    pub fn shutdown(&self) {
        let _ = self.cmd_tx.send(AudioThreadCmd::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_writer_roundtrip() {
        let tmp_path = std::env::temp_dir().join("airteach_test_audio.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        {
            let mut writer = WavWriter::create(&tmp_path, spec).expect("Create WAV");
            for i in 0..4410 {
                let val = ((i as f32 * 0.1).sin() * 12000.0) as i16;
                writer.write_sample(val).expect("Write sample");
            }
            writer.finalize().expect("Finalize WAV");
        }
        assert!(tmp_path.exists());
        let _ = std::fs::remove_file(&tmp_path);
    }
}
