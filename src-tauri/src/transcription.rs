use std::collections::BTreeSet;
use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::{NormalizedTranscript, TranscriptSegment, TranscriptWord};

pub async fn transcribe_deepgram(audio_path: &str, api_key: &str) -> Result<NormalizedTranscript> {
    let bytes = tokio::fs::read(audio_path)
        .await
        .with_context(|| format!("reading audio file {audio_path}"))?;

    let response = reqwest::Client::new()
        .post("https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&diarize=true&punctuate=true&filler_words=true")
        .header("Authorization", format!("Token {api_key}"))
        .header("Content-Type", "audio/wav")
        .body(bytes)
        .send()
        .await
        .context("calling Deepgram")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Deepgram request failed ({status}): {body}"));
    }

    let value: Value = response.json().await.context("parsing Deepgram response")?;
    normalize_deepgram(value)
}

fn normalize_deepgram(value: Value) -> Result<NormalizedTranscript> {
    let alternative = value
        .pointer("/results/channels/0/alternatives/0")
        .ok_or_else(|| anyhow!("Deepgram response did not include an alternative transcript"))?;

    let language = value
        .pointer("/metadata/language")
        .and_then(Value::as_str)
        .unwrap_or("en")
        .to_string();

    let duration = value
        .pointer("/metadata/duration")
        .and_then(Value::as_f64)
        .unwrap_or_default();

    let raw_words = alternative
        .get("words")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Deepgram response did not include word timestamps"))?;

    let mut speakers = BTreeSet::new();
    let mut words = Vec::with_capacity(raw_words.len());

    for word in raw_words {
        let text = word
            .get("punctuated_word")
            .or_else(|| word.get("word"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }

        let speaker = word
            .get("speaker")
            .and_then(Value::as_i64)
            .map(|speaker| format!("S{}", speaker + 1));
        if let Some(speaker) = &speaker {
            speakers.insert(speaker.clone());
        }

        words.push(TranscriptWord {
            text,
            start: word
                .get("start")
                .and_then(Value::as_f64)
                .unwrap_or_default(),
            end: word.get("end").and_then(Value::as_f64).unwrap_or_default(),
            speaker,
        });
    }

    let segments = build_segments(&words);

    Ok(NormalizedTranscript {
        language,
        duration,
        speakers: speakers.into_iter().collect(),
        words,
        segments,
    })
}

pub fn build_segments(words: &[TranscriptWord]) -> Vec<TranscriptSegment> {
    let mut segments = Vec::new();
    let mut current: Option<TranscriptSegment> = None;

    for word in words {
        let should_break = current.as_ref().map_or(false, |segment| {
            let pause = word.start - segment.end;
            let speaker_changed = segment.speaker != word.speaker;
            let sentence_end = segment.text.ends_with(['.', '!', '?']);
            pause > 0.9 || speaker_changed || sentence_end
        });

        if should_break {
            if let Some(segment) = current.take() {
                segments.push(segment);
            }
        }

        match &mut current {
            Some(segment) => {
                segment.end = word.end;
                segment.text.push(' ');
                segment.text.push_str(&word.text);
            }
            None => {
                current = Some(TranscriptSegment {
                    start: word.start,
                    end: word.end,
                    speaker: word.speaker.clone(),
                    text: word.text.clone(),
                });
            }
        }
    }

    if let Some(segment) = current {
        segments.push(segment);
    }

    segments
}

pub fn ffmpeg_whisper_exists() -> bool {
    static HAS_FFMPEG_WHISPER: OnceLock<bool> = OnceLock::new();

    *HAS_FFMPEG_WHISPER.get_or_init(|| {
        let output = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-filters"])
            .output();

        match output {
            Ok(output) => {
                let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                text.push_str(&String::from_utf8_lossy(&output.stderr));
                text.lines().any(|line| {
                    let trimmed = line.trim_start();
                    trimmed.starts_with(".. whisper")
                        || trimmed.starts_with("T. whisper")
                        || line.contains(" whisper ")
                })
            }
            Err(_) => false,
        }
    })
}

async fn ensure_whisper_model(model_path: &std::path::Path) -> Result<()> {
    const MODEL_URL: &str =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
    const MIN_EXPECTED_BYTES: u64 = 50_000_000;

    if let Ok(meta) = tokio::fs::metadata(model_path).await {
        if meta.len() >= MIN_EXPECTED_BYTES {
            return Ok(());
        }
    }

    if let Some(parent) = model_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("creating Whisper model directory")?;
    }

    let response = reqwest::Client::new()
        .get(MODEL_URL)
        .send()
        .await
        .context("downloading ggml-base.bin")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Whisper model download failed ({status}): {body}"
        ));
    }

    let bytes = response
        .bytes()
        .await
        .context("reading Whisper model download")?;

    if bytes.len() < MIN_EXPECTED_BYTES as usize {
        return Err(anyhow!(
            "Whisper model download was unexpectedly small ({} bytes)",
            bytes.len()
        ));
    }

    let temp_path = model_path.with_extension("bin.part");
    tokio::fs::write(&temp_path, &bytes)
        .await
        .context("writing temporary Whisper model")?;

    if tokio::fs::metadata(model_path).await.is_ok() {
        let _ = tokio::fs::remove_file(model_path).await;
    }

    tokio::fs::rename(&temp_path, model_path)
        .await
        .context("installing Whisper model")?;

    Ok(())
}

fn parse_srt_timestamp(value: &str) -> Result<f64> {
    let normalized = value.trim().replace(',', ":");
    let parts = normalized.split(':').collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err(anyhow!("Invalid SRT timestamp: {value}"));
    }

    let hours = parts[0].parse::<f64>().context("parsing SRT hours")?;
    let minutes = parts[1].parse::<f64>().context("parsing SRT minutes")?;
    let seconds = parts[2].parse::<f64>().context("parsing SRT seconds")?;
    let millis = parts[3].parse::<f64>().context("parsing SRT milliseconds")?;

    Ok(hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0)
}

fn normalize_ffmpeg_whisper_srt(raw: &str) -> Result<NormalizedTranscript> {
    let normalized = raw.replace("\r\n", "\n");
    let mut segments = Vec::new();
    let mut words = Vec::new();

    for block in normalized.split("\n\n") {
        let lines = block
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();

        if lines.len() < 3 {
            continue;
        }

        let timing = lines[1];
        let mut timing_parts = timing.split("-->");
        let start_text = timing_parts
            .next()
            .ok_or_else(|| anyhow!("Missing SRT start timestamp"))?;
        let end_text = timing_parts
            .next()
            .ok_or_else(|| anyhow!("Missing SRT end timestamp"))?;

        let start = parse_srt_timestamp(start_text)?;
        let end = parse_srt_timestamp(end_text)?;
        let text = lines[2..].join(" ").trim().to_string();

        if text.is_empty() || end <= start {
            continue;
        }

        segments.push(TranscriptSegment {
            start,
            end,
            speaker: Some("S1".to_string()),
            text: text.clone(),
        });

        let tokens = text
            .split_whitespace()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();

        if !tokens.is_empty() {
            let step = (end - start) / tokens.len() as f64;

            for (index, token) in tokens.iter().enumerate() {
                let word_start = start + step * index as f64;
                let word_end = if index + 1 == tokens.len() {
                    end
                } else {
                    start + step * (index + 1) as f64
                };

                words.push(TranscriptWord {
                    text: (*token).to_string(),
                    start: word_start,
                    end: word_end,
                    speaker: Some("S1".to_string()),
                });
            }
        }
    }

    if segments.is_empty() {
        return Err(anyhow!(
            "FFmpeg Whisper completed but no transcript segments were produced"
        ));
    }

    let duration = segments.last().map(|segment| segment.end).unwrap_or(0.0);

    Ok(NormalizedTranscript {
        language: "auto".to_string(),
        duration,
        speakers: vec!["S1".to_string()],
        words,
        segments,
    })
}

pub async fn transcribe_local(
    audio_path: &str,
    data_dir: &str,
) -> Result<NormalizedTranscript> {
    if !ffmpeg_whisper_exists() {
        return Err(anyhow!(
            "Your FFmpeg build does not include the whisper filter. Install an FFmpeg build compiled with --enable-whisper."
        ));
    }

    let model_path = std::path::Path::new(data_dir)
        .join("models")
        .join("ggml-tiny.bin");

    ensure_whisper_model(&model_path).await?;

    let model_dir = model_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid Whisper model directory"))?
        .to_path_buf();

    let model_name = model_path
        .file_name()
        .ok_or_else(|| anyhow!("Invalid Whisper model file name"))?
        .to_string_lossy()
        .to_string();

    let output_name = format!("autoshorts-whisper-{}.srt", uuid::Uuid::new_v4());
    let output_path = model_dir.join(&output_name);

    let run_whisper = |use_gpu: bool| {
        let audio_path_owned = audio_path.to_string();
        let model_dir_for_command = model_dir.clone();
        let model_name = model_name.clone();
        let output_name = output_name.clone();

        tokio::task::spawn_blocking(move || -> Result<std::process::Output> {
            let null_sink = if cfg!(windows) { "NUL" } else { "/dev/null" };
            let filter = format!(
                "whisper=model={}:language=lock:queue=20:use_gpu={}:destination={}:format=srt",
                model_name,
                if use_gpu { "true" } else { "false" },
                output_name
            );

            std::process::Command::new("ffmpeg")
                .current_dir(&model_dir_for_command)
                .arg("-hide_banner")
                .arg("-loglevel")
                .arg("warning")
                .arg("-y")
                .arg("-i")
                .arg(&audio_path_owned)
                .arg("-vn")
                .arg("-af")
                .arg(&filter)
                .arg("-f")
                .arg("null")
                .arg(null_sink)
                .output()
                .context("executing FFmpeg Whisper filter")
        })
    };

    let mut output = run_whisper(true)
        .await
        .context("FFmpeg Whisper GPU worker failed")??;

    if !output.status.success() {
        output = run_whisper(false)
            .await
            .context("FFmpeg Whisper CPU fallback worker failed")??;
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = tokio::fs::remove_file(&output_path).await;

        return Err(anyhow!(
            "FFmpeg Whisper failed.\nStderr: {}\nStdout: {}",
            stderr,
            stdout
        ));
    }

    let srt = tokio::fs::read_to_string(&output_path)
        .await
        .context("reading FFmpeg Whisper transcript")?;

    let _ = tokio::fs::remove_file(&output_path).await;

    normalize_ffmpeg_whisper_srt(&srt)
}
