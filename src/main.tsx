import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import {
  AudioLines,
  BadgeCheck,
  Captions,
  Check,
  ChevronRight,
  Clapperboard,
  Download,
  FileVideo,
  Loader2,
  Play,
  RefreshCw,
  Scissors,
  SlidersHorizontal,
  Sparkles,
  Wand2,
  Copy,
  Database,
  Cloud,
  Youtube,
  AlertTriangle,
} from "lucide-react";
import "./styles.css";

type EnvironmentStatus = {
  dataDir: string;
  hasFfmpeg: boolean;
  hasFfprobe: boolean;
  hasDeepgramKey: boolean;
  hasAnthropicKey: boolean;
  hasDeepseekKey: boolean;
  hasGeminiKey: boolean;
  hasOpenaiKey: boolean;
  hasOpenrouterKey: boolean;
  hasGroqKey: boolean;
  llmProvider: string;
  hasLocalWhisperModel: boolean;
  hasOllama: boolean;
  hasYtdlp: boolean;
};

type Project = {
  id: string;
  name: string | null;
  sourcePath: string;
  sourceDuration: number | null;
  status: string;
  transcriptionMode: string;
  captionStyle?: string | null;
  captionLanguage?: string | null;
  createdAt: string;
  updatedAt: string;
};

type Transcript = {
  id: string;
  projectId: string;
  engine: string;
  rawJson: string;
  language: string | null;
  createdAt: string;
};

type Candidate = {
  id: string;
  projectId: string;
  startSec: number;
  endSec: number;
  score: number;
  hook: string;
  rationale: string;
  rank: number;
  selected: boolean;
};

type Clip = {
  id: string;
  candidateId: string;
  status: string;
  outputPath: string | null;
  faceTrackJson: string | null;
  captionAssPath: string | null;
  renderLog: string | null;
};

type ProjectDetail = {
  project: Project;
  transcript: Transcript | null;
  candidates: Candidate[];
  clips: Clip[];
};

type NormalizedTranscript = {
  language: string;
  duration: number;
  speakers: string[];
  segments: Array<{
    start: number;
    end: number;
    speaker: string | null;
    text: string;
  }>;
  words?: Array<{
    text: string;
    start: number;
    end: number;
    speaker?: string | null;
  }>;
};

type BusyState =
  | "idle"
  | "import"
  | "transcribe"
  | "demoTranscript"
  | "moments"
  | "clipCount"
  | "cut";

type TranscriptionProgress = {
  projectId: string;
  processedSec: number;
  totalSec: number;
  percentage: number;
  elapsedSec: number;
  speed: number;
  etaSec: number | null;
  usingGpu: boolean;
};

function App() {
  const [environment, setEnvironment] = useState<EnvironmentStatus | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [busy, setBusy] = useState<BusyState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [renderingCandidateId, setRenderingCandidateId] = useState<string | null>(null);
  const [showStyleModal, setShowStyleModal] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState("modern-box");
  const [selectedCaptionLanguage, setSelectedCaptionLanguage] = useState(() =>
    localStorage.getItem("autoshorts_caption_language") || "hinglish"
  );
  const [mediaPathToImport, setMediaPathToImport] = useState<string | null>(null);

  const [youtubeModalOpen, setYoutubeModalOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeStatus, setYoutubeStatus] = useState<"idle" | "checking" | "warning" | "downloading">("idle");
  const [youtubeWarningLicense, setYoutubeWarningLicense] = useState<string | null>(null);

  // Persistence logic from localStorage
  const [isOnboarded, setIsOnboarded] = useState<boolean | null>(null);
  const [transcriptionEngine, setTranscriptionEngine] = useState<"deepgram" | "local">(() => {
    return (localStorage.getItem("autoshorts_transcription_engine") as "deepgram" | "local") || "local";
  });
  const [llmEngine, setLlmEngine] = useState<"claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq">(() => {
    return (localStorage.getItem("autoshorts_llm_engine") as "claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq") || "openrouter";
  });
  const [localLlmModel, setLocalLlmModel] = useState(() => {
    return localStorage.getItem("autoshorts_local_llm_model") || "llama3.2";
  });
  const [deepgramKey, setDeepgramKey] = useState(() => {
    return localStorage.getItem("autoshorts_deepgram_key") || "";
  });
  const [anthropicKey, setAnthropicKey] = useState(() => {
    return localStorage.getItem("autoshorts_anthropic_key") || "";
  });
  const [deepseekKey, setDeepseekKey] = useState(() => {
    return localStorage.getItem("autoshorts_deepseek_key") || "";
  });
  const [deepseekModel, setDeepseekModel] = useState(() => {
    return localStorage.getItem("autoshorts_deepseek_model") || "";
  });
  const [geminiKey, setGeminiKey] = useState(() => {
    return localStorage.getItem("autoshorts_gemini_key") || "";
  });
  const [openaiKey, setOpenaiKey] = useState(() => {
    return localStorage.getItem("autoshorts_openai_key") || "";
  });
  const [openrouterKey, setOpenrouterKey] = useState(() => {
    return localStorage.getItem("autoshorts_openrouter_key") || "";
  });
  const [groqKey, setGroqKey] = useState(() => {
    return localStorage.getItem("autoshorts_groq_key") || "";
  });
  const [openrouterModel, setOpenrouterModel] = useState(() => {
    return localStorage.getItem("autoshorts_openrouter_model") || "nvidia/nemotron-3-ultra-550b-a55b:free";
  });

  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [modelDownloadStatus, setModelDownloadStatus] = useState("");
  const [modelDownloadProgress, setModelDownloadProgress] = useState(0);
  const [transcriptionProgress, setTranscriptionProgress] = useState<TranscriptionProgress | null>(null);
  const [momentInstructions, setMomentInstructions] = useState(() =>
    localStorage.getItem("autoshorts_moment_instructions") ||
    "Find the strongest viral moments. Prioritize funny reactions, rage, arguments, unexpected moments, clutch gameplay, strong opinions and clear payoffs. Avoid boring normal gameplay and context-heavy clips."
  );
  const [previewCandidateId, setPreviewCandidateId] = useState<string | null>(null);
  const [previewTime, setPreviewTime] = useState(0);

  const transcript = useMemo(() => {
    if (!detail?.transcript) return null;
    try {
      return JSON.parse(detail.transcript.rawJson) as NormalizedTranscript;
    } catch {
      return null;
    }
  }, [detail?.transcript]);

  const previewCandidate = useMemo(
    () => detail?.candidates.find((candidate) => candidate.id === previewCandidateId) ?? null,
    [detail?.candidates, previewCandidateId],
  );

  const selectedCount = detail?.candidates.filter((candidate) => candidate.selected).length ?? 0;
  const clipByCandidate = useMemo(() => {
    return new Map(detail?.clips.map((clip) => [clip.candidateId, clip]) ?? []);
  }, [detail?.clips]);
  const previewClip = previewCandidate ? clipByCandidate.get(previewCandidate.id) ?? null : null;
  const previewIsRendered = previewClip?.status === "done" && Boolean(previewClip.outputPath);

  const previewCaptionText = useMemo(() => {
    if (!previewCandidate || previewIsRendered || !transcript) return "";

    const words = transcript.words ?? [];
    if (words.length > 0) {
      const activeIndex = words.findIndex((word) => word.end > previewTime && word.start <= previewTime);
      if (activeIndex >= 0) {
        const chunkStart = Math.floor(activeIndex / 2) * 2;
        return words
          .slice(chunkStart, chunkStart + 2)
          .filter((word) => word.end > previewCandidate.startSec && word.start < previewCandidate.endSec)
          .map((word) => word.text)
          .join(" ")
          .toUpperCase();
      }
    }

    const segment = transcript.segments.find(
      (item) => item.end > previewTime && item.start <= previewTime
    );
    if (!segment) return "";

    return segment.text
      .trim()
      .split(/\s+/)
      .slice(0, 5)
      .join(" ")
      .toUpperCase();
  }, [previewCandidate, previewIsRendered, previewTime, transcript]);
  const selectedCandidates = detail?.candidates.filter((candidate) => candidate.selected) ?? [];
  const selectedCutCount = selectedCandidates.filter((candidate) => {
    const clip = clipByCandidate.get(candidate.id);
    return clip?.status === "done" && Boolean(clip.outputPath);
  }).length;
  const selectedCaptionsCount = selectedCandidates.filter((candidate) => {
    const clip = clipByCandidate.get(candidate.id);
    return clip?.status === "done" && Boolean(clip.captionAssPath);
  }).length;
  const canUseCloudKey = environment?.hasDeepgramKey || deepgramKey.trim().length > 0;
  const canUseClaude = environment?.hasAnthropicKey || anthropicKey.trim().length > 0;
  const canUseDeepseek = environment?.hasDeepseekKey || deepseekKey.trim().length > 0;
  const canUseGemini = environment?.hasGeminiKey || geminiKey.trim().length > 0;
  const canUseOpenai = environment?.hasOpenaiKey || openaiKey.trim().length > 0;
  const canUseOpenrouter = environment?.hasOpenrouterKey || openrouterKey.trim().length > 0;
  const canUseGroq = environment?.hasGroqKey || groqKey.trim().length > 0;

  const canTranscribe = transcriptionEngine === "local"
    ? Boolean(environment?.hasLocalWhisperModel)
    : canUseCloudKey;

  const canUseActiveLlm = llmEngine === "local"
    ? Boolean(environment?.hasOllama)
    : llmEngine === "claude"
      ? canUseClaude
      : llmEngine === "deepseek"
        ? canUseDeepseek
        : llmEngine === "gemini"
          ? canUseGemini
          : llmEngine === "openai"
            ? canUseOpenai
            : llmEngine === "openrouter"
              ? canUseOpenrouter
              : llmEngine === "groq"
                ? canUseGroq
                : false;

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    void listen<TranscriptionProgress>("transcription-progress", (event) => {
      const payload = event.payload;

      if (!detail?.project.id || payload.projectId === detail.project.id) {
        setTranscriptionProgress(payload);
      }
    }).then((cleanup) => {
      unlisten = cleanup;
    });

    return () => {
      unlisten?.();
    };
  }, [detail?.project.id]);

  useEffect(() => {
    void refresh();

    if (localStorage.getItem("autoshorts_custom_defaults_applied") !== "true") {
      const customModel = "nvidia/nemotron-3-ultra-550b-a55b:free";

      setTranscriptionEngine("local");
      setLlmEngine("openrouter");
      setOpenrouterModel(customModel);

      localStorage.setItem("autoshorts_transcription_engine", "local");
      localStorage.setItem("autoshorts_llm_engine", "openrouter");
      localStorage.setItem("autoshorts_openrouter_model", customModel);
      localStorage.setItem("autoshorts_custom_defaults_applied", "true");
    }

    localStorage.setItem("autoshorts_onboarded", "true");
    setIsOnboarded(true);
  }, []);

  useEffect(() => {
    localStorage.setItem("autoshorts_transcription_engine", transcriptionEngine);
  }, [transcriptionEngine]);

  useEffect(() => {
    localStorage.setItem("autoshorts_llm_engine", llmEngine);
  }, [llmEngine]);

  useEffect(() => {
    localStorage.setItem("autoshorts_local_llm_model", localLlmModel);
  }, [localLlmModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepgram_key", deepgramKey);
  }, [deepgramKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_anthropic_key", anthropicKey);
  }, [anthropicKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepseek_key", deepseekKey);
  }, [deepseekKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepseek_model", deepseekModel);
  }, [deepseekModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_gemini_key", geminiKey);
  }, [geminiKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openai_key", openaiKey);
  }, [openaiKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openrouter_key", openrouterKey);
  }, [openrouterKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_groq_key", groqKey);
  }, [groqKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openrouter_model", openrouterModel);
  }, [openrouterModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_moment_instructions", momentInstructions);
  }, [momentInstructions]);

  useEffect(() => {
    localStorage.setItem("autoshorts_caption_language", selectedCaptionLanguage);
  }, [selectedCaptionLanguage]);

  const pullModelDirectly = async (modelName: string) => {
    setDownloadingModelName(modelName);
    setModelDownloadProgress(0);
    setModelDownloadStatus("Connecting to Ollama...");
    try {
      const unlisten = await listen<{
        status: string;
        completed?: number;
        total?: number;
        percentage?: number;
      }>("ollama-pull-progress", (event) => {
        const payload = event.payload;
        setModelDownloadStatus(payload.status);
        if (payload.percentage !== undefined && payload.percentage !== null) {
          setModelDownloadProgress(Math.round(payload.percentage));
        }
      });

      await invoke("pull_ollama_model", { modelName });
      unlisten();
      setModelDownloadStatus("Download complete!");
      setModelDownloadProgress(100);
      setTimeout(() => setDownloadingModelName(null), 500);
    } catch (err) {
      alert("Failed to download model: " + String(err));
      setDownloadingModelName(null);
    }
  };

  async function refresh(nextProjectId?: string) {
    setError(null);

    if (nextProjectId) {
      const [projectList, nextDetail] = await Promise.all([
        invoke<Project[]>("list_projects"),
        invoke<ProjectDetail>("get_project_detail", { projectId: nextProjectId }),
      ]);
      setProjects(projectList);
      setDetail(nextDetail);
      return;
    }

    const [env, projectList] = await Promise.all([
      invoke<EnvironmentStatus>("environment_status"),
      invoke<Project[]>("list_projects"),
    ]);
    setEnvironment(env);
    setProjects(projectList);
    setDetail(null);
  }

  async function run(action: BusyState, task: () => Promise<void>) {
    setBusy(action);
    setError(null);
    try {
      await task();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("idle");
    }
  }

  async function importMedia() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "Media",
          extensions: ["mp4", "mov", "mp3", "wav", "m4a"],
        },
      ],
    });
    if (typeof selected !== "string") return;
    setMediaPathToImport(selected);
    setShowStyleModal(true);
  }

  async function handleYoutubeImport() {
    if (!youtubeUrl) return;

    // FAST MODE: directly download instead of doing a second yt-dlp metadata request first.
    await executeYoutubeDownload();
  }

  async function executeYoutubeDownload() {
    setYoutubeStatus("downloading");
    setError(null);
    try {
      const downloadedPath = await invoke<string>("download_youtube_video", { url: youtubeUrl });
      setYoutubeModalOpen(false);
      setYoutubeUrl("");
      setYoutubeStatus("idle");
      setMediaPathToImport(downloadedPath);
      setShowStyleModal(true);
    } catch (err: any) {
      setError(err.toString());
      setYoutubeStatus("idle");
    }
  }

  async function confirmImport(style: string, captionLanguage: string) {
    if (!mediaPathToImport) return;
    const selected = mediaPathToImport;
    setMediaPathToImport(null);
    setShowStyleModal(false);

    let newProjectId: string | null = null;
    await run("import", async () => {
      const project = await invoke<Project>("create_project_from_path", {
        path: selected,
        transcriptionMode: transcriptionEngine === "local" ? "local" : "cloud",
        captionStyle: style,
        captionLanguage,
      });
      newProjectId = project.id;
      await refresh(project.id);
    });

    if (newProjectId) {
      await runAutoPipeline(newProjectId);
    }
  }

  async function runAutoPipeline(projectId: string) {
    setError(null);
    const env = environment ?? await invoke<EnvironmentStatus>("environment_status");

    if (transcriptionEngine === "local") {
      if (!env.hasLocalWhisperModel) {
        setError("Import successful. Local Whisper is unavailable because this FFmpeg build does not expose the whisper filter.");
        return;
      }
    } else {
      const hasDG = env.hasDeepgramKey || deepgramKey.trim().length > 0;
      if (!hasDG) {
        setError("Import successful. Deepgram key is missing. Please add it to start transcription.");
        return;
      }
    }

    try {
      setTranscriptionProgress(null);
      setBusy("transcribe");
      await invoke<Transcript>("transcribe_project", {
        projectId,
        provider: transcriptionEngine,
        apiKey: transcriptionEngine === "deepgram" ? (deepgramKey.trim() || null) : null,
      });
      await refresh(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("idle");
    }
  }

  async function renameProject(projectId: string) {
    const project = projects.find((p) => p.id === projectId);
    if (!project) return;
    const currentName = project.name || fileName(project.sourcePath);
    const newName = window.prompt("Rename Project:", currentName);
    if (newName === null) return;
    const trimmed = newName.trim();
    if (!trimmed) return;

    try {
      await invoke("rename_project", { projectId, name: trimmed });
      await refresh(detail?.project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function deleteProject(projectId: string) {
    const project = projects.find((p) => p.id === projectId);
    if (!project) return;
    const name = project.name || fileName(project.sourcePath);
    if (!window.confirm(`Are you sure you want to delete the project "${name}"?`)) return;

    try {
      await invoke("delete_project", { projectId });
      const nextActiveId = detail?.project.id === projectId ? null : detail?.project.id;
      await refresh(nextActiveId ?? undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function selectProject(projectId: string) {
    await run("idle", async () => {
      const nextDetail = await invoke<ProjectDetail>("get_project_detail", { projectId });
      setDetail(nextDetail);
    });
  }

  async function transcribe() {
    if (!detail) return;
    setTranscriptionProgress(null);

    await run("transcribe", async () => {
      await invoke<Transcript>("transcribe_project", {
        projectId: detail.project.id,
        provider: transcriptionEngine,
        apiKey: transcriptionEngine === "deepgram" ? (deepgramKey.trim() || null) : null,
      });
      await refresh(detail.project.id);
    });
  }

  async function moments(allowDemo: boolean) {
    if (!detail) return;
    await run("moments", async () => {
      const activeKey =
        llmEngine === "claude" ? anthropicKey.trim() :
          llmEngine === "deepseek" ? deepseekKey.trim() :
            llmEngine === "gemini" ? geminiKey.trim() :
              llmEngine === "openai" ? openaiKey.trim() :
                llmEngine === "openrouter" ? openrouterKey.trim() :
                  llmEngine === "groq" ? groqKey.trim() : "";
      try {
        await invoke<Candidate[]>("generate_candidates", {
          projectId: detail.project.id,
          apiKey: activeKey || null,
          provider: llmEngine,
          modelName: llmEngine === "local" ? localLlmModel.trim() : (llmEngine === "deepseek" ? (deepseekModel.trim() || null) : (llmEngine === "openrouter" ? (openrouterModel.trim() || null) : null)),
          instructions: momentInstructions.trim() || null,
          allowDemo,
        });
        await refresh(detail.project.id);
      } catch (err) {
        const errMsg = String(err);
        if (llmEngine === "local" && (errMsg.includes("not found") || errMsg.includes("404"))) {
          if (window.confirm(`Ollama model "${localLlmModel}" is not downloaded. Would you like to download it now?`)) {
            setTimeout(() => {
              void pullModelDirectly(localLlmModel).then(() => {
                void refresh(detail.project.id);
              });
            }, 100);
            return;
          }
        }
        throw err;
      }
    });
  }

  async function updateProjectCaptionSettings(captionStyle: string, captionLanguage: string) {
    if (!detail) return;

    try {
      const project = await invoke<Project>("update_project_caption_settings", {
        projectId: detail.project.id,
        captionStyle,
        captionLanguage,
      });

      setDetail((current) => current
        ? {
            ...current,
            project,
            clips: current.clips.map((clip) => ({
              ...clip,
              status: "pending",
              outputPath: null,
              faceTrackJson: null,
              captionAssPath: null,
              renderLog: null,
            })),
          }
        : current);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function setCandidateApproved(candidateId: string, selected: boolean) {
    if (!detail) return;
    try {
      const updated = await invoke<Candidate>("set_candidate_selected", { candidateId, selected });
      setDetail((current) => current
        ? { ...current, candidates: current.candidates.map((candidate) => candidate.id === updated.id ? updated : candidate) }
        : current);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function adjustCandidate(candidate: Candidate, edge: "start" | "end", delta: number) {
    if (!detail) return;

    const maxDuration = detail.project.sourceDuration ?? transcript?.duration ?? Number.POSITIVE_INFINITY;
    let startSec = candidate.startSec;
    let endSec = candidate.endSec;

    if (edge === "start") {
      startSec = Math.max(0, Math.min(candidate.startSec + delta, endSec - 1));
    } else {
      endSec = Math.max(startSec + 1, candidate.endSec + delta);
      if (Number.isFinite(maxDuration)) {
        endSec = Math.min(endSec, maxDuration);
      }
    }

    try {
      const updated = await invoke<Candidate>("update_candidate_timing", {
        candidateId: candidate.id,
        startSec,
        endSec,
      });
      setDetail((current) => current
        ? {
            ...current,
            candidates: current.candidates.map((item) => item.id === updated.id ? updated : item),
            clips: current.clips.map((clip) => clip.candidateId === updated.id
              ? {
                  ...clip,
                  status: "pending",
                  outputPath: null,
                  faceTrackJson: null,
                  captionAssPath: null,
                  renderLog: null,
                }
              : clip),
          }
        : current);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function updateClipCount(count: number) {
    if (!detail) return;
    await run("clipCount", async () => {
      const candidates = await invoke<Candidate[]>("set_selected_clip_count", {
        projectId: detail.project.id,
        count,
      });
      setDetail({ ...detail, candidates });
    });
  }

  async function cutCandidate(candidateId: string) {
    if (!detail) return;
    setRenderingCandidateId(candidateId);
    setBusy("cut");
    setError(null);
    try {
      await invoke<string>("render_flat_clip_for_candidate", {
        candidateId,
        captionApiKey: openrouterKey.trim() || null,
        captionModelName: openrouterModel.trim() || null,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRenderingCandidateId(null);
      setBusy("idle");
      await refresh(detail.project.id);
    }
  }

  async function cutSelected() {
    if (!detail) return;
    setBusy("cut");
    setError(null);
    try {
      for (const candidate of selectedCandidates) {
        setRenderingCandidateId(candidate.id);
        await invoke<string>("render_flat_clip_for_candidate", {
          candidateId: candidate.id,
          captionApiKey: openrouterKey.trim() || null,
          captionModelName: openrouterModel.trim() || null,
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRenderingCandidateId(null);
      setBusy("idle");
      await refresh(detail.project.id);
    }
  }

  if (isOnboarded === null) {
    return (
      <div className="onboarding-loading" style={{ display: 'grid', placeItems: 'center', height: '100vh', background: 'var(--bg-base)' }}>
        <Loader2 className="spin" size={32} color="var(--accent-primary)" />
      </div>
    );
  }

  if (isOnboarded === false) {
    return (
      <Onboarding
        environment={environment}
        onComplete={() => setIsOnboarded(true)}
        setTranscriptionEngine={setTranscriptionEngine}
        setLlmEngine={setLlmEngine}
        setLocalLlmModel={setLocalLlmModel}
        setDeepgramKey={setDeepgramKey}
        setAnthropicKey={setAnthropicKey}
        setDeepseekKey={setDeepseekKey}
        setGroqKey={setGroqKey}
        deepgramKey={deepgramKey}
        anthropicKey={anthropicKey}
        deepseekKey={deepseekKey}
        groqKey={groqKey}
        refreshEnv={() => refresh()}
      />
    );
  }

  return (
    <div className="app-shell-container">
      <main className="app-shell">
        <aside className="sidebar">
          <div
            className="brand-row"
            onClick={() => setDetail(null)}
            style={{ cursor: "pointer" }}
            title="Go to Home Dashboard"
          >
            <div className="brand-mark">
              <Clapperboard size={20} />
            </div>
            <div>
              <h1>AutoShorts</h1>
              <p>Long recording in. Short clips out.</p>
            </div>
          </div>

          <button className="primary-action" onClick={importMedia} disabled={busy !== "idle"}>
            {busy === "import" ? <Loader2 className="spin" size={18} /> : <FileVideo size={18} />}
            Import recording
          </button>
          <button 
            className="secondary-action" 
            onClick={() => setYoutubeModalOpen(true)} 
            disabled={busy !== "idle" || !environment?.hasYtdlp}
            title={!environment?.hasYtdlp ? "Please install yt-dlp to use this feature" : "Download a video from YouTube"}
            style={{ width: "100%", padding: "0.75rem", borderRadius: "10px", marginTop: "0.5rem", display: "flex", gap: "0.5rem", alignItems: "center", justifyContent: "center", border: "1px solid var(--border)", background: "transparent", color: "var(--foreground)", cursor: "pointer", fontSize: "0.95rem" }}
          >
            <Youtube size={18} />
            Import from YouTube
          </button>

          <section className="project-list" aria-label="Projects">
            <button
              className={`project-row ${!detail ? "active" : ""}`}
              onClick={() => setDetail(null)}
            >
              <Clapperboard size={15} />
              <span>All Projects</span>
              <ChevronRight size={14} />
            </button>

            {projects.map((project) => (
              <button
                key={project.id}
                className={`project-row ${detail?.project.id === project.id ? "active" : ""}`}
                onClick={() => void selectProject(project.id)}
              >
                <FileVideo size={15} />
                <span>{project.name || fileName(project.sourcePath)}</span>
                <ChevronRight size={14} />
              </button>
            ))}
          </section>
        </aside>

        <section className="workspace">
          {detail ? (
            <>
              <header className="topbar">
                <div className="project-info">
                  <div className="eyebrow">{detail.project.status}</div>
                  <h2>{detail.project.name || fileName(detail.project.sourcePath)}</h2>
                </div>
                <div className="topbar-actions">
                  <button
                    className={`icon-button settings-toggle ${showSettings ? "active" : ""}`}
                    onClick={() => setShowSettings(!showSettings)}
                    title="API Settings"
                  >
                    <SlidersHorizontal size={16} />
                    <span>API Settings</span>
                  </button>
                  <button className="icon-button" onClick={() => void refresh(detail.project.id)} title="Refresh">
                    <RefreshCw size={18} />
                  </button>
                </div>
              </header>

              {showSettings && (
                <div className="settings-panel">
                  <div className="key-stack-horizontal">
                    <label>
                      <span>Transcription Engine</span>
                      <select
                        value={transcriptionEngine}
                        onChange={(event) => setTranscriptionEngine(event.target.value as "deepgram" | "local")}
                      >
                        <option value="local">Local Whisper (Offline)</option>
                        <option value="deepgram">Deepgram (Cloud)</option>
                      </select>
                      <span>LLM Engine</span>
                      <select
                        value={llmEngine}
                        onChange={(event) => setLlmEngine(event.target.value as any)}
                      >
                        <option value="local">Ollama (Offline Local)</option>
                        <option value="claude">Claude (Cloud)</option>
                        <option value="deepseek">DeepSeek (Cloud)</option>
                        <option value="gemini">Google Gemini (Cloud)</option>
                        <option value="openai">OpenAI (Cloud)</option>
                        <option value="openrouter">OpenRouter (Cloud)</option>
                        <option value="groq">Groq (Cloud)</option>
                      </select>
                    </label>
                    {transcriptionEngine === "deepgram" && (
                      <label>
                        <span>Deepgram API Key</span>
                        <input
                          value={deepgramKey}
                          onChange={(event) => setDeepgramKey(event.target.value)}
                          placeholder={environment?.hasDeepgramKey ? "Loaded from env" : "Optional (Deepgram API Key)"}
                          type="password"
                        />
                      </label>
                    )}
                    {llmEngine === "claude" && (
                      <label>
                        <span>Claude API Key</span>
                        <input
                          value={anthropicKey}
                          onChange={(event) => setAnthropicKey(event.target.value)}
                          placeholder={environment?.hasAnthropicKey ? "Loaded from env" : "Optional (Claude API Key)"}
                          type="password"
                        />
                      </label>
                    )}
                    {llmEngine === "deepseek" && (
                      <>
                        <label>
                          <span>DeepSeek API Key</span>
                          <input
                            value={deepseekKey}
                            onChange={(event) => setDeepseekKey(event.target.value)}
                            placeholder={environment?.hasDeepseekKey ? "Loaded from env" : "Optional (DeepSeek API Key)"}
                            type="password"
                          />
                        </label>
                        <label>
                          <span>DeepSeek Model</span>
                          <input
                            value={deepseekModel}
                            onChange={(event) => setDeepseekModel(event.target.value)}
                            placeholder="Optional (e.g. deepseek-chat, deepseek-v4-pro)"
                            type="text"
                          />
                        </label>
                      </>
                    )}
                    {llmEngine === "gemini" && (
                      <label>
                        <span>Gemini API Key</span>
                        <input
                          value={geminiKey}
                          onChange={(event) => setGeminiKey(event.target.value)}
                          placeholder={environment?.hasGeminiKey ? "Loaded from env" : "Optional (Gemini API Key)"}
                          type="password"
                        />
                      </label>
                    )}
                    {llmEngine === "openai" && (
                      <label>
                        <span>OpenAI API Key</span>
                        <input
                          value={openaiKey}
                          onChange={(event) => setOpenaiKey(event.target.value)}
                          placeholder={environment?.hasOpenaiKey ? "Loaded from env" : "Optional (OpenAI API Key)"}
                          type="password"
                        />
                      </label>
                    )}
                    {llmEngine === "openrouter" && (
                      <>
                        <label>
                          <span>OpenRouter API Key</span>
                          <input
                            value={openrouterKey}
                            onChange={(event) => setOpenrouterKey(event.target.value)}
                            placeholder={environment?.hasOpenrouterKey ? "Loaded from env" : "Optional (OpenRouter API Key)"}
                            type="password"
                          />
                        </label>
                        <label>
                          <span>OpenRouter Model</span>
                          <input
                            value={openrouterModel}
                            onChange={(event) => setOpenrouterModel(event.target.value)}
                            placeholder="Optional (e.g. google/gemini-2.5-flash, deepseek/deepseek-r1)"
                            type="text"
                          />
                        </label>
                      </>
                    )}
                    {llmEngine === "groq" && (
                      <label>
                        <span>Groq API Key</span>
                        <input
                          value={groqKey}
                          onChange={(event) => setGroqKey(event.target.value)}
                          placeholder={environment?.hasGroqKey ? "Loaded from env" : "Optional (Groq API Key)"}
                          type="password"
                        />
                      </label>
                    )}

                    {llmEngine === "local" && (
                      <label>
                        <span>Ollama Model Name</span>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <input
                            value={localLlmModel}
                            onChange={(event) => setLocalLlmModel(event.target.value)}
                            placeholder="e.g. llama3.2, qwen2.5:7b"
                            type="text"
                          />
                          <button
                            type="button"
                            className="icon-button"
                            style={{ minHeight: '36px', height: '36px' }}
                            onClick={() => pullModelDirectly(localLlmModel)}
                          >
                            <Download size={14} /> Pull
                          </button>
                        </div>
                      </label>
                    )}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
                    <button
                      type="button"
                      className="icon-button"
                      style={{ background: 'rgba(239, 68, 68, 0.08)', borderColor: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}
                      onClick={() => {
                        if (window.confirm("Are you sure you want to reset your configuration and restart onboarding from scratch?")) {
                          localStorage.clear();
                          window.location.reload();
                        }
                      }}
                    >
                      Reset App Configuration & Onboarding
                    </button>
                  </div>
                </div>
              )}

              {error && <div className="error-banner">{error}</div>}

              <div className="pipeline-strip">
                <PipelineStep icon={<AudioLines size={16} />} label="Transcript" done={Boolean(detail.transcript)} />
                <PipelineStep icon={<Sparkles size={16} />} label="Moments" done={detail.candidates.length > 0} />
                <PipelineStep icon={<Scissors size={16} />} label="Cut" done={selectedCount > 0 && selectedCutCount === selectedCount} />
                <PipelineStep icon={<Captions size={16} />} label="Captions" done={selectedCount > 0 && selectedCaptionsCount === selectedCount} />
                <PipelineStep icon={<Download size={16} />} label="Export" done={selectedCount > 0 && selectedCutCount === selectedCount} />
              </div>

              <div className="work-grid">
                <section className="panel transcript-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Transcript</h3>
                      <p>{transcript ? `${transcript.segments.length} segments` : "No transcript"}</p>
                    </div>
                    <div className="button-pair">
                      <button onClick={transcribe} disabled={busy !== "idle" || !canTranscribe}>
                        {busy === "transcribe" ? <Loader2 className="spin" size={16} /> : <AudioLines size={16} />}
                        Transcribe
                      </button>
                    </div>
                  </div>

                  {busy === "transcribe" && transcriptionEngine === "local" && (
                    <div style={{
                      margin: "0 0 14px",
                      padding: "14px",
                      border: "1px solid var(--border)",
                      borderRadius: "10px",
                      background: "rgba(255,255,255,0.025)",
                    }}>
                      <div style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: "12px",
                        marginBottom: "9px",
                      }}>
                        <strong>Transcribing...</strong>
                        <span>{Math.round(transcriptionProgress?.percentage ?? 0)}%</span>
                      </div>

                      <div style={{
                        height: "8px",
                        overflow: "hidden",
                        borderRadius: "999px",
                        background: "rgba(255,255,255,0.09)",
                        marginBottom: "10px",
                      }}>
                        <div style={{
                          height: "100%",
                          width: `${Math.max(1, transcriptionProgress?.percentage ?? 0)}%`,
                          borderRadius: "inherit",
                          background: "var(--accent-primary)",
                          transition: "width 350ms ease",
                        }} />
                      </div>

                      {transcriptionProgress ? (
                        <div style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                          gap: "6px 16px",
                          fontSize: "0.82rem",
                          opacity: 0.82,
                        }}>
                          <span>
                            Processed: {formatDurationLong(transcriptionProgress.processedSec)}
                            {" / "}
                            {formatDurationLong(transcriptionProgress.totalSec)}
                          </span>
                          <span>Elapsed: {formatDurationLong(transcriptionProgress.elapsedSec)}</span>
                          <span>
                            Speed: {transcriptionProgress.speed > 0
                              ? `${transcriptionProgress.speed.toFixed(1)}x realtime`
                              : "Calculating..."}
                          </span>
                          <span>
                            Estimated remaining: {transcriptionProgress.etaSec != null
                              ? `~${formatDurationLong(transcriptionProgress.etaSec)}`
                              : "Calculating..."}
                          </span>
                          <span>Mode: {transcriptionProgress.usingGpu ? "GPU" : "CPU fallback"}</span>
                        </div>
                      ) : (
                        <div style={{ fontSize: "0.82rem", opacity: 0.75 }}>
                          Starting Local Whisper and waiting for the first progress update...
                        </div>
                      )}
                    </div>
                  )}

                  {!canTranscribe && (
                    <div className="api-warning">
                      {transcriptionEngine === "local"
                        ? "⚠️ Local Whisper requires an FFmpeg build with the whisper filter enabled."
                        : "⚠️ Deepgram API Key is missing. Transcribing will not work. Please add your key in API Settings."}
                    </div>
                  )}

                  <div className="transcript-list">
                    {transcript?.segments.map((segment, index) => (
                      <article key={`${segment.start}-${index}`} className="segment-row">
                        <span>{formatTime(segment.start)}</span>
                        <p>{segment.text}</p>
                      </article>
                    )) ?? <EmptyState icon={<AudioLines size={28} />} label="Transcript pending" />}
                  </div>
                </section>

                <section className="panel candidate-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Clip Candidates</h3>
                      <p>{detail.candidates.length ? `${selectedCount} approved / ${detail.candidates.length} found` : "No candidates yet"}</p>
                    </div>
                    <div className="button-pair">
                      <button onClick={() => void moments(false)} disabled={busy !== "idle" || !detail.transcript || !canUseActiveLlm}>
                        {busy === "moments" ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}
                        Find Moments
                      </button>
                      <button
                        onClick={cutSelected}
                        disabled={
                          busy !== "idle" ||
                          selectedCount === 0 ||
                          !environment?.hasFfmpeg ||
                          ((detail.project.captionLanguage ?? "original") !== "original" && !canUseOpenrouter)
                        }
                      >
                        {busy === "cut" ? <Loader2 className="spin" size={16} /> : <Scissors size={16} />}
                        Render Approved
                      </button>
                    </div>
                  </div>

                  <div className="caption-settings-bar">
                    <div className="caption-setting">
                      <span>Caption Style</span>
                      <select
                        value={detail.project.captionStyle ?? "modern-box"}
                        onChange={(event) => void updateProjectCaptionSettings(
                          event.target.value,
                          detail.project.captionLanguage ?? "original",
                        )}
                        disabled={busy !== "idle"}
                      >
                        <option value="modern-box">Modern Box</option>
                        <option value="classic-outline">Classic Outline</option>
                        <option value="minimal-shadow">Minimal Shadow</option>
                        <option value="vibrant-cyan">Vibrant Cyan</option>
                        <option value="vibrant-yellow-box">Vibrant Yellow Box</option>
                        <option value="vibrant-green">Vibrant Green</option>
                        <option value="vibrant-red">Vibrant Red</option>
                      </select>
                    </div>

                    <div className="caption-setting">
                      <span>Caption Language</span>
                      <select
                        value={detail.project.captionLanguage ?? "original"}
                        onChange={(event) => void updateProjectCaptionSettings(
                          detail.project.captionStyle ?? "modern-box",
                          event.target.value,
                        )}
                        disabled={busy !== "idle"}
                      >
                        <option value="original">Auto / Original</option>
                        <option value="english">English</option>
                        <option value="hindi">Hindi (देवनागरी)</option>
                        <option value="hinglish">Hinglish (Roman Hindi + English)</option>
                      </select>
                    </div>

                    <div className="caption-setting-note">
                      Full-clip captions are burned into every approved render.
                      {(detail.project.captionLanguage ?? "original") !== "original"
                        ? " OpenRouter rewrites the timed captions into the selected language before FFmpeg renders."
                        : " Auto / Original keeps Whisper's detected wording."}
                    </div>
                  </div>

                  {(detail.project.captionLanguage ?? "original") !== "original" && !canUseOpenrouter && (
                    <div className="api-warning">
                      ⚠️ English / Hindi / Hinglish caption conversion needs your OpenRouter API key. Add it in API Settings or choose Auto / Original.
                    </div>
                  )}
                  {!canUseActiveLlm && (
                    <div className="api-warning">
                      {llmEngine === "local"
                        ? "⚠️ Ollama local server is not running at http://localhost:11434. Moment detection will not work."
                        : `⚠️ ${llmEngine === "claude" ? "Claude" :
                          llmEngine === "deepseek" ? "DeepSeek" :
                            llmEngine === "gemini" ? "Gemini" :
                              llmEngine === "openai" ? "OpenAI" :
                                llmEngine === "openrouter" ? "OpenRouter" :
                                  llmEngine === "groq" ? "Groq" : "LLM"
                        } API Key is missing. Viral moment identification will not work. Please add your key in API Settings.`}
                    </div>
                  )}

                  <div className="moment-instructions-box">
                    <div className="moment-instructions-heading">
                      <div>
                        <strong>AI Instructions</strong>
                        <span>Tell OpenRouter exactly what kind of moments you want before analysis.</span>
                      </div>
                    </div>
                    <textarea
                      value={momentInstructions}
                      onChange={(event) => setMomentInstructions(event.target.value)}
                      rows={4}
                      placeholder="Example: Find funny rage, arguments, unexpected reactions and clutch moments. Avoid normal gameplay."
                      disabled={busy === "moments"}
                    />
                    <div className="moment-presets">
                      <button type="button" onClick={() => setMomentInstructions("Find the strongest viral moments with a hook in the first 3 seconds. Prefer high-energy reactions, surprising turns and clear payoffs. Avoid boring setup.")}>Viral</button>
                      <button type="button" onClick={() => setMomentInstructions("Find funny moments, rage, arguments, fails and unexpected reactions. Avoid normal gameplay and weak context-heavy moments.")}>Funny / Rage</button>
                      <button type="button" onClick={() => setMomentInstructions("Find insane gameplay, clutch plays, close fights, smart outplays and strong reactions immediately after them.")}>Clutch</button>
                      <button type="button" onClick={() => setMomentInstructions("Find controversial statements, strong opinions, debates, arguments and moments likely to trigger comments or discussion.")}>Controversial</button>
                    </div>
                  </div>

                  {previewCandidate && (
                    <div className="candidate-source-preview">
                      <div className="candidate-source-preview-header">
                        <div>
                          <strong>
                            {previewIsRendered
                              ? `Final rendered clip #${previewCandidate.rank}`
                              : `Previewing candidate #${previewCandidate.rank}`}
                          </strong>
                          <span>
                            {previewIsRendered
                              ? `captions burned in · ${detail.project.captionLanguage ?? "original"} · ${detail.project.captionStyle ?? "modern-box"}`
                              : `${formatTime(previewCandidate.startSec)} - ${formatTime(previewCandidate.endSec)} · live caption-style preview`}
                          </span>
                        </div>
                        <button type="button" onClick={() => setPreviewCandidateId(null)}>Close</button>
                      </div>

                      <div className="candidate-video-stage">
                        <video
                          key={`${previewIsRendered ? previewClip?.outputPath : detail.project.sourcePath}-${previewCandidate.id}-${previewCandidate.startSec}-${previewCandidate.endSec}`}
                          className="candidate-source-video"
                          src={convertFileSrc(
                            previewIsRendered && previewClip?.outputPath
                              ? previewClip.outputPath
                              : detail.project.sourcePath
                          )}
                          controls
                          preload="metadata"
                          onLoadedMetadata={(event) => {
                            if (previewIsRendered) {
                              event.currentTarget.currentTime = 0;
                              setPreviewTime(previewCandidate.startSec);
                            } else {
                              event.currentTarget.currentTime = previewCandidate.startSec;
                              setPreviewTime(previewCandidate.startSec);
                            }
                            void event.currentTarget.play().catch(() => undefined);
                          }}
                          onPlay={(event) => {
                            if (
                              !previewIsRendered &&
                              (event.currentTarget.currentTime < previewCandidate.startSec ||
                                event.currentTarget.currentTime >= previewCandidate.endSec)
                            ) {
                              event.currentTarget.currentTime = previewCandidate.startSec;
                            }
                          }}
                          onTimeUpdate={(event) => {
                            if (previewIsRendered) return;
                            setPreviewTime(event.currentTarget.currentTime);
                            if (event.currentTarget.currentTime >= previewCandidate.endSec) {
                              event.currentTarget.pause();
                            }
                          }}
                        />

                        {!previewIsRendered && previewCaptionText && (
                          <div className={`live-caption-preview caption-${detail.project.captionStyle ?? "modern-box"}`}>
                            {previewCaptionText}
                          </div>
                        )}
                      </div>

                      {!previewIsRendered && (detail.project.captionLanguage ?? "original") !== "original" && (
                        <div className="preview-caption-note">
                          Style/placement is previewed live. The exact {detail.project.captionLanguage} wording is generated by OpenRouter during render; after rendering, Preview plays the exact final clip with burned captions.
                        </div>
                      )}

                      {previewIsRendered && previewClip?.outputPath && (
                        <div className="output-path">
                          Final clip: {previewClip.outputPath}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="candidate-list">
                    {detail.candidates.map((candidate) => {
                      const clip = clipByCandidate.get(candidate.id);
                      const isCut = clip?.status === "done" && Boolean(clip.outputPath);
                      const isPreviewing = previewCandidateId === candidate.id;
                      return (
                        <article key={candidate.id} className={`candidate-card ${candidate.selected ? "selected" : ""}`}>
                          <div className="portrait-preview-container">
                            <button
                              type="button"
                              className={`portrait-preview-mock preview-trigger ${isPreviewing ? "active" : ""}`}
                              onClick={() => setPreviewCandidateId(candidate.id)}
                              title="Preview this moment from the original video"
                            >
                              <div className="mock-video-active">
                                <Play size={20} className="play-icon-mock" />
                              </div>
                            </button>
                            <div className="candidate-rank">
                              <span>#{candidate.rank}</span>
                              {candidate.selected && <Check size={14} />}
                            </div>
                          </div>

                          <div className="candidate-body">
                            <div className="candidate-meta">
                              <span>{formatTime(candidate.startSec)} - {formatTime(candidate.endSec)}</span>
                              <span className="candidate-score">{Math.round(candidate.score * 100)}% Match</span>
                            </div>
                            <h4>{candidate.hook}</h4>
                            <p className="candidate-rationale">{candidate.rationale}</p>

                            <div className="timing-adjust">
                              <span>Start</span>
                              <button type="button" onClick={() => void adjustCandidate(candidate, "start", -5)}>-5s</button>
                              <button type="button" onClick={() => void adjustCandidate(candidate, "start", 5)}>+5s</button>
                              <span>End</span>
                              <button type="button" onClick={() => void adjustCandidate(candidate, "end", -5)}>-5s</button>
                              <button type="button" onClick={() => void adjustCandidate(candidate, "end", 5)}>+5s</button>
                            </div>

                            <div className="candidate-actions review-actions">
                              <span className={`clip-status ${isCut ? "ready" : clip?.status === "error" ? "error" : candidate.selected ? "ready" : ""}`}>
                                {isCut ? "Rendered" : clip?.status === "error" ? "Render failed" : candidate.selected ? "Approved" : "Needs review"}
                              </span>
                              <div className="review-button-group">
                                <button type="button" className={`review-button preview ${isPreviewing ? "active" : ""}`} onClick={() => setPreviewCandidateId(candidate.id)}>
                                  <Play size={14} /> {isCut ? "Preview Final" : "Preview"}
                                </button>
                                <button type="button" className="review-button approve" onClick={() => void setCandidateApproved(candidate.id, true)} disabled={candidate.selected}>
                                  <Check size={14} /> {candidate.selected ? "Approved" : "Approve"}
                                </button>
                                <button type="button" className="review-button reject" onClick={() => void setCandidateApproved(candidate.id, false)} disabled={!candidate.selected}>
                                  Reject
                                </button>
                              </div>
                            </div>

                            {clip?.outputPath && <div className="output-path">{clip.outputPath}</div>}
                            {clip?.captionAssPath && (
                              <div className="output-path" style={{ background: "rgba(142, 230, 199, 0.05)", borderColor: "var(--accent-primary)", color: "var(--accent-primary)", marginTop: "4px" }}>
                                Subtitles: {clip.captionAssPath}
                              </div>
                            )}
                            {clip?.renderLog && <div className="render-log">{clip.renderLog}</div>}
                          </div>
                        </article>
                      );
                    })}
                    {detail.candidates.length === 0 && <EmptyState icon={<Sparkles size={28} />} label={detail.transcript ? "Add AI instructions, then Find Moments" : "Transcribe first"} />}
                  </div>
                </section>              </div>
            </>
          ) : (
            <div className="home-dashboard">
              <header className="home-header">
                <div>
                  <h2>All Projects</h2>
                  <p>Select a project below or import a new media file to get started.</p>
                </div>
                <button className="primary-action compact" onClick={importMedia} disabled={busy !== "idle"}>
                  {busy === "import" ? <Loader2 className="spin" size={18} /> : <FileVideo size={18} />}
                  Import recording
                </button>
                <button 
                  className="secondary-action compact" 
                  onClick={() => setYoutubeModalOpen(true)} 
                  disabled={busy !== "idle" || !environment?.hasYtdlp}
                  title={!environment?.hasYtdlp ? "Please install yt-dlp to use this feature" : "Download a video from YouTube"}
                  style={{ display: "flex", gap: "0.5rem", alignItems: "center", padding: "0.5rem 1rem", borderRadius: "8px", border: "1px solid var(--border)", background: "transparent", color: "var(--foreground)", cursor: "pointer", fontSize: "0.95rem", marginLeft: "1rem" }}
                >
                  <Youtube size={18} />
                  Import from YouTube
                </button>
              </header>

              {projects.length > 0 ? (
                <div className="projects-grid">
                  {projects.map((project) => {
                    const name = project.name || fileName(project.sourcePath);
                    return (
                      <article key={project.id} className="project-card">
                        <div className="project-card-header">
                          <FileVideo size={24} className="project-card-icon" />
                          <span className="project-card-status">{project.status}</span>
                        </div>
                        <h3 className="project-card-title">{name}</h3>
                        <div className="project-card-meta">
                          <span>Duration: {project.sourceDuration ? formatDurationLong(project.sourceDuration) : "Probing..."}</span>
                          <span>Created: {new Date(project.createdAt).toLocaleDateString()}</span>
                        </div>
                        <div className="project-card-actions">
                          <button className="action-btn open-btn" onClick={() => void selectProject(project.id)}>
                            Open
                          </button>
                          <button className="action-btn rename-btn" onClick={() => void renameProject(project.id)}>
                            Rename
                          </button>
                          <button className="action-btn delete-btn" onClick={() => void deleteProject(project.id)}>
                            Delete
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-dashboard-state">
                  <Clapperboard size={48} className="empty-state-icon" />
                  <h3>No projects found</h3>
                  <p>Import your first recording to begin creating shorts.</p>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="status-bar">
        <div className="status-bar-left">
          <span className="app-status-indicator">System Ready</span>
        </div>
        <div className="status-bar-right">
          <div className="status-indicators">
            <span className={`indicator ${environment?.hasFfmpeg ? "active" : ""}`} title="FFmpeg status">ffmpeg</span>
            <span className={`indicator ${environment?.hasFfprobe ? "active" : ""}`} title="FFprobe status">ffprobe</span>
            <span className={`indicator ${environment?.hasYtdlp ? "active" : ""}`} title="yt-dlp status">yt-dlp</span>
            <span className={`indicator ${environment?.hasLocalWhisperModel ? "active" : ""}`} title="Whisper Model status">Whisper Model</span>
            <span className={`indicator ${environment?.hasOllama ? "active" : ""}`} title="Ollama status">Ollama</span>
            <span className={`indicator ${canUseCloudKey ? "active" : ""}`} title="Deepgram Key status">Deepgram</span>
            <span className={`indicator ${canUseClaude ? "active" : ""}`} title="Claude Key status">Claude</span>
            <span className={`indicator ${canUseDeepseek ? "active" : ""}`} title="DeepSeek Key status">DeepSeek</span>
          </div>
        </div>
      </footer>

      {showStyleModal && (
        <div className="style-modal-overlay">
          <div className="style-modal">
            <div className="style-modal-header">
              <h3>Choose Caption Style</h3>
              <p>Select how your automated captions should look on the portrait short-form video clips.</p>
            </div>

            <div className="style-grid">
              <div
                className={`style-card ${selectedStyle === "modern-box" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("modern-box")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-box">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Modern Box</div>
                <div className="style-card-desc">Sleek white text inside a semi-transparent black background padding box. Highly readable.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "classic-outline" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("classic-outline")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-outline">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Classic Outline</div>
                <div className="style-card-desc">Vibrant bold yellow text with a clean black outline. High-energy CapCut formatting.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "minimal-shadow" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("minimal-shadow")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-shadow">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Minimal Shadow</div>
                <div className="style-card-desc">Pure white text with a soft, elegant drop shadow. Unobtrusive and modern.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "vibrant-cyan" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("vibrant-cyan")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-cyan">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Vibrant Cyan</div>
                <div className="style-card-desc">Vibrant tech cyan text with a black drop shadow for a clean look.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "vibrant-yellow-box" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("vibrant-yellow-box")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-yellow-box">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Vibrant Yellow Box</div>
                <div className="style-card-desc">Bold black text inside a solid yellow padding box. Punchy and high visibility.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "vibrant-green" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("vibrant-green")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-green">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Vibrant Green</div>
                <div className="style-card-desc">High-energy neon green text with black borders and a drop shadow (Hormozi style).</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "vibrant-red" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("vibrant-red")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-red">BRAINFOOD BECAUSE</span>
                </div>
                <div className="style-card-title">Vibrant Red</div>
                <div className="style-card-desc">Dramatic neon crimson text with outline and drop shadow (gaming/action style).</div>
              </div>
            </div>

            <div className="caption-language-section">
              <div className="caption-language-header">
                <h4>Caption Language</h4>
                <p>Choose how the full spoken clip should appear in burned-in captions.</p>
              </div>

              <div className="caption-language-options">
                <button
                  type="button"
                  className={selectedCaptionLanguage === "hinglish" ? "selected" : ""}
                  onClick={() => setSelectedCaptionLanguage("hinglish")}
                >
                  <strong>Hinglish</strong>
                  <span>Roman Hindi + English — best for mixed gaming speech.</span>
                </button>
                <button
                  type="button"
                  className={selectedCaptionLanguage === "english" ? "selected" : ""}
                  onClick={() => setSelectedCaptionLanguage("english")}
                >
                  <strong>English</strong>
                  <span>Translate Hindi/Hinglish speech into natural English.</span>
                </button>
                <button
                  type="button"
                  className={selectedCaptionLanguage === "hindi" ? "selected" : ""}
                  onClick={() => setSelectedCaptionLanguage("hindi")}
                >
                  <strong>Hindi</strong>
                  <span>Natural Hindi captions in Devanagari script.</span>
                </button>
                <button
                  type="button"
                  className={selectedCaptionLanguage === "original" ? "selected" : ""}
                  onClick={() => setSelectedCaptionLanguage("original")}
                >
                  <strong>Auto / Original</strong>
                  <span>Keep Whisper's detected words without translation.</span>
                </button>
              </div>
            </div>
            <div className="style-modal-actions">
              <button className="btn-cancel" onClick={() => { setShowStyleModal(false); setMediaPathToImport(null); }}>
                Cancel
              </button>
              <button className="btn-confirm" onClick={() => confirmImport(selectedStyle, selectedCaptionLanguage)}>
                Confirm & Import
              </button>
            </div>
          </div>
        </div>
      )}
      {downloadingModelName && (
        <div className="onboarding-overlay" style={{ zIndex: 20000 }}>
          <div className="onboarding-card" style={{ maxWidth: '480px', textAlign: 'center' }}>
            <div className="onboarding-header compact" style={{ textAlign: 'center' }}>
              <h2>Downloading Ollama Model</h2>
              <p>Downloading model weights for "{downloadingModelName}". Please do not close the app.</p>
            </div>

            <div className="download-progress-container">
              <div className="download-loader">
                <Loader2 className="spin" size={48} />
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${modelDownloadProgress}%` }}></div>
              </div>

              <div className="download-stats">
                <span className="download-status">{modelDownloadStatus}</span>
                <span className="download-percentage">{modelDownloadProgress}%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {youtubeModalOpen && (
        <div className="style-modal-overlay">
          <div className="style-modal" style={{ maxWidth: "500px" }}>
            <div className="style-modal-header">
              <h3>Import from YouTube</h3>
              <p>Paste a YouTube URL below to download and import it directly.</p>
            </div>
            <div style={{ padding: "1rem" }}>
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                disabled={youtubeStatus !== "idle" && youtubeStatus !== "warning"}
                style={{ width: "100%", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border)", background: "var(--background)", color: "var(--foreground)", fontSize: "1rem", marginBottom: "1rem" }}
              />

              {youtubeStatus === "warning" && (
                <div style={{ background: "rgba(255, 165, 0, 0.1)", border: "1px solid orange", padding: "1rem", borderRadius: "8px", marginBottom: "1rem", color: "orange" }}>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem", fontWeight: "bold" }}>
                    <AlertTriangle size={20} />
                    Copyright Warning
                  </div>
                  <p style={{ margin: 0, fontSize: "0.9rem" }}>
                    This video is not explicitly marked for reuse (Creative Commons). Its license appears to be: <strong>{youtubeWarningLicense}</strong>.
                    <br/><br/>
                    Clipping this video may lead to copyright strikes. Are you sure you want to proceed?
                  </p>
                </div>
              )}

              <div className="style-modal-actions" style={{ marginTop: "1rem" }}>
                <button 
                  className="btn-cancel" 
                  onClick={() => { setYoutubeModalOpen(false); setYoutubeUrl(""); setYoutubeStatus("idle"); }}
                  disabled={youtubeStatus === "checking" || youtubeStatus === "downloading"}
                >
                  Cancel
                </button>
                {youtubeStatus === "warning" ? (
                  <button className="btn-confirm" onClick={executeYoutubeDownload}>
                    Yes, I understand the risks
                  </button>
                ) : (
                  <button 
                    className="btn-confirm" 
                    onClick={handleYoutubeImport}
                    disabled={!youtubeUrl || youtubeStatus !== "idle"}
                    style={{ minWidth: "120px" }}
                  >
                    {youtubeStatus === "checking" ? <><Loader2 className="spin" size={18} /> Checking...</> :
                     youtubeStatus === "downloading" ? <><Loader2 className="spin" size={18} /> Downloading...</> : 
                     "Check & Download"}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusPill({ label, active }: { label: string; active?: boolean }) {
  return (
    <div className={`status-pill ${active ? "active" : ""}`}>
      <BadgeCheck size={14} />
      {label}
    </div>
  );
}

function PipelineStep({ icon, label, done }: { icon: React.ReactNode; label: string; done: boolean }) {
  return (
    <div className={`pipeline-step ${done ? "done" : ""}`}>
      {icon}
      <span>{label}</span>
    </div>
  );
}

function EmptyState({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{label}</span>
    </div>
  );
}

function fileName(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${remaining.toString().padStart(2, "0")}`;
  }

  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

function formatDurationLong(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0s";

  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (hours > 0) {
    return `${hours}h ${minutes.toString().padStart(2, "0")}m ${secs.toString().padStart(2, "0")}s`;
  }

  if (minutes > 0) {
    return `${minutes}m ${secs.toString().padStart(2, "0")}s`;
  }

  return `${secs}s`;
}

interface OnboardingProps {
  environment: EnvironmentStatus | null;
  onComplete: () => void;
  setTranscriptionEngine: (engine: "deepgram" | "local") => void;
  setLlmEngine: (engine: "claude" | "deepseek" | "local" | "groq") => void;
  setLocalLlmModel: (model: string) => void;
  setDeepgramKey: (key: string) => void;
  setAnthropicKey: (key: string) => void;
  setDeepseekKey: (key: string) => void;
  setGroqKey: (key: string) => void;
  deepgramKey: string;
  anthropicKey: string;
  deepseekKey: string;
  groqKey: string;
  refreshEnv: () => Promise<void>;
}

function Onboarding({
  environment,
  onComplete,
  setTranscriptionEngine,
  setLlmEngine,
  setLocalLlmModel,
  setDeepgramKey,
  setAnthropicKey,
  setDeepseekKey,
  setGroqKey,
  deepgramKey: initialDeepgramKey,
  anthropicKey: initialAnthropicKey,
  deepseekKey: initialDeepseekKey,
  groqKey: initialGroqKey,
  refreshEnv,
}: OnboardingProps) {
  const [setupMode, setSetupMode] = useState<"choose" | "local" | "cloud" | "downloading">("choose");
  const [selectedModel, setSelectedModel] = useState<string>("llama3.2");

  const [dgKey, setDgKey] = useState(initialDeepgramKey);
  const [antKey, setAntKey] = useState(initialAnthropicKey);
  const [dsKey, setDsKey] = useState(initialDeepseekKey);
  const [grKey, setGrKey] = useState(initialGroqKey);

  const [downloadStatus, setDownloadStatus] = useState("Initializing download...");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [checkingOllama, setCheckingOllama] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyWhisperCommand = () => {
    navigator.clipboard.writeText("pip3 install -U openai-whisper");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCloudSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!dgKey.trim()) {
      setError("Deepgram API Key is required for cloud mode.");
      return;
    }
    if (!antKey.trim() && !dsKey.trim() && !grKey.trim()) {
      setError("Please provide at least one LLM Key (Claude, DeepSeek, or Groq).");
      return;
    }

    setTranscriptionEngine("deepgram");
    setDeepgramKey(dgKey.trim());
    localStorage.setItem("autoshorts_deepgram_key", dgKey.trim());
    localStorage.setItem("autoshorts_transcription_engine", "deepgram");

    if (antKey.trim()) {
      setLlmEngine("claude");
      setAnthropicKey(antKey.trim());
      localStorage.setItem("autoshorts_anthropic_key", antKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "claude");
    } else if (dsKey.trim()) {
      setLlmEngine("deepseek");
      setDeepseekKey(dsKey.trim());
      localStorage.setItem("autoshorts_deepseek_key", dsKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "deepseek");
    } else if (grKey.trim()) {
      setLlmEngine("groq");
      setGroqKey(grKey.trim());
      localStorage.setItem("autoshorts_groq_key", grKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "groq");
    }

    localStorage.setItem("autoshorts_onboarded", "true");
    onComplete();
  };

  const startLocalSetup = async () => {
    setError(null);
    setCheckingOllama(true);
    setDownloadProgress(0);

    await refreshEnv();

    let isOllamaRunning = false;
    try {
      const currentEnv = await invoke<EnvironmentStatus>("environment_status");
      isOllamaRunning = currentEnv.hasOllama;
    } catch (e) {
      // ignore
    }

    setCheckingOllama(false);

    if (!isOllamaRunning) {
      setSetupMode("downloading");
      setDownloadStatus("Ollama not found. Starting automatic installer...");

      try {
        const unlistenInstall = await listen<string>("ollama-install-status", (event) => {
          setDownloadStatus(event.payload);
        });

        await invoke("install_ollama");
        unlistenInstall();
      } catch (err) {
        setError("Automatic installation failed: " + String(err) + ". Please install it manually from ollama.com.");
        setSetupMode("local");
        return;
      }
    }

    setSetupMode("downloading");
    setDownloadStatus("Ollama connected. Initiating model download...");

    try {
      const unlisten = await listen<{
        status: string;
        completed?: number;
        total?: number;
        percentage?: number;
      }>("ollama-pull-progress", (event) => {
        const payload = event.payload;
        setDownloadStatus(payload.status);
        if (payload.percentage !== undefined && payload.percentage !== null) {
          setDownloadProgress(Math.round(payload.percentage));
        }
      });

      await invoke("pull_ollama_model", { modelName: selectedModel });

      unlisten();

      setTranscriptionEngine("local");
      setLlmEngine("local");
      setLocalLlmModel(selectedModel);

      localStorage.setItem("autoshorts_transcription_engine", "local");
      localStorage.setItem("autoshorts_llm_engine", "local");
      localStorage.setItem("autoshorts_local_llm_model", selectedModel);
      localStorage.setItem("autoshorts_onboarded", "true");

      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSetupMode("local");
    }
  };

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-card">
        {setupMode === "choose" && (
          <>
            <div className="onboarding-header">
              <div className="brand-mark large">
                <Clapperboard size={36} />
              </div>
              <h2>Welcome to AutoShorts</h2>
              <p>Long recording in. Short clips out. Select how you would like to run the studio.</p>
            </div>

            <div className="onboarding-choices">
              <div className="choice-card clickable" onClick={() => setSetupMode("local")}>
                <div className="choice-icon">
                  <Database size={28} />
                </div>
                <h3>Fully Offline & Private</h3>
                <p>Process everything locally on your computer. Private, secure, and completely free.</p>
                <div className="choice-badge local">Offline (Ollama)</div>
              </div>

              <div className="choice-card clickable" onClick={() => setSetupMode("cloud")}>
                <div className="choice-icon">
                  <Cloud size={28} />
                </div>
                <h3>Cloud APIs</h3>
                <p>Use high-speed cloud services for transcription and analysis. No local GPU needed.</p>
                <div className="choice-badge cloud">API Keys Required</div>
              </div>
            </div>
          </>
        )}

        {setupMode === "local" && (
          <div className="local-setup-flow">
            <div className="onboarding-header compact">
              <h2>Configure Offline Mode</h2>
              <p>Follow these steps to set up your local studio.</p>
            </div>

            {error && <div className="error-banner" style={{ marginBottom: "16px" }}>{error}</div>}

            <div className="setup-steps">
              <div className="setup-step">
                <div className="step-num">1</div>
                <div className="step-body">
                  <h4>Install Python Whisper</h4>
                  <p>Open your terminal and run the following command to install the transcription engine:</p>
                  <div className="code-block-container">
                    <code>pip3 install -U openai-whisper</code>
                    <button type="button" className="copy-btn" onClick={copyWhisperCommand}>
                      {copied ? <Check size={14} /> : <Copy size={14} />}
                      {copied ? "Copied!" : "Copy"}
                    </button>
                  </div>
                  {environment?.hasLocalWhisperModel ? (
                    <span className="step-check success"><BadgeCheck size={14} /> Whisper installed in Python!</span>
                  ) : (
                    <span className="step-check warning">⚠️ Python package 'whisper' not detected yet. Run the command above.</span>
                  )}
                </div>
              </div>

              <div className="setup-step">
                <div className="step-num">2</div>
                <div className="step-body">
                  <h4>Set up local LLM (Ollama)</h4>
                  <p>
                    Ollama must be installed and running on your machine.
                    If you don't have it installed, you can download it from <a href="https://ollama.com" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-primary)', textDecoration: 'underline' }}>ollama.com</a>.
                  </p>
                  <p>Select a model to download:</p>

                  <div className="model-cards">
                    <div
                      className={`model-card ${selectedModel === "llama3.2" ? "active" : ""}`}
                      onClick={() => setSelectedModel("llama3.2")}
                    >
                      <div className="model-card-header">
                        <h5>LLaMA 3.2 3B</h5>
                        <span className="model-size">1.9 GB</span>
                      </div>
                      <p>Requires 8GB+ RAM. Recommended for standard setups. Fast and efficient.</p>
                    </div>

                    <div
                      className={`model-card ${selectedModel === "qwen2.5:3b" ? "active" : ""}`}
                      onClick={() => setSelectedModel("qwen2.5:3b")}
                    >
                      <div className="model-card-header">
                        <h5>Qwen 2.5 3B</h5>
                        <span className="model-size">2.0 GB</span>
                      </div>
                      <p>Requires 8GB+ RAM. Excellent coding and logical reasoning abilities.</p>
                    </div>

                    <div
                      className={`model-card ${selectedModel === "qwen2.5:7b" ? "active" : ""}`}
                      onClick={() => setSelectedModel("qwen2.5:7b")}
                    >
                      <div className="model-card-header">
                        <h5>Qwen 2.5 7B</h5>
                        <span className="model-size">4.7 GB</span>
                      </div>
                      <p>Requires 16GB+ RAM. High-quality moment detection and hook precision.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="onboarding-actions">
              <button type="button" className="icon-button" onClick={() => setSetupMode("choose")}>Back</button>
              <button
                type="button"
                className="primary-action compact"
                onClick={startLocalSetup}
                disabled={checkingOllama}
              >
                {checkingOllama ? <Loader2 className="spin" size={18} /> : null}
                {checkingOllama ? "Checking Ollama..." : "Download & Start Setup"}
              </button>
            </div>
          </div>
        )}

        {setupMode === "cloud" && (
          <form className="cloud-setup-flow" onSubmit={handleCloudSubmit}>
            <div className="onboarding-header compact">
              <h2>Configure Cloud APIs</h2>
              <p>Add your keys below. AutoShorts will route transcription and analysis to the cloud.</p>
            </div>

            {error && <div className="error-banner" style={{ marginBottom: "16px" }}>{error}</div>}

            <div className="form-stack">
              <div className="input-group">
                <label>Deepgram API Key *</label>
                <input
                  type="password"
                  value={dgKey}
                  onChange={(e) => setDgKey(e.target.value)}
                  placeholder="Insert your Deepgram API Key (for transcription)"
                />
              </div>

              <div className="input-group">
                <label>Claude API Key</label>
                <input
                  type="password"
                  value={antKey}
                  onChange={(e) => setAntKey(e.target.value)}
                  placeholder="Insert your Anthropic API Key (moment detection)"
                />
              </div>

              <div className="input-group">
                <label>DeepSeek API Key</label>
                <input
                  type="password"
                  value={dsKey}
                  onChange={(e) => setDsKey(e.target.value)}
                  placeholder="Insert your DeepSeek API Key (alternative moment detection)"
                />
              </div>

              <div className="input-group">
                <label>Groq API Key</label>
                <input
                  type="password"
                  value={grKey}
                  onChange={(e) => setGrKey(e.target.value)}
                  placeholder="Insert your Groq API Key (alternative moment detection)"
                />
              </div>
              <p className="form-help">* Deepgram Key + at least one LLM Key (Claude, DeepSeek, or Groq) is required.</p>
            </div>

            <div className="onboarding-actions">
              <button type="button" className="icon-button" onClick={() => setSetupMode("choose")}>Back</button>
              <button type="submit" className="primary-action compact">Save & Start</button>
            </div>
          </form>
        )}

        {setupMode === "downloading" && (
          <div className="downloading-flow">
            <div className="onboarding-header compact">
              <h2>Downloading Local Model</h2>
              <p>Please wait while your local environment is downloaded. Do not close the application.</p>
            </div>

            <div className="download-progress-container">
              <div className="download-loader">
                <Loader2 className="spin" size={48} />
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${downloadProgress}%` }}></div>
              </div>

              <div className="download-stats">
                <span className="download-status">{downloadStatus}</span>
                <span className="download-percentage">{downloadProgress}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
