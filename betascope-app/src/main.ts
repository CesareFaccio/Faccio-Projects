import { extractPose, UnsupportedVideoError } from "./poseExtraction";
import type { PoseData, FrameEntry } from "./types";
import { computeAnalysis, DEFAULT_ANALYSIS_OPTIONS } from "./analysis";
import type { AnalysisResult, AnalysisOptions } from "./analysis";
import { computeWeightDistribution } from "./forces";
import { computeMotion } from "./motion";
import type { MotionFrameEntry } from "./motion";
import { buildSmoothedWeightByFrame, renderPlusOverlay } from "./plusRender";
import type { SmoothedWeightFrame } from "./plusRender";

const fileInput = document.getElementById("file-input") as HTMLInputElement;
const dropzone = document.getElementById("dropzone") as HTMLDivElement;
const pipelineStepsEl = document.getElementById("pipeline-steps") as HTMLOListElement;
const stepModelEl = document.getElementById("step-model") as HTMLLIElement;
const stepExtractEl = document.getElementById("step-extract") as HTMLLIElement;
const errorNoteEl = document.getElementById("error-note") as HTMLDivElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const progressBar = document.getElementById("progress-bar") as HTMLDivElement;
const progressWrap = document.getElementById("progress-wrap") as HTMLDivElement;
const resultEl = document.getElementById("result") as HTMLDivElement;
const canvas = document.getElementById("preview-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const downloadJsonBtn = document.getElementById("download-json") as HTMLButtonElement;
const downloadVideoBtn = document.getElementById("download-video") as HTMLButtonElement;
const analysisSettingsEl = document.getElementById("analysis-settings") as HTMLDetailsElement;
const optHoldDurationEl = document.getElementById("opt-hold-duration") as HTMLInputElement;
const optGapFramesEl = document.getElementById("opt-gap-frames") as HTMLInputElement;
const optVelocityThresholdEl = document.getElementById("opt-velocity-threshold") as HTMLInputElement;
const optSmoothWindowEl = document.getElementById("opt-smooth-window") as HTMLInputElement;
const applySettingsBtn = document.getElementById("apply-settings") as HTMLButtonElement;
const settingsNoteEl = document.getElementById("settings-note") as HTMLParagraphElement;

const DOWNLOAD_VIDEO_DEFAULT_LABEL = "Download video with overlay";
// Bundled demo climb, shown automatically on first page load so visitors see
// a real analysis without having to upload anything themselves. Lives in
// public/ so Vite copies it as-is; BASE_URL keeps it correct under the
// GitHub Pages subpath (same pattern as poseExtraction.ts's WASM/model paths).
const DEFAULT_CLIMB_VIDEO_URL = `${import.meta.env.BASE_URL}default-climb.mp4`;
// Measured with ffprobe when the demo clip was encoded — passed to
// extractPose() so it can skip its play()-based fps estimation (which
// needs a real user gesture on some browsers) for the auto-loaded demo.
const DEFAULT_CLIMB_FPS = 60;

let currentPoseData: PoseData | null = null;
let currentAnalysis: AnalysisResult | null = null;
let currentWeightByFrame: Map<number, SmoothedWeightFrame> | null = null;
let currentMotionByFrame: Map<number, MotionFrameEntry> | null = null;
let videoWidth = 0;
let videoHeight = 0;
let activeVideoEl: HTMLVideoElement | null = null;
let playbackRafId: number | null = null;
let isBusy = false; // extracting or recording — ignore new drops meanwhile

function setStatus(text: string) {
  statusEl.textContent = text;
}

type StepState = "pending" | "active" | "done";
function setStep(el: HTMLLIElement, state: StepState) {
  el.classList.toggle("active", state === "active");
  el.classList.toggle("done", state === "done");
}
function resetSteps() {
  for (const el of [stepModelEl, stepExtractEl]) setStep(el, "pending");
}

function showError(text: string) {
  errorNoteEl.textContent = text;
  errorNoteEl.hidden = false;
}
function hideError() {
  errorNoteEl.hidden = true;
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Runs the full hold/CoM/force/motion analysis pipeline for the given pose
 * data and options, updating all the module-level analysis state and debug
 * hooks. Shared by the initial post-extraction run and by the "Apply &
 * re-analyze" button, which re-runs it with user-edited AnalysisOptions
 * without re-extracting poses from the video.
 */
function runAnalysis(data: PoseData, options: AnalysisOptions) {
  currentAnalysis = computeAnalysis(data, options);
  const weight = computeWeightDistribution(data, currentAnalysis);
  currentWeightByFrame = buildSmoothedWeightByFrame(weight);
  const motionArr = computeMotion(currentAnalysis.com, data.video.fps);
  currentMotionByFrame = motionArr
    ? new Map(motionArr.filter((m) => m.speed_px_s !== null).map((m) => [m.frame, m]))
    : null;
  (window as any).__betascopeAnalysis = currentAnalysis; // debugging convenience
  (window as any).__betascopeWeight = weight; // debugging convenience
  (window as any).__betascopeWeightByFrame = currentWeightByFrame; // debugging convenience
  (window as any).__betascopeMotionByFrame = currentMotionByFrame; // debugging convenience
}

/** Fills the analysis-settings inputs with the given options, converting the frame-based hold threshold to seconds for the given fps. */
function populateSettingsInputs(options: AnalysisOptions, fps: number) {
  optHoldDurationEl.value = (options.minHoldFrames / fps).toFixed(2);
  optGapFramesEl.value = String(options.maxGapFrames);
  optVelocityThresholdEl.value = String(options.velocityThreshold);
  optSmoothWindowEl.value = String(options.smoothWindow);
}

/** Reads the analysis-settings inputs into an AnalysisOptions, converting the seconds-based hold threshold back to frames for the given fps. Falls back to the default for any blank/invalid field. */
function readSettingsInputs(fps: number): AnalysisOptions {
  const holdDurationS = parseFloat(optHoldDurationEl.value);
  const gapFrames = parseInt(optGapFramesEl.value, 10);
  const velocityThreshold = parseFloat(optVelocityThresholdEl.value);
  const smoothWindow = parseInt(optSmoothWindowEl.value, 10);
  return {
    minHoldFrames:
      Number.isFinite(holdDurationS) && holdDurationS > 0
        ? Math.max(1, Math.round(holdDurationS * fps))
        : DEFAULT_ANALYSIS_OPTIONS.minHoldFrames,
    maxGapFrames:
      Number.isFinite(gapFrames) && gapFrames >= 0 ? gapFrames : DEFAULT_ANALYSIS_OPTIONS.maxGapFrames,
    velocityThreshold:
      Number.isFinite(velocityThreshold) && velocityThreshold > 0
        ? velocityThreshold
        : DEFAULT_ANALYSIS_OPTIONS.velocityThreshold,
    smoothWindow:
      Number.isFinite(smoothWindow) && smoothWindow >= 1 ? smoothWindow : DEFAULT_ANALYSIS_OPTIONS.smoothWindow,
  };
}

function frameAtTime(t: number): FrameEntry | null {
  if (!currentPoseData) return null;
  const { fps } = currentPoseData.video;
  const idx = Math.min(
    currentPoseData.landmarks.length - 1,
    Math.max(0, Math.round(t * fps))
  );
  return currentPoseData.landmarks[idx] ?? null;
}

/** Draws the climbing_plus.mp4-equivalent dual-panel overlay (left: weight/motion schematic, right: skeleton + forces + CoM + holds) — assumes the video frame itself is already drawn into the canvas's right half. */
function drawOverlayForFrame(entry: FrameEntry | null) {
  renderPlusOverlay(ctx, entry, currentAnalysis, currentWeightByFrame, currentMotionByFrame, videoWidth, videoHeight);
}

/** Draws the video's current frame (into the right half only — the left half is the synthetic schematic) plus the overlay onto the canvas. */
function drawCurrentFrame() {
  if (!activeVideoEl) return;
  ctx.drawImage(activeVideoEl, videoWidth, 0, videoWidth, videoHeight);
  drawOverlayForFrame(frameAtTime(activeVideoEl.currentTime));
}

function loopTick() {
  drawCurrentFrame();
  playbackRafId = requestAnimationFrame(loopTick);
}

function startLoopPlayback() {
  stopLoopPlayback();
  if (!activeVideoEl) return;
  activeVideoEl.loop = true;
  activeVideoEl.currentTime = 0;
  activeVideoEl.play().catch(() => {
    // Autoplay can be blocked in some contexts; the still frame from the
    // last progress tick stays visible, which is a fine fallback.
  });
  playbackRafId = requestAnimationFrame(loopTick);
}

function stopLoopPlayback() {
  if (playbackRafId !== null) {
    cancelAnimationFrame(playbackRafId);
    playbackRafId = null;
  }
  activeVideoEl?.pause();
}

async function handleFile(file: File, isDemo = false, knownFps?: number) {
  if (isBusy) return;
  isBusy = true;
  stopLoopPlayback();
  currentPoseData = null;
  currentAnalysis = null;
  currentWeightByFrame = null;
  currentMotionByFrame = null;
  activeVideoEl = null;

  resultEl.hidden = true;
  pipelineStepsEl.hidden = false;
  resetSteps();
  hideError();
  progressWrap.hidden = false;
  progressBar.style.width = "0%";
  downloadJsonBtn.hidden = true;
  downloadVideoBtn.hidden = true;
  downloadVideoBtn.disabled = false;
  downloadVideoBtn.textContent = DOWNLOAD_VIDEO_DEFAULT_LABEL;
  analysisSettingsEl.hidden = true;
  analysisSettingsEl.open = false;
  settingsNoteEl.textContent = "";
  setStatus("");

  const startedAt = performance.now();

  try {
    const result = await extractPose(file, (p) => {
      if (p.phase === "loading-model") {
        setStep(stepModelEl, "active");
        setStatus("Loading pose model…");
      } else if (p.phase === "loading-video") {
        setStep(stepModelEl, "done");
        setStep(stepExtractEl, "active");
        setStatus("Loading video…");
      } else if (p.phase === "estimating-fps") {
        setStep(stepExtractEl, "active");
        setStatus("Reading video frame rate…");
      } else if (p.phase === "extracting" && p.totalFrames) {
        setStep(stepExtractEl, "active");
        const pct = Math.round(((p.frame ?? 0) / p.totalFrames) * 100);
        progressBar.style.width = `${pct}%`;
        setStatus(`Extracting pose: frame ${p.frame}/${p.totalFrames} (${pct}%)`);
      } else if (p.phase === "done") {
        setStep(stepExtractEl, "done");
        setStatus("Done.");
      }
    }, knownFps);

    setStep(stepExtractEl, "done");

    const data = result.data;
    currentPoseData = data;
    runAnalysis(data, DEFAULT_ANALYSIS_OPTIONS);
    populateSettingsInputs(DEFAULT_ANALYSIS_OPTIONS, data.video.fps);
    activeVideoEl = result.videoElement;
    (window as any).__betascopePoseData = data; // debugging convenience
    (window as any).__betascopeVideoEl = activeVideoEl; // debugging convenience

    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    const detectedCount = data.landmarks.filter((f) => f.detected).length;
    const detectionRate = ((detectedCount / data.landmarks.length) * 100).toFixed(1);

    videoWidth = data.video.width;
    videoHeight = data.video.height;
    canvas.width = data.video.width * 2; // side-by-side: schematic panel + video panel
    canvas.height = data.video.height;

    resultEl.hidden = false;
    downloadJsonBtn.hidden = false;
    downloadVideoBtn.hidden = false;
    analysisSettingsEl.hidden = false;
    progressBar.style.width = "100%";
    setStatus(
      `${isDemo ? "Demo climb — " : ""}Processed ${data.landmarks.length} frames in ${elapsed}s — ` +
        `${detectionRate}% detection rate (${data.video.width}x${data.video.height} @ ${data.video.fps.toFixed(2)}fps)`
    );

    startLoopPlayback();
  } catch (err) {
    if (err instanceof UnsupportedVideoError) {
      showError(err.message);
      setStatus("Couldn't process this video — see the note above.");
    } else {
      setStatus(`Something went wrong: ${(err as Error).message}`);
      console.error(err);
    }
    progressWrap.hidden = true;
  } finally {
    isBusy = false;
  }
}

applySettingsBtn.addEventListener("click", () => {
  if (!currentPoseData || isBusy) return;
  const options = readSettingsInputs(currentPoseData.video.fps);
  runAnalysis(currentPoseData, options);
  populateSettingsInputs(options, currentPoseData.video.fps); // reflect any clamped/defaulted values back
  if (!activeVideoEl || activeVideoEl.paused) drawCurrentFrame(); // playing loop will pick it up on its own next tick
  const handCount = currentAnalysis!.handholds.length;
  const footCount = currentAnalysis!.footholds.length;
  settingsNoteEl.textContent = `Re-analyzed: ${handCount} handhold${handCount === 1 ? "" : "s"}, ${footCount} foothold${footCount === 1 ? "" : "s"} detected.`;
});

downloadJsonBtn.addEventListener("click", () => {
  if (!currentPoseData) return;
  const blob = new Blob([JSON.stringify(currentPoseData, null, 2)], { type: "application/json" });
  triggerDownload(blob, "pose_data.json");
});

async function recordOverlayVideoOnce(mimeType: string): Promise<Blob> {
  if (!activeVideoEl) throw new Error("No playable video available to record from.");

  // Automatic-capture mode, sampled at the video's own fps: the video
  // plays through in real time below, which is exactly the steady cadence
  // automatic sampling is designed for.
  const fps = currentPoseData!.video.fps;
  const stream: MediaStream = (canvas as any).captureStream(fps);
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  const recordingStopped = new Promise<void>((resolve) => {
    recorder.onstop = () => resolve();
  });

  const video = activeVideoEl;
  video.loop = false;
  video.currentTime = 0;
  await new Promise<void>((resolve) => {
    video.addEventListener("seeked", () => resolve(), { once: true });
  });

  recorder.start();
  await new Promise<void>((resolve) => {
    let rafId = 0;
    const tick = () => {
      drawCurrentFrame();
      rafId = requestAnimationFrame(tick);
    };
    const onTimeUpdate = () => {
      if (!video.duration) return;
      const pct = Math.round((video.currentTime / video.duration) * 100);
      downloadVideoBtn.textContent = `Recording overlay video… ${pct}%`;
    };
    const onEnded = () => {
      cancelAnimationFrame(rafId);
      video.removeEventListener("timeupdate", onTimeUpdate);
      resolve();
    };
    video.addEventListener("ended", onEnded, { once: true });
    video.addEventListener("timeupdate", onTimeUpdate);
    rafId = requestAnimationFrame(tick);
    video.play();
  });
  recorder.stop();
  await recordingStopped;
  return new Blob(chunks, { type: mimeType });
}

async function downloadOverlayVideo() {
  if (!currentPoseData || isBusy) return;
  isBusy = true;
  stopLoopPlayback();
  downloadJsonBtn.disabled = true;
  downloadVideoBtn.disabled = true;

  try {
    if (typeof (canvas as any).captureStream !== "function" || typeof MediaRecorder === "undefined") {
      setStatus("Your browser doesn't support recording the canvas to video (no captureStream/MediaRecorder). Try a recent Chrome, Edge, or Firefox.");
      return;
    }

    const mimeCandidates = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"];
    const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported?.(m)) ?? "video/webm";

    // captureStream()+MediaRecorder occasionally produces a near-empty
    // recording under load — a real timing race in the browser's capture
    // pipeline, not something this code can prevent outright. A
    // few-hundred-byte file is just a container header with no frames, so
    // rather than risk silently handing over a corrupt download, validate
    // the size and retry before giving up.
    const MIN_VALID_BYTES = 2000;
    const MAX_ATTEMPTS = 3;
    let blob: Blob | null = null;
    let lastError: unknown = null;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      if (attempt > 1) {
        setStatus(`Recording came out empty — retrying (attempt ${attempt}/${MAX_ATTEMPTS})…`);
      }
      try {
        const candidate = await recordOverlayVideoOnce(mimeType);
        if (candidate.size >= MIN_VALID_BYTES) {
          blob = candidate;
          break;
        }
        lastError = new Error(`Recording produced only ${candidate.size} bytes.`);
      } catch (err) {
        lastError = err;
      }
    }

    if (!blob) {
      setStatus("Couldn't record the overlay video after a few tries — please try again.");
      console.error("downloadOverlayVideo: all attempts failed", lastError);
      return;
    }

    triggerDownload(blob, "climbing_pose_overlay.webm");
  } finally {
    downloadVideoBtn.textContent = DOWNLOAD_VIDEO_DEFAULT_LABEL;
    downloadJsonBtn.disabled = false;
    downloadVideoBtn.disabled = false;
    isBusy = false;
    startLoopPlayback();
  }
}

downloadVideoBtn.addEventListener("click", () => {
  downloadOverlayVideo();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

/**
 * Loads the bundled demo climb and runs it through the exact same pipeline
 * as a user-dropped file, so the page shows a real analysis immediately on
 * first load. Dropping/selecting a video afterwards goes through the normal
 * dropzone handlers above, which call handleFile() the same way and fully
 * reset all state — so switching to your own video "just works" with no
 * special-casing needed for the demo having run first.
 */
async function loadDefaultClimb() {
  try {
    const response = await fetch(DEFAULT_CLIMB_VIDEO_URL);
    if (!response.ok) throw new Error(`fetch failed: ${response.status}`);
    const blob = await response.blob();
    const file = new File([blob], "default-climb.mp4", { type: "video/mp4" });
    await handleFile(file, true, DEFAULT_CLIMB_FPS);
  } catch (err) {
    // Non-fatal: the page just falls back to its normal empty/upload state.
    console.error("Failed to auto-load the demo climb:", err);
  }
}

loadDefaultClimb();
