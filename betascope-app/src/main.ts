import { extractPose, UnsupportedVideoError } from "./poseExtraction";
import { decodeMp4Frames } from "./webcodecsDecode";
import { CONNECTIONS } from "./types";
import type { PoseData, FrameEntry } from "./types";
import type { DemuxedVideo } from "./mp4Demux";

const fileInput = document.getElementById("file-input") as HTMLInputElement;
const dropzone = document.getElementById("dropzone") as HTMLDivElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const progressBar = document.getElementById("progress-bar") as HTMLDivElement;
const progressWrap = document.getElementById("progress-wrap") as HTMLDivElement;
const resultEl = document.getElementById("result") as HTMLDivElement;
const canvas = document.getElementById("preview-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const downloadJsonBtn = document.getElementById("download-json") as HTMLButtonElement;
const downloadVideoBtn = document.getElementById("download-video") as HTMLButtonElement;

// Matches VISIBILITY_THRESHOLD in reconstruct_video.py / reconstruct_plus.py —
// low-confidence landmarks (occluded limbs, edge-of-frame, etc.) are skipped
// rather than drawn, which is what keeps the overlay legible instead of a
// tangle of low-confidence guesses.
const VISIBILITY_THRESHOLD = 0.5;
const DOWNLOAD_VIDEO_DEFAULT_LABEL = "Download video with overlay";

let currentPoseData: PoseData | null = null;
let playbackRafId: number | null = null;
let isBusy = false; // extracting or recording — ignore new drops meanwhile

// Which extraction backend produced currentPoseData, and the handle needed
// to redraw frames for playback/export from that same backend:
//  - "video-element": activeVideoEl is a real <video>, already loaded.
//  - "webcodecs": currentDemuxed decodes straight from the original file —
//    no <video> element or re-encoded copy involved anywhere in this path.
let currentBackend: "video-element" | "webcodecs" | null = null;
let activeVideoEl: HTMLVideoElement | null = null;
let currentDemuxed: DemuxedVideo | null = null;
let playbackAbortController: AbortController | null = null;

function setStatus(text: string) {
  statusEl.textContent = text;
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
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

/** Draws the pose overlay for one frame onto the canvas — assumes the video frame itself is already drawn. */
function drawOverlayForFrame(entry: FrameEntry | null) {
  if (!entry || !entry.detected) return;

  const visible = entry.landmarks.map((lm) => lm.visibility >= VISIBILITY_THRESHOLD);

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#5eead4";
  for (const [a, b] of CONNECTIONS) {
    if (!visible[a] || !visible[b]) continue;
    const la = entry.landmarks[a], lb = entry.landmarks[b];
    ctx.beginPath();
    ctx.moveTo(la.x * canvas.width, la.y * canvas.height);
    ctx.lineTo(lb.x * canvas.width, lb.y * canvas.height);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffea00";
  entry.landmarks.forEach((lm, i) => {
    if (!visible[i]) return;
    ctx.beginPath();
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
    ctx.fill();
  });
}

/** video-element backend only: draws the video's current frame plus its overlay onto the canvas. */
function drawCurrentFrame() {
  if (!activeVideoEl) return;
  ctx.drawImage(activeVideoEl, 0, 0, canvas.width, canvas.height);
  drawOverlayForFrame(frameAtTime(activeVideoEl.currentTime));
}

function loopTick() {
  drawCurrentFrame();
  playbackRafId = requestAnimationFrame(loopTick);
}

/**
 * webcodecs backend only: there's no <video> element to .play(), so looping
 * playback is implemented by repeatedly re-decoding the (already-parsed,
 * cheap-to-redecode) original file, pacing frames to real time. Each frame
 * comes with an exact index, so no nearest-time lookup is needed — unlike
 * the video-element path's frameAtTime().
 */
async function runWebCodecsPlaybackLoop(demuxed: DemuxedVideo, signal: AbortSignal) {
  while (!signal.aborted) {
    try {
      await decodeMp4Frames(demuxed, {
        realtime: true,
        signal,
        onFrame: (sourceCanvas, index) => {
          ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
          drawOverlayForFrame(currentPoseData?.landmarks[index] ?? null);
        },
      });
    } catch (err) {
      if (!signal.aborted) console.error("Playback loop error:", err);
      break;
    }
  }
}

function startLoopPlayback() {
  stopLoopPlayback();
  if (currentBackend === "webcodecs" && currentDemuxed) {
    playbackAbortController = new AbortController();
    void runWebCodecsPlaybackLoop(currentDemuxed, playbackAbortController.signal);
  } else if (activeVideoEl) {
    activeVideoEl.loop = true;
    activeVideoEl.currentTime = 0;
    activeVideoEl.play().catch(() => {
      // Autoplay can be blocked in some contexts; the still frame from the
      // last progress tick stays visible, which is a fine fallback.
    });
    playbackRafId = requestAnimationFrame(loopTick);
  }
}

function stopLoopPlayback() {
  if (playbackRafId !== null) {
    cancelAnimationFrame(playbackRafId);
    playbackRafId = null;
  }
  if (playbackAbortController) {
    playbackAbortController.abort();
    playbackAbortController = null;
  }
  activeVideoEl?.pause();
}

async function handleFile(file: File) {
  if (isBusy) return;
  isBusy = true;
  stopLoopPlayback();
  currentPoseData = null;
  currentBackend = null;
  activeVideoEl = null;
  currentDemuxed = null;

  resultEl.hidden = true;
  progressWrap.hidden = false;
  progressBar.style.width = "0%";
  downloadJsonBtn.hidden = true;
  downloadVideoBtn.hidden = true;
  downloadVideoBtn.disabled = false;
  downloadVideoBtn.textContent = DOWNLOAD_VIDEO_DEFAULT_LABEL;
  setStatus("");

  const startedAt = performance.now();

  try {
    const result = await extractPose(file, (p) => {
      if (p.phase === "loading-model") setStatus("Loading pose model…");
      else if (p.phase === "loading-video") setStatus("Loading video…");
      else if (p.phase === "estimating-fps") setStatus("Reading video frame rate…");
      else if (p.phase === "extracting" && p.totalFrames) {
        const pct = Math.round(((p.frame ?? 0) / p.totalFrames) * 100);
        progressBar.style.width = `${pct}%`;
        setStatus(`Extracting pose: frame ${p.frame}/${p.totalFrames} (${pct}%)`);
      } else if (p.phase === "done") {
        setStatus("Done.");
      }
    });

    const data = result.data;
    currentPoseData = data;
    currentBackend = result.backend;
    activeVideoEl = result.videoElement ?? null;
    currentDemuxed = result.demuxed ?? null;
    (window as any).__betascopePoseData = data; // debugging convenience

    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    const detectedCount = data.landmarks.filter((f) => f.detected).length;
    const detectionRate = ((detectedCount / data.landmarks.length) * 100).toFixed(1);

    canvas.width = data.video.width;
    canvas.height = data.video.height;

    resultEl.hidden = false;
    downloadJsonBtn.hidden = false;
    downloadVideoBtn.hidden = false;
    progressBar.style.width = "100%";
    const backendNote =
      result.backend === "webcodecs" ? " · decoded directly from the original file (no re-encoding)" : "";
    setStatus(
      `Processed ${data.landmarks.length} frames in ${elapsed}s — ${detectionRate}% detection rate ` +
        `(${data.video.width}x${data.video.height} @ ${data.video.fps.toFixed(2)}fps)${backendNote}`
    );

    startLoopPlayback();
  } catch (err) {
    if (err instanceof UnsupportedVideoError) {
      setStatus(err.message);
    } else {
      setStatus(`Something went wrong: ${(err as Error).message}`);
      console.error(err);
    }
    progressWrap.hidden = true;
  } finally {
    isBusy = false;
  }
}

downloadJsonBtn.addEventListener("click", () => {
  if (!currentPoseData) return;
  const blob = new Blob([JSON.stringify(currentPoseData, null, 2)], { type: "application/json" });
  triggerDownload(blob, "pose_data.json");
});

async function recordOverlayVideoOnce(mimeType: string): Promise<Blob> {
  if (currentBackend === "webcodecs" && currentDemuxed) {
    // Manual-capture mode (0 = no automatic sampling): decode isn't paced
    // to real time here, so we push exactly one frame per decoded frame
    // via track.requestFrame(). Testing found this needs a real pause
    // between calls — pushing frames back-to-back with no yield starves
    // the encoder and produces a near-empty recording (the same failure
    // signature the automatic-sampling path below can also hit under
    // load — see the retry wrapper in downloadOverlayVideo), so this path
    // is deliberately paced rather than run flat out.
    const stream: MediaStream = (canvas as any).captureStream(0);
    const track = stream.getVideoTracks()[0] as MediaStreamTrack & { requestFrame?: () => void };
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    const recordingStopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });

    recorder.start();
    const totalFrames = currentPoseData!.landmarks.length;
    await decodeMp4Frames(currentDemuxed, {
      onFrame: async (sourceCanvas, index) => {
        ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
        drawOverlayForFrame(currentPoseData!.landmarks[index] ?? null);
        track.requestFrame?.();
        const pct = Math.round(((index + 1) / totalFrames) * 100);
        downloadVideoBtn.textContent = `Recording overlay video… ${pct}%`;
        await new Promise((resolve) => setTimeout(resolve, 20));
      },
    });
    recorder.stop();
    await recordingStopped;
    return new Blob(chunks, { type: mimeType });
  }

  if (activeVideoEl) {
    // Automatic-capture mode, sampled at the video's own fps: the video
    // plays through in real time below, which is exactly the steady
    // cadence automatic sampling is designed for.
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

  throw new Error("No playable source available to record from.");
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
    // recording under load (observed in testing on both the manual- and
    // automatic-sampling paths) — a real timing race in the browser's
    // capture pipeline, not something this code can prevent outright. A
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
