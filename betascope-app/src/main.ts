import { extractPose, UnsupportedVideoError } from "./poseExtraction";
import { CONNECTIONS } from "./types";
import type { PoseData, FrameEntry } from "./types";

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
const previewVideo = document.getElementById("preview-video") as HTMLVideoElement;

// Matches VISIBILITY_THRESHOLD in reconstruct_video.py / reconstruct_plus.py —
// low-confidence landmarks (occluded limbs, edge-of-frame, etc.) are skipped
// rather than drawn, which is what keeps the overlay legible instead of a
// tangle of low-confidence guesses.
const VISIBILITY_THRESHOLD = 0.5;
const DOWNLOAD_VIDEO_DEFAULT_LABEL = "Download video with overlay";

let currentPoseData: PoseData | null = null;
let playbackRafId: number | null = null;
let isBusy = false; // extracting or recording — ignore new drops meanwhile

function setStatus(text: string) {
  statusEl.textContent = text;
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

/** Draws the video's current frame plus the pose overlay for that instant onto the canvas. */
function drawCurrentFrame() {
  ctx.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);
  const entry = frameAtTime(previewVideo.currentTime);
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

function loopTick() {
  drawCurrentFrame();
  playbackRafId = requestAnimationFrame(loopTick);
}

function startLoopPlayback() {
  stopLoopPlayback();
  previewVideo.loop = true;
  previewVideo.currentTime = 0;
  previewVideo.play().catch(() => {
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
  previewVideo.pause();
}

async function handleFile(file: File) {
  if (isBusy) return;
  isBusy = true;
  stopLoopPlayback();
  currentPoseData = null;

  resultEl.hidden = true;
  progressWrap.hidden = false;
  progressBar.style.width = "0%";
  downloadJsonBtn.hidden = true;
  downloadVideoBtn.hidden = true;
  downloadVideoBtn.disabled = false;
  downloadVideoBtn.textContent = DOWNLOAD_VIDEO_DEFAULT_LABEL;

  previewVideo.src = URL.createObjectURL(file);
  await new Promise<void>((resolve) => {
    if (previewVideo.readyState >= 1) {
      resolve();
      return;
    }
    previewVideo.addEventListener("loadedmetadata", () => resolve(), { once: true });
    previewVideo.load();
  });

  const startedAt = performance.now();

  try {
    const data = await extractPose(file, (p) => {
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

    currentPoseData = data;
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
    setStatus(
      `Processed ${data.landmarks.length} frames in ${elapsed}s — ${detectionRate}% detection rate ` +
        `(${data.video.width}x${data.video.height} @ ${data.video.fps.toFixed(2)}fps)`
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
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pose_data.json";
  a.click();
  URL.revokeObjectURL(url);
});

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

    const fps = currentPoseData.video.fps;
    const stream: MediaStream = (canvas as any).captureStream(fps);
    // vp8 first: some Chromium builds' software vp9 encoder silently
    // produces a near-empty recording when fed a canvas stream sourced
    // from video content (observed in headless/sandboxed testing) even
    // though isTypeSupported() reports vp9 as fine. vp8 has proven
    // reliable for this canvas-recording use case, so it's preferred over
    // vp9's better compression here.
    const mimeCandidates = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"];
    const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported?.(m)) ?? "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    const recordingStopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });

    previewVideo.loop = false;
    previewVideo.currentTime = 0;
    await new Promise<void>((resolve) => {
      previewVideo.addEventListener("seeked", () => resolve(), { once: true });
    });

    let rafId = 0;
    const tick = () => {
      drawCurrentFrame();
      rafId = requestAnimationFrame(tick);
    };

    const onEnded = () => {
      cancelAnimationFrame(rafId);
      recorder.stop();
    };
    const onTimeUpdate = () => {
      if (!previewVideo.duration) return;
      const pct = Math.round((previewVideo.currentTime / previewVideo.duration) * 100);
      downloadVideoBtn.textContent = `Recording overlay video… ${pct}%`;
    };
    previewVideo.addEventListener("ended", onEnded, { once: true });
    previewVideo.addEventListener("timeupdate", onTimeUpdate);

    recorder.start();
    rafId = requestAnimationFrame(tick);
    await previewVideo.play();

    await recordingStopped;
    previewVideo.removeEventListener("timeupdate", onTimeUpdate);

    const blob = new Blob(chunks, { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "climbing_pose_overlay.webm";
    a.click();
    URL.revokeObjectURL(url);
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
