import { extractPose, UnsupportedVideoError } from "./poseExtraction";
import { CONNECTIONS } from "./types";
import type { PoseData } from "./types";

const fileInput = document.getElementById("file-input") as HTMLInputElement;
const dropzone = document.getElementById("dropzone") as HTMLDivElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const progressBar = document.getElementById("progress-bar") as HTMLDivElement;
const progressWrap = document.getElementById("progress-wrap") as HTMLDivElement;
const resultEl = document.getElementById("result") as HTMLDivElement;
const canvas = document.getElementById("preview-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const downloadBtn = document.getElementById("download-json") as HTMLButtonElement;
const previewVideo = document.getElementById("preview-video") as HTMLVideoElement;

let lastResult: PoseData | null = null;

function setStatus(text: string) {
  statusEl.textContent = text;
}

// Matches VISIBILITY_THRESHOLD in reconstruct_video.py / reconstruct_plus.py —
// low-confidence landmarks (occluded limbs, edge-of-frame, etc.) are skipped
// rather than drawn, which is what keeps the overlay legible instead of a
// tangle of low-confidence guesses.
const VISIBILITY_THRESHOLD = 0.5;

function drawSkeleton(entry: PoseData["landmarks"][number]) {
  ctx.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);
  if (!entry.detected) return;

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

async function handleFile(file: File) {
  resultEl.hidden = true;
  progressWrap.hidden = false;
  progressBar.style.width = "0%";
  downloadBtn.hidden = true;

  previewVideo.src = URL.createObjectURL(file);
  await new Promise<void>((resolve) => {
    if (previewVideo.readyState >= 1) { resolve(); return; }
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

    lastResult = data;
    (window as any).__betascopePoseData = data; // debugging convenience
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    const detectedCount = data.landmarks.filter((f) => f.detected).length;
    const detectionRate = ((detectedCount / data.landmarks.length) * 100).toFixed(1);

    canvas.width = data.video.width;
    canvas.height = data.video.height;
    const lastDetected = [...data.landmarks].reverse().find((f) => f.detected);
    if (lastDetected) {
      previewVideo.currentTime = lastDetected.timestamp_s;
      await new Promise((r) => (previewVideo.onseeked = r));
      drawSkeleton(lastDetected);
    }

    resultEl.hidden = false;
    downloadBtn.hidden = false;
    setStatus(
      `Processed ${data.landmarks.length} frames in ${elapsed}s — ${detectionRate}% detection rate ` +
        `(${data.video.width}x${data.video.height} @ ${data.video.fps.toFixed(2)}fps)`
    );
    progressBar.style.width = "100%";
  } catch (err) {
    if (err instanceof UnsupportedVideoError) {
      setStatus(err.message);
    } else {
      setStatus(`Something went wrong: ${(err as Error).message}`);
      console.error(err);
    }
    progressWrap.hidden = true;
  }
}

downloadBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pose_data.json";
  a.click();
  URL.revokeObjectURL(url);
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
