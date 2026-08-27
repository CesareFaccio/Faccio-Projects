import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { FrameEntry, PoseData } from "./types";
import { LANDMARK_NAMES } from "./types";

export interface ExtractionProgress {
  phase: "loading-model" | "loading-video" | "estimating-fps" | "extracting" | "done";
  frame?: number;
  totalFrames?: number;
}

export class UnsupportedVideoError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedVideoError";
  }
}

// Self-hosted (see public/wasm, public/models) rather than CDN — keeps the
// live site working without depending on a third party's CDN uptime.
// BASE_URL is resolved relative to wherever this page actually gets
// deployed (e.g. /betascope/ under a GitHub Pages project subpath), so
// these must not be hardcoded as absolute root paths.
const WASM_BASE = `${import.meta.env.BASE_URL}wasm`;
const MODEL_PATH = `${import.meta.env.BASE_URL}models/pose_landmarker_full.task`;

/**
 * Loads a video File into a hidden <video> element and waits for it to be
 * ready to play. Rejects with UnsupportedVideoError if the browser can't
 * decode it (e.g. HEVC on non-Safari browsers).
 */
function loadVideo(file: File): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = URL.createObjectURL(file);

    const onReady = () => {
      cleanup();
      resolve(video);
    };
    const onError = () => {
      cleanup();
      reject(
        new UnsupportedVideoError(
          "This browser couldn't play that video. If it was recorded on an iPhone, " +
            "try re-exporting or sharing it as an H.264 video (iOS: Settings > Camera > " +
            'Formats > "Most Compatible") and upload that instead.'
        )
      );
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("error", onError);
    };

    video.addEventListener("loadedmetadata", onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.load();
  });
}

/**
 * Empirically measures the video's frame rate by playing it briefly and
 * timing real presented frames via requestVideoFrameCallback. Browsers don't
 * expose container fps directly, so this is the practical way to get it.
 */
function estimateFps(video: HTMLVideoElement, sampleFrames = 20): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!("requestVideoFrameCallback" in video)) {
      reject(new Error("This browser doesn't support frame-accurate video processing (requestVideoFrameCallback)."));
      return;
    }

    const times: number[] = [];
    const onFrame = (_now: number, metadata: any) => {
      times.push(metadata.mediaTime);
      if (times.length >= sampleFrames) {
        video.pause();
        const deltas: number[] = [];
        for (let i = 1; i < times.length; i++) deltas.push(times[i] - times[i - 1]);
        deltas.sort((a, b) => a - b);
        const median = deltas[Math.floor(deltas.length / 2)];
        if (!median || median <= 0) {
          reject(new Error("Could not determine the video's frame rate."));
          return;
        }
        resolve(1 / median);
        return;
      }
      (video as any).requestVideoFrameCallback(onFrame);
    };

    (video as any).requestVideoFrameCallback(onFrame);
    video.currentTime = 0;
    video.play().catch(reject);
  });
}

function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.removeEventListener("seeked", onSeeked);
      clearTimeout(timer);
      resolve();
    };
    const onSeeked = () => finish();
    video.addEventListener("seeked", onSeeked);
    // Safety net: some seeks (e.g. to a time within the currently displayed
    // frame) never fire 'seeked' — don't stall the extraction loop forever.
    const timer = setTimeout(finish, 1500);
    video.currentTime = t;
  });
}

export async function extractPose(
  file: File,
  onProgress?: (p: ExtractionProgress) => void
): Promise<PoseData> {
  onProgress?.({ phase: "loading-model" });
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  const landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  }).catch(async () => {
    // GPU delegate isn't available on every device/browser — fall back to CPU.
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  });

  onProgress?.({ phase: "loading-video" });
  const video = await loadVideo(file);

  onProgress?.({ phase: "estimating-fps" });
  const fps = await estimateFps(video);
  video.currentTime = 0;

  const width = video.videoWidth;
  const height = video.videoHeight;
  const totalFrames = Math.max(1, Math.round(video.duration * fps));

  const frames: FrameEntry[] = [];
  onProgress?.({ phase: "extracting", frame: 0, totalFrames });

  for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    const t = frameIdx / fps;
    await seekTo(video, Math.min(t, video.duration));

    const timestampMs = Math.round(t * 1000);
    const result = landmarker.detectForVideo(video, timestampMs);
    const detected = result.landmarks && result.landmarks.length > 0;

    frames.push({
      frame: frameIdx,
      timestamp_s: Math.round(t * 10000) / 10000,
      detected,
      landmarks: detected
        ? result.landmarks[0].map((lm, i) => ({
            name: LANDMARK_NAMES[i],
            x: Math.round(lm.x * 1e6) / 1e6,
            y: Math.round(lm.y * 1e6) / 1e6,
            z: Math.round(lm.z * 1e6) / 1e6,
            visibility: Math.round((lm.visibility ?? 0) * 1e4) / 1e4,
          }))
        : [],
    });

    if (frameIdx % 5 === 0 || frameIdx === totalFrames - 1) {
      onProgress?.({ phase: "extracting", frame: frameIdx + 1, totalFrames });
    }
  }

  landmarker.close();
  URL.revokeObjectURL(video.src);
  onProgress?.({ phase: "done" });

  return {
    video: { path: file.name, fps, width, height, total_frames: totalFrames },
    landmarks: frames,
  };
}
