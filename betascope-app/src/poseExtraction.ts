import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { FrameEntry, PoseData } from "./types";
import { LANDMARK_NAMES } from "./types";
import { demuxMp4, UnsupportedContainerError, type DemuxedVideo } from "./mp4Demux";
import { checkWebCodecsSupport, decodeMp4Frames } from "./webcodecsDecode";

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

export interface ExtractionResult {
  data: PoseData;
  /**
   * "webcodecs": frames were decoded straight from the original container
   * (no re-encode, no browser <video> involved) via WebCodecs — this same
   * demuxed handle drives playback/export too, so neither ever touches a
   * re-encoded copy of the video.
   * "video-element": the classic <video>+seek path — used whenever the
   * browser can't decode this file's original codec via WebCodecs (e.g.
   * HEVC on most non-Mac platforms) or the container isn't MP4/MOV. The
   * video was already loaded into the returned element; reuse it for
   * playback rather than reloading the file.
   */
  backend: "webcodecs" | "video-element";
  videoElement?: HTMLVideoElement;
  demuxed?: DemuxedVideo;
}

// Self-hosted (see public/wasm, public/models) rather than CDN — keeps the
// live site working without depending on a third party's CDN uptime.
// BASE_URL is resolved relative to wherever this page actually gets
// deployed (e.g. /betascope/ under a GitHub Pages project subpath), so
// these must not be hardcoded as absolute root paths.
const WASM_BASE = `${import.meta.env.BASE_URL}wasm`;
const MODEL_PATH = `${import.meta.env.BASE_URL}models/pose_landmarker_full.task`;

async function createLandmarker(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>): Promise<PoseLandmarker> {
  const config = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: "VIDEO" as const,
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  // GPU delegate isn't available on every device/browser — fall back to CPU.
  return PoseLandmarker.createFromOptions(vision, config("GPU")).catch(() =>
    PoseLandmarker.createFromOptions(vision, config("CPU"))
  );
}

/**
 * Loads a video File into a hidden <video> element and waits for it to be
 * ready to play. Rejects with UnsupportedVideoError if the browser can't
 * decode it (e.g. HEVC on non-Safari browsers, and on browsers/videos the
 * WebCodecs path above also can't handle).
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
 * expose container fps directly, so this is the practical way to get it —
 * only needed for the <video>-element fallback path; the WebCodecs path
 * reads the container's true frame rate directly (see mp4Demux.ts).
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

function landmarksToFrameEntry(frame: number, timestampS: number, result: { landmarks: any[] }): FrameEntry {
  const detected = result.landmarks && result.landmarks.length > 0;
  return {
    frame,
    timestamp_s: Math.round(timestampS * 10000) / 10000,
    detected,
    landmarks: detected
      ? result.landmarks[0].map((lm: any, i: number) => ({
          name: LANDMARK_NAMES[i],
          x: Math.round(lm.x * 1e6) / 1e6,
          y: Math.round(lm.y * 1e6) / 1e6,
          z: Math.round(lm.z * 1e6) / 1e6,
          visibility: Math.round((lm.visibility ?? 0) * 1e4) / 1e4,
        }))
      : [],
  };
}

/**
 * Decodes the video straight from its original container via WebCodecs —
 * no <video> element, no re-encode step, ever. This is what closes the
 * accuracy gap against the Python reference pipeline: browsers can't play
 * HEVC (the default iPhone codec) in a <video> tag at all, so previously
 * every HEVC upload had to be re-encoded to a browser-playable format
 * first, and that re-encode measurably degrades pose landmark accuracy
 * (confirmed by isolating the effect: re-encoding alone, nothing else
 * different, shifted landmarks by ~15px on average vs. the original file).
 * Throws (falls back to extractPoseViaVideoElement) if the container isn't
 * MP4/MOV, or this browser can't decode its codec via WebCodecs.
 */
async function extractPoseViaWebCodecs(
  file: File,
  landmarker: PoseLandmarker,
  onProgress?: (p: ExtractionProgress) => void
): Promise<ExtractionResult> {
  const demuxed = await demuxMp4(file);
  const supported = await checkWebCodecsSupport(demuxed);
  if (!supported) {
    throw new UnsupportedContainerError("This browser can't decode this video's codec via WebCodecs.");
  }

  const totalFrames = demuxed.samples.length;
  const frames: FrameEntry[] = [];
  onProgress?.({ phase: "extracting", frame: 0, totalFrames });

  let lastTimestampMs = -1;
  await decodeMp4Frames(demuxed, {
    onFrame: (canvas, index, timestampUs) => {
      let timestampMs = Math.round(timestampUs / 1000);
      // MediaPipe requires strictly increasing integer ms timestamps.
      if (timestampMs <= lastTimestampMs) timestampMs = lastTimestampMs + 1;
      lastTimestampMs = timestampMs;

      const result = landmarker.detectForVideo(canvas, timestampMs);
      frames.push(landmarksToFrameEntry(index, timestampUs / 1e6, result));

      if (index % 5 === 0 || index === totalFrames - 1) {
        onProgress?.({ phase: "extracting", frame: index + 1, totalFrames });
      }
    },
  });

  landmarker.close();

  return {
    data: {
      video: {
        path: file.name,
        fps: demuxed.fps,
        width: demuxed.displayWidth,
        height: demuxed.displayHeight,
        total_frames: totalFrames,
      },
      landmarks: frames,
    },
    backend: "webcodecs",
    demuxed,
  };
}

/** The original <video>+seek extraction path — kept as the universal fallback. */
async function extractPoseViaVideoElement(
  file: File,
  landmarker: PoseLandmarker,
  onProgress?: (p: ExtractionProgress) => void
): Promise<ExtractionResult> {
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
    frames.push(landmarksToFrameEntry(frameIdx, t, result));

    if (frameIdx % 5 === 0 || frameIdx === totalFrames - 1) {
      onProgress?.({ phase: "extracting", frame: frameIdx + 1, totalFrames });
    }
  }

  landmarker.close();

  return {
    data: {
      video: { path: file.name, fps, width, height, total_frames: totalFrames },
      landmarks: frames,
    },
    backend: "video-element",
    videoElement: video,
  };
}

export async function extractPose(
  file: File,
  onProgress?: (p: ExtractionProgress) => void
): Promise<ExtractionResult> {
  onProgress?.({ phase: "loading-model" });
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  const landmarker = await createLandmarker(vision);

  try {
    return await extractPoseViaWebCodecs(file, landmarker, onProgress);
  } catch (webCodecsErr) {
    // Any failure here (unsupported container, unsupported codec/config on
    // this browser, a mid-decode error) falls back to the classic
    // <video>+seek path, which covers anything the browser's native video
    // pipeline can already play. A genuinely unplayable file (e.g. HEVC
    // that also can't be decoded via WebCodecs on this browser/platform)
    // surfaces as UnsupportedVideoError from that path, same as before
    // this feature existed — this only ever improves accuracy, never
    // removes support for a file that used to work.
    console.warn("WebCodecs extraction unavailable, falling back to <video>-based extraction:", webCodecsErr);
    return extractPoseViaVideoElement(file, landmarker, onProgress);
  }
}
