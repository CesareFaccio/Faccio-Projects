// Decodes a DemuxedVideo (see mp4Demux.ts) via the WebCodecs VideoDecoder,
// one frame at a time, drawing each rotation-corrected frame onto a reusable
// canvas. This is the single primitive both pose extraction and playback
// (looping preview / video export) are built on — decoding straight from
// the container's original compressed samples means neither path ever
// touches a re-encoded copy of the video.

import type { DemuxedVideo } from "./mp4Demux";

/** Feature-detects + config-checks whether this browser can actually decode this video via WebCodecs. */
export async function checkWebCodecsSupport(demuxed: DemuxedVideo): Promise<boolean> {
  if (typeof VideoDecoder === "undefined") return false;
  try {
    const support = await VideoDecoder.isConfigSupported({
      codec: demuxed.codec,
      codedWidth: demuxed.codedWidth,
      codedHeight: demuxed.codedHeight,
      description: demuxed.description,
    });
    return support.supported === true;
  } catch {
    return false;
  }
}

export interface DecodePassOptions {
  /**
   * Called once per decoded frame, in presentation order, with a canvas
   * already drawn in correct display orientation. The canvas is reused
   * across calls — draw/read from it synchronously or copy what you need
   * before returning/resolving, since its contents change on the next frame.
   */
  onFrame: (canvas: OffscreenCanvas, index: number, timestampUs: number) => Promise<void> | void;
  /** Pace frames to roughly match real playback speed (for a looping preview). Omit/false to decode as fast as possible (extraction, export). */
  realtime?: boolean;
  signal?: AbortSignal;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function drawRotated(
  ctx: OffscreenCanvasRenderingContext2D,
  frame: VideoFrame,
  rotationDegrees: 0 | 90 | 180 | 270,
  displayWidth: number,
  displayHeight: number
) {
  ctx.save();
  try {
    switch (rotationDegrees) {
      case 0:
        ctx.drawImage(frame, 0, 0, displayWidth, displayHeight);
        break;
      case 90:
        ctx.translate(displayWidth, 0);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(frame, 0, 0, displayHeight, displayWidth);
        break;
      case 180:
        ctx.translate(displayWidth, displayHeight);
        ctx.rotate(Math.PI);
        ctx.drawImage(frame, 0, 0, displayWidth, displayHeight);
        break;
      case 270:
        ctx.translate(0, displayHeight);
        ctx.rotate(-Math.PI / 2);
        ctx.drawImage(frame, 0, 0, displayHeight, displayWidth);
        break;
    }
  } finally {
    ctx.restore();
  }
}

/**
 * Runs one full decode pass over every sample, front to back. Decodes one
 * frame at a time (never more than one VideoFrame outstanding) — simpler
 * and safer than pipelining, and decode is not the bottleneck here (pose
 * inference and/or realtime pacing dominate), so there's nothing to gain
 * from overlapping decode calls.
 */
export async function decodeMp4Frames(demuxed: DemuxedVideo, opts: DecodePassOptions): Promise<void> {
  const canvas = new OffscreenCanvas(demuxed.displayWidth, demuxed.displayHeight);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get a 2D context on an OffscreenCanvas.");

  let pending: { resolve: (frame: VideoFrame) => void; reject: (err: unknown) => void } | null = null;

  const decoder = new VideoDecoder({
    output: (frame) => {
      if (pending) {
        const { resolve } = pending;
        pending = null;
        resolve(frame);
      } else {
        // Shouldn't happen given the one-in-flight pacing below, but never leak a frame.
        frame.close();
      }
    },
    error: (err) => {
      if (pending) {
        const { reject } = pending;
        pending = null;
        reject(err);
      }
    },
  });

  try {
    decoder.configure({
      codec: demuxed.codec,
      codedWidth: demuxed.codedWidth,
      codedHeight: demuxed.codedHeight,
      description: demuxed.description,
    });

    for (let i = 0; i < demuxed.samples.length; i++) {
      if (opts.signal?.aborted) break;
      const sample = demuxed.samples[i];

      const framePromise = new Promise<VideoFrame>((resolve, reject) => {
        pending = { resolve, reject };
      });
      decoder.decode(
        new EncodedVideoChunk({
          type: sample.isKey ? "key" : "delta",
          timestamp: sample.timestampUs,
          duration: sample.durationUs,
          data: sample.data,
        })
      );

      const frame = await framePromise;
      try {
        drawRotated(ctx, frame, demuxed.rotationDegrees, demuxed.displayWidth, demuxed.displayHeight);
      } finally {
        frame.close();
      }

      const frameStart = performance.now();
      await opts.onFrame(canvas, i, sample.timestampUs);

      if (opts.realtime) {
        const targetMs = sample.durationUs / 1000;
        const waitMs = targetMs - (performance.now() - frameStart);
        if (waitMs > 0) await sleep(waitMs);
      }
    }
    await decoder.flush();
  } finally {
    if (decoder.state !== "closed") decoder.close();
  }
}
