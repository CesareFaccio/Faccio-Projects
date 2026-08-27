// Parses an MP4/MOV container (via mp4box.js) into a plain, backend-agnostic
// description of its video track: codec config for WebCodecs' VideoDecoder,
// display rotation, true container frame rate, and every compressed sample.
//
// This intentionally does no decoding itself — see webcodecsDecode.ts for
// that. Keeping demuxing and decoding separate means the (cheap) parsed
// result can be decoded more than once (once for pose extraction, again
// later for looping preview playback / video export) without re-reading or
// re-parsing the file each time.

import {
  createFile,
  MultiBufferStream,
  MP4BoxBuffer,
  type ISOFile,
  type Track,
  type VisualSampleEntry,
} from "mp4box";

export interface DemuxedSample {
  data: Uint8Array;
  isKey: boolean;
  timestampUs: number;
  durationUs: number;
}

export interface DemuxedVideo {
  codec: string;
  codedWidth: number;
  codedHeight: number;
  description: Uint8Array;
  /** Rotation to apply when displaying a decoded (coded-orientation) frame. */
  rotationDegrees: 0 | 90 | 180 | 270;
  /** Dimensions after applying rotationDegrees (i.e. what the video actually looks like upright). */
  displayWidth: number;
  displayHeight: number;
  /** True average frame rate computed from the container's own sample count/timescale — not an estimate. */
  fps: number;
  samples: DemuxedSample[];
}

export class UnsupportedContainerError extends Error {}

/**
 * Reads the QuickTime/ISO track transformation matrix and classifies it as
 * one of the four cardinal rotations real-world camera recordings use.
 * Returns null for anything else (skew, non-90-multiple rotation, mirroring)
 * — the caller should treat that as "can't handle this file via WebCodecs".
 */
function classifyRotation(matrix: ArrayLike<number>): 0 | 90 | 180 | 270 | null {
  // Matrix is [a, b, u, c, d, v, x, y, w] in 16.16 fixed point for a/b/c/d.
  const FP = 0x10000;
  const a = Math.round(matrix[0] / FP);
  const b = Math.round(matrix[1] / FP);
  const c = Math.round(matrix[3] / FP);
  const d = Math.round(matrix[4] / FP);

  if (a === 1 && b === 0 && c === 0 && d === 1) return 0;
  if (a === 0 && b === 1 && c === -1 && d === 0) return 90;
  if (a === -1 && b === 0 && c === 0 && d === -1) return 180;
  if (a === 0 && b === -1 && c === 1 && d === 0) return 270;
  return null;
}

/** Extracts the raw hvcC/avcC config box bytes needed for VideoDecoderConfig.description. */
function getDescription(isoFile: ISOFile, track: Track): Uint8Array {
  const trak = isoFile.getTrackById(track.id);
  const entries = trak?.mdia?.minf?.stbl?.stsd?.entries ?? [];
  for (const entry of entries) {
    const visual = entry as VisualSampleEntry;
    const box = visual.avcC ?? visual.hvcC;
    if (box) {
      // MultiBufferStream (a DataStream subtype) defaults to big-endian,
      // which is what these ISO BMFF config boxes are written in.
      const stream = new MultiBufferStream();
      box.write(stream);
      // write() includes the box's own 8-byte header (size + fourcc); the
      // WebCodecs description wants just the payload after that.
      return new Uint8Array(stream.buffer, 8);
    }
  }
  throw new UnsupportedContainerError("No avcC/hvcC configuration box found on the video track.");
}

export function demuxMp4(file: File): Promise<DemuxedVideo> {
  return new Promise((resolve, reject) => {
    const isoFile = createFile();
    let failed: unknown = null;
    let trackMeta: {
      codec: string;
      codedWidth: number;
      codedHeight: number;
      description: Uint8Array;
      rotationDegrees: 0 | 90 | 180 | 270;
      fps: number;
    } | null = null;
    const samples: DemuxedSample[] = [];

    isoFile.onError = (module, message) => {
      failed = failed ?? new UnsupportedContainerError(`${module}: ${message}`);
    };

    isoFile.onReady = (info) => {
      try {
        const track = info.videoTracks[0];
        if (!track) throw new UnsupportedContainerError("No video track found.");

        const rotationDegrees = classifyRotation(track.matrix);
        if (rotationDegrees === null) {
          throw new UnsupportedContainerError("Video has a non-cardinal rotation/transform — can't handle via WebCodecs.");
        }

        const codedWidth = track.video?.width;
        const codedHeight = track.video?.height;
        if (!codedWidth || !codedHeight) {
          throw new UnsupportedContainerError("Could not determine coded video dimensions.");
        }

        const description = getDescription(isoFile, track);

        const durationSec = track.duration / track.timescale;
        const fps = durationSec > 0 ? track.nb_samples / durationSec : 0;
        if (!fps || !Number.isFinite(fps)) {
          throw new UnsupportedContainerError("Could not determine a valid frame rate from the container.");
        }

        trackMeta = { codec: track.codec, codedWidth, codedHeight, description, rotationDegrees, fps };

        isoFile.onSamples = (_id, _user, newSamples) => {
          for (const s of newSamples) {
            if (!s.data) continue;
            samples.push({
              data: s.data,
              isKey: s.is_sync,
              timestampUs: Math.round((s.cts / s.timescale) * 1e6),
              durationUs: Math.round((s.duration / s.timescale) * 1e6),
            });
          }
        };
        isoFile.setExtractionOptions(track.id, undefined, { nbSamples: Infinity });
        isoFile.start();
      } catch (err) {
        failed = failed ?? err;
      }
    };

    file
      .arrayBuffer()
      .then((buf) => {
        const mp4boxBuffer = MP4BoxBuffer.fromArrayBuffer(buf, 0);
        isoFile.appendBuffer(mp4boxBuffer, true);
        isoFile.flush();

        if (failed) {
          reject(failed instanceof Error ? failed : new UnsupportedContainerError(String(failed)));
          return;
        }
        if (!trackMeta) {
          reject(new UnsupportedContainerError("Container parsed but no ready video track info was produced."));
          return;
        }
        if (samples.length === 0) {
          reject(new UnsupportedContainerError("No samples extracted from the video track."));
          return;
        }
        samples.sort((a, b) => a.timestampUs - b.timestampUs);

        const { rotationDegrees, codedWidth, codedHeight } = trackMeta;
        const swapped = rotationDegrees === 90 || rotationDegrees === 270;
        resolve({
          ...trackMeta,
          displayWidth: swapped ? codedHeight : codedWidth,
          displayHeight: swapped ? codedWidth : codedHeight,
          samples,
        });
      })
      .catch((err) => reject(err instanceof Error ? err : new UnsupportedContainerError(String(err))));
  });
}
