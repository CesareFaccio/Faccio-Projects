// forces.ts
// TypeScript port of stats1.py: estimates the fraction of body weight
// carried by each extremity (hands/feet) at every frame via a static-
// equilibrium minimum-norm force distribution, then derives limb-segment
// axial forces and knee/elbow joint angles from the pose geometry. Mirrors
// the Python script's algorithm, constants, and rounding.

import type { FrameEntry, PoseData } from "./types";
import type { AnalysisResult, Hold } from "./analysis";

const BODY_WEIGHT_KG = 70.0;
const UPWARD: [number, number] = [0.0, -1.0];

export const EXTREMITY_NAMES = ["left_hand", "right_hand", "left_foot", "right_foot"] as const;
export type ExtremityName = (typeof EXTREMITY_NAMES)[number];

interface LimbSegmentDef {
  name: string;
  distal: string;
  proximal: string;
  ext: ExtremityName;
}
export const LIMB_SEGMENTS: LimbSegmentDef[] = [
  { name: "left_lower_leg", distal: "LEFT_ANKLE", proximal: "LEFT_KNEE", ext: "left_foot" },
  { name: "left_upper_leg", distal: "LEFT_KNEE", proximal: "LEFT_HIP", ext: "left_foot" },
  { name: "right_lower_leg", distal: "RIGHT_ANKLE", proximal: "RIGHT_KNEE", ext: "right_foot" },
  { name: "right_upper_leg", distal: "RIGHT_KNEE", proximal: "RIGHT_HIP", ext: "right_foot" },
  { name: "left_forearm", distal: "LEFT_WRIST", proximal: "LEFT_ELBOW", ext: "left_hand" },
  { name: "left_upper_arm", distal: "LEFT_ELBOW", proximal: "LEFT_SHOULDER", ext: "left_hand" },
  { name: "right_forearm", distal: "RIGHT_WRIST", proximal: "RIGHT_ELBOW", ext: "right_hand" },
  { name: "right_upper_arm", distal: "RIGHT_ELBOW", proximal: "RIGHT_SHOULDER", ext: "right_hand" },
];

interface JointDef {
  name: string;
  distal: string;
  joint: string;
  proximal: string;
}
export const JOINT_DEFS: JointDef[] = [
  { name: "left_knee", distal: "LEFT_ANKLE", joint: "LEFT_KNEE", proximal: "LEFT_HIP" },
  { name: "right_knee", distal: "RIGHT_ANKLE", joint: "RIGHT_KNEE", proximal: "RIGHT_HIP" },
  { name: "left_elbow", distal: "LEFT_WRIST", joint: "LEFT_ELBOW", proximal: "LEFT_SHOULDER" },
  { name: "right_elbow", distal: "RIGHT_WRIST", joint: "RIGHT_ELBOW", proximal: "RIGHT_SHOULDER" },
];

export interface ExtremityForce {
  active: boolean;
  axial_kg: number | null;
  axial_pct: number | null;
  vertical_kg: number | null;
  vertical_pct: number | null;
}
export interface SegmentForce {
  axial_kg: number;
  axial_pct: number;
  angle_deg: number;
}
export interface WeightFrameEntry {
  frame: number;
  timestamp_s: number;
  left_hand: ExtremityForce;
  right_hand: ExtremityForce;
  left_foot: ExtremityForce;
  right_foot: ExtremityForce;
  limb_segments: Record<string, SegmentForce | null>;
  joint_angles: Record<string, number | null>;
}
export interface WeightResult {
  body_weight_kg: number;
  frames_with_contact: number;
  per_frame: WeightFrameEntry[]; // sparse: only frames with a detected CoM AND >=1 active hold
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}
function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
function round1(v: number): number {
  return Math.round(v * 10) / 10;
}
function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}
function round6(v: number): number {
  return Math.round(v * 1000000) / 1000000;
}
function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
const RAD2DEG = 180 / Math.PI;

/**
 * Moore-Penrose minimum-norm least-squares solve of D @ f = UPWARD, where D
 * is 2xN (N = number of active contacts). Matches numpy.linalg.lstsq's
 * result for a full-rank D: the "full column rank" formula for N=1
 * (over-determined, 2 equations / 1 unknown) and the "full row rank" formula
 * for N>=2 (exactly/under-determined).
 */
function distributeWeight(
  comXY: [number, number],
  contacts: Record<string, [number, number]>
): Record<string, { axial: number; vertical: number }> {
  const names = Object.keys(contacts);
  const N = names.length;
  if (N === 0) return {};

  const Dx = new Array(N).fill(0);
  const Dy = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    const [cx, cy] = contacts[names[i]];
    const vx = cx - comXY[0];
    const vy = cy - comXY[1];
    const norm = Math.sqrt(vx * vx + vy * vy);
    if (norm > 1e-6) {
      let s = Math.sign(comXY[1] - cy);
      if (s === 0) s = 1.0;
      Dx[i] = (s * vx) / norm;
      Dy[i] = (s * vy) / norm;
    }
  }

  let f: number[];
  if (N === 1) {
    const dtd = Dx[0] * Dx[0] + Dy[0] * Dy[0];
    const dtb = Dx[0] * UPWARD[0] + Dy[0] * UPWARD[1];
    f = [dtd > 1e-12 ? dtb / dtd : 0];
  } else {
    let g00 = 0,
      g01 = 0,
      g11 = 0;
    for (let i = 0; i < N; i++) {
      g00 += Dx[i] * Dx[i];
      g01 += Dx[i] * Dy[i];
      g11 += Dy[i] * Dy[i];
    }
    const det = g00 * g11 - g01 * g01;
    let gi00 = 0,
      gi01 = 0,
      gi11 = 0;
    if (Math.abs(det) > 1e-12) {
      gi00 = g11 / det;
      gi01 = -g01 / det;
      gi11 = g00 / det;
    }
    const z0 = gi00 * UPWARD[0] + gi01 * UPWARD[1];
    const z1 = gi01 * UPWARD[0] + gi11 * UPWARD[1];
    f = new Array(N);
    for (let i = 0; i < N; i++) f[i] = Dx[i] * z0 + Dy[i] * z1;
  }

  const result: Record<string, { axial: number; vertical: number }> = {};
  for (let i = 0; i < N; i++) {
    const axial = f[i];
    let vertical = f[i] * Math.abs(Dy[i]);
    if (f[i] < 0) vertical = -Math.abs(vertical);
    result[names[i]] = { axial, vertical };
  }
  return result;
}

function getLmMap(entry: FrameEntry, minVis: number): Record<string, [number, number]> {
  const map: Record<string, [number, number]> = {};
  if (!entry.detected || !entry.landmarks || entry.landmarks.length === 0) return map;
  for (const lm of entry.landmarks) {
    if (lm.visibility >= minVis) map[lm.name] = [lm.x, lm.y];
  }
  return map;
}

interface SegResult {
  axial_frac: number | null;
  angle_deg: number;
}

function computeLimbForces(
  lmMap: Record<string, [number, number]>,
  vertFracs: Partial<Record<ExtremityName, number>>
): { segments: Record<string, SegResult | null>; joints: Record<string, number | null> } {
  const segments: Record<string, SegResult | null> = {};
  for (const seg of LIMB_SEGMENTS) {
    const d = lmMap[seg.distal];
    const p = lmMap[seg.proximal];
    if (!d || !p) {
      segments[seg.name] = null;
      continue;
    }
    const vertFrac = vertFracs[seg.ext];
    if (vertFrac === undefined) {
      segments[seg.name] = null;
      continue;
    }
    const vecx = p[0] - d[0];
    const vecy = p[1] - d[1];
    const segLen = Math.sqrt(vecx * vecx + vecy * vecy);
    if (segLen < 1e-6) {
      segments[seg.name] = null;
      continue;
    }
    const vertExtent = Math.abs(vecy);
    if (vertExtent < segLen * 0.01) {
      segments[seg.name] = { axial_frac: null, angle_deg: round1(Math.acos(0.01) * RAD2DEG) };
      continue;
    }
    const cosTheta = vertExtent / segLen;
    const angleDeg = Math.acos(clamp(cosTheta, 0, 1)) * RAD2DEG;
    const axialFrac = vertFrac / cosTheta;
    segments[seg.name] = { axial_frac: round6(axialFrac), angle_deg: round2(angleDeg) };
  }

  const joints: Record<string, number | null> = {};
  for (const j of JOINT_DEFS) {
    const d = lmMap[j.distal];
    const jp = lmMap[j.joint];
    const p = lmMap[j.proximal];
    if (!d || !jp || !p) {
      joints[j.name] = null;
      continue;
    }
    const v1x = d[0] - jp[0],
      v1y = d[1] - jp[1];
    const v2x = p[0] - jp[0],
      v2y = p[1] - jp[1];
    const n1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const n2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (n1 < 1e-6 || n2 < 1e-6) {
      joints[j.name] = null;
      continue;
    }
    const cosA = (v1x / n1) * (v2x / n2) + (v1y / n1) * (v2y / n2);
    joints[j.name] = round2(Math.acos(clamp(cosA, -1, 1)) * RAD2DEG);
  }

  return { segments, joints };
}

function loadActiveHoldsByFrame(
  handholds: Hold[],
  footholds: Hold[],
  totalFrames: number
): Record<string, [number, number]>[] {
  const active: Record<string, [number, number]>[] = Array.from({ length: totalFrames }, () => ({}));
  const kindSideToName: Record<string, ExtremityName> = {
    "hand:left": "left_hand",
    "hand:right": "right_hand",
    "foot:left": "left_foot",
    "foot:right": "right_foot",
  };
  for (const hold of [...handholds, ...footholds]) {
    const name = kindSideToName[`${hold.kind}:${hold.side}`];
    const end = Math.min(hold.end_frame + 1, totalFrames);
    for (let fi = hold.start_frame; fi < end; fi++) {
      active[fi][name] = [hold.x, hold.y];
    }
  }
  return active;
}

/** Runs the full stats1.py weight-distribution/limb-force/joint-angle pipeline. */
export function computeWeightDistribution(poseData: PoseData, analysis: AnalysisResult): WeightResult {
  const { fps, total_frames } = poseData.video;
  const frames = poseData.landmarks;

  const comByFrame = new Map<number, [number, number]>();
  for (const c of analysis.com) {
    if (c.x !== null && c.y !== null) comByFrame.set(c.frame, [c.x, c.y]);
  }

  const activeHoldsByFrame = loadActiveHoldsByFrame(analysis.handholds, analysis.footholds, total_frames);

  const framesLm: Record<string, [number, number]>[] = new Array(total_frames);
  for (const entry of frames) {
    if (entry.frame < total_frames) framesLm[entry.frame] = getLmMap(entry, 0.3);
  }

  let framesWithAnyContact = 0;
  const perFrame: WeightFrameEntry[] = [];

  for (let fi = 0; fi < total_frames; fi++) {
    const com = comByFrame.get(fi);
    if (!com) continue;
    const activeHolds = activeHoldsByFrame[fi];
    if (!activeHolds || Object.keys(activeHolds).length === 0) continue;

    const forces = distributeWeight(com, activeHolds);
    if (Object.keys(forces).length === 0) continue;

    framesWithAnyContact++;

    const extremities: Record<ExtremityName, ExtremityForce> = {} as any;
    for (const name of EXTREMITY_NAMES) {
      const fr = forces[name];
      if (fr) {
        extremities[name] = {
          active: true,
          axial_kg: round3(fr.axial * BODY_WEIGHT_KG),
          axial_pct: round3(fr.axial * 100),
          vertical_kg: round3(fr.vertical * BODY_WEIGHT_KG),
          vertical_pct: round3(fr.vertical * 100),
        };
      } else {
        extremities[name] = { active: false, axial_kg: null, axial_pct: null, vertical_kg: null, vertical_pct: null };
      }
    }

    const vertFracs: Partial<Record<ExtremityName, number>> = {};
    for (const name of EXTREMITY_NAMES) {
      const fr = forces[name];
      if (fr) vertFracs[name] = fr.vertical;
    }

    const lmMap = framesLm[fi] || {};
    const { segments, joints } = computeLimbForces(lmMap, vertFracs);

    const limbSegments: Record<string, SegmentForce | null> = {};
    for (const seg of LIMB_SEGMENTS) {
      const s = segments[seg.name];
      if (s && s.axial_frac !== null) {
        limbSegments[seg.name] = {
          axial_kg: round3(s.axial_frac * BODY_WEIGHT_KG),
          axial_pct: round3(s.axial_frac * 100),
          angle_deg: s.angle_deg,
        };
      } else {
        limbSegments[seg.name] = null;
      }
    }

    const jointAngles: Record<string, number | null> = {};
    for (const j of JOINT_DEFS) jointAngles[j.name] = joints[j.name] ?? null;

    perFrame.push({
      frame: fi,
      timestamp_s: round4(fi / fps),
      left_hand: extremities.left_hand,
      right_hand: extremities.right_hand,
      left_foot: extremities.left_foot,
      right_foot: extremities.right_foot,
      limb_segments: limbSegments,
      joint_angles: jointAngles,
    });
  }

  return { body_weight_kg: BODY_WEIGHT_KG, frames_with_contact: framesWithAnyContact, per_frame: perFrame };
}
