// Mirrors the JSON schema produced by extract_pose.py / extract_pose_RTMPose.py,
// so downstream code (and validation against the existing pose_data.json) works
// the same way regardless of whether the pose came from Python or the browser.

export interface Landmark {
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface FrameEntry {
  frame: number;
  timestamp_s: number;
  detected: boolean;
  landmarks: Landmark[];
}

export interface VideoInfo {
  path: string;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
}

export interface PoseData {
  video: VideoInfo;
  landmarks: FrameEntry[];
}

// Same 33-landmark topology/order as MediaPipe Pose (and the Python pipeline).
export const LANDMARK_NAMES = [
  "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
  "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
  "LEFT_EAR", "RIGHT_EAR",
  "MOUTH_LEFT", "MOUTH_RIGHT",
  "LEFT_SHOULDER", "RIGHT_SHOULDER",
  "LEFT_ELBOW", "RIGHT_ELBOW",
  "LEFT_WRIST", "RIGHT_WRIST",
  "LEFT_PINKY", "RIGHT_PINKY",
  "LEFT_INDEX", "RIGHT_INDEX",
  "LEFT_THUMB", "RIGHT_THUMB",
  "LEFT_HIP", "RIGHT_HIP",
  "LEFT_KNEE", "RIGHT_KNEE",
  "LEFT_ANKLE", "RIGHT_ANKLE",
  "LEFT_HEEL", "RIGHT_HEEL",
  "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
] as const;

export const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];
