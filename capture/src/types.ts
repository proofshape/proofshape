// Hand-written from contracts/openapi.yaml (F-03, frozen; v1.1 since A-11, D-056). Keep these
// in sync with the spec by hand — there is no code generator wired up yet.

export interface Session {
  session_id: string;
  order_code: string;
  status: "capturing" | "finished";
}

// v1.1 (A-11): the newest completed VLM check. It describes an earlier sampled frame, named by
// its own frame_index. `hint` is model output, so it is only ever shown as plain text.
export interface VlmFeedback {
  frame_index: number;
  hint: string;
  // Null for frame 0, the reference the other frames are compared against.
  same_part: boolean | null;
  hands_occluding: boolean;
  screen_photo: boolean;
  lighting_adequate: boolean;
}

export interface UploadFrameResult {
  accepted: boolean;
  frame_index: number;
  // Absent (never null) until a check of this session has completed.
  vlm_feedback?: VlmFeedback;
}

export type ErrorCode =
  | "board_not_detected"
  | "board_partially_occluded"
  | "insufficient_baseline"
  | "metric_alignment_failed"
  | "scale_disagreement"
  | "no_reference_detected"
  | "frame_rejected_blur"
  | "frame_rejected_exposure"
  | "frame_rejected_duplicate"
  | "vlm_veto"
  | "session_not_found"
  | "session_already_finished"
  | "too_few_frames"
  | "order_code_unknown"
  | "no_cad_uploaded"
  | "malformed_request"
  | "internal_error";

export interface ErrorDetail {
  code: ErrorCode;
  message: string;
  frame_indices?: number[];
}

export interface Gyro {
  alpha: number;
  beta: number;
  gamma: number;
}

export type ReferenceTier = "charuco_board" | "blank_sheet" | "credit_card" | "none";

export interface ScaleAgreement {
  sources_compared: Array<"board_pitch" | "caliper_reading" | "cad_registration_residual">;
  flagged: boolean;
}

export interface ReconstructionResult {
  status: "pending" | "complete" | "failed";
  glb_url?: string;
  reference_tier_used?: ReferenceTier;
  metric?: boolean;
  observed_fraction?: number;
  scale_agreement?: ScaleAgreement;
  stage_timings_s?: Record<string, number>;
  warnings?: string[];
}
