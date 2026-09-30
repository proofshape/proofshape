// Hand-written from contracts/openapi.yaml (F-03, frozen). Keep these in sync with the spec
// by hand — there is no code generator wired up yet.

export interface Session {
  session_id: string;
  order_code: string;
  status: "capturing" | "finished";
}

export interface UploadFrameResult {
  accepted: boolean;
  frame_index: number;
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
  | "no_cad_uploaded";

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
