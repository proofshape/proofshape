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

export type Provenance = "observed" | "unobserved" | "inferred";

export type Verdict = "Pass" | "Fail" | "Rescan" | "Unverifiable";

export interface Region {
  name: string;
  provenance: Provenance;
  deviation_mm?: number;
  contributes_to_verdict: boolean;
}

export interface VerdictResult {
  verdict: Verdict;
  declared_at: "intake" | "post_reconstruction";
  orientation_ambiguous: boolean;
  regions: Region[];
  reason_codes?: string[];
}

/**
 * C-22 temporary capture-side shape used only by the contract stub until A-07's D-031
 * phone-facing field/route is agreed. This is deliberately not added to the frozen OpenAPI
 * contract yet.
 */
export interface AgentMessageResult {
  message: string | null;
  fallback: boolean;
  fallback_reason?: string;
}
