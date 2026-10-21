/** Shared layout metrics so the WebGL viewports and the HTML chrome never disagree. */

export const COMPACT_BREAKPOINT = 768;
export const DOCK_HEIGHT = 56;
export const EDGE_COMPACT = 12;
export const EDGE_REGULAR = 20;
export const PIP_BEZEL = 4;

export interface PipLayout {
  /** Inner (rendered) rectangle in CSS pixels, measured from the top-left corner */
  left: number;
  top: number;
  width: number;
  height: number;
  /** Same rectangle's Y origin measured from the bottom edge, as WebGL viewports expect */
  glY: number;
  compact: boolean;
}

export function isCompactWidth(width: number): boolean {
  return width < COMPACT_BREAKPOINT;
}

/**
 * Picture-in-Picture placement: top-right on regular screens, bottom-right above the dock on
 * compact screens (where the telemetry strip owns the top edge).
 */
export function computePipLayout(width: number, height: number): PipLayout {
  const compact = isCompactWidth(width);
  if (compact) {
    const w = Math.round(Math.min(168, width * 0.42));
    const h = Math.round(w * 0.625);
    const edge = EDGE_COMPACT + PIP_BEZEL;
    const bottom = EDGE_COMPACT + DOCK_HEIGHT + 12 + PIP_BEZEL;
    const top = height - bottom - h;
    return { left: width - edge - w, top, width: w, height: h, glY: bottom, compact };
  }
  const w = Math.round(Math.min(340, width * 0.25));
  const h = Math.round(w * 0.625);
  const edge = EDGE_REGULAR + PIP_BEZEL;
  return {
    left: width - edge - w,
    top: edge,
    width: w,
    height: h,
    glY: height - edge - h,
    compact,
  };
}
