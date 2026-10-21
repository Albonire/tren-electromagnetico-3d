import * as THREE from 'three';

export interface TrackFrame {
  u: number;
  position: THREE.Vector3;
  tangent: THREE.Vector3;
  normal: THREE.Vector3;
  binormal: THREE.Vector3;
  inCoil: boolean;
  curvature: number;
}

export interface ElectromagneticTrainState {
  /** Normalized position along the closed track loop [0, 1) */
  u: number;
  /** Direction of travel: +1 (standard forward) or -1 (reversed battery polarity) */
  direction: 1 | -1;
  /** Distance traveled in meters */
  distanceMeters: number;
  /** Instantaneous linear speed magnitude in world units/s */
  speed: number;
  /** Instantaneous tangential acceleration in world units/s^2 */
  acceleration: number;
  /** Electric current in Amperes flowing through the bridged coil turns */
  currentAmps: number;
  /** Axial solenoid magnetic field B inside bridged turns (mT) */
  magneticFieldmT: number;
  /** Net electromagnetic Lorentz/Laplace propulsion force (N) */
  propulsionForceN: number;
  /** Sliding contact spark intensity [0..1] at magnet-to-coil brushes */
  sparkIntensity: number;
  /** Whether the battery train is currently inside the wire coil (vs crossing the open gap) */
  inCoil: boolean;
  /** Normalized [0..1] fraction of the train currently coupled inside the coil */
  coilCoupling: number;
  /** Range of helical turn indices currently energized between rear and front magnets */
  activeTurnRange: [number, number];
  /** Total number of turns in the coil */
  totalTurns: number;
  /** Lap counter */
  lapCount: number;
}

/**
 * Total physical length of the closed loop track in world units (1 unit = 10 cm, total ~9.2 units = 92 cm)
 */
export const TRACK_TOTAL_LENGTH = 9.2;

/**
 * Normalized parameter bounds where the wire coil starts and ends.
 * In [0, COIL_START_U) and (COIL_END_U, 1.0], there is the open gap at the center-top of the heart/kidney loop,
 * matching the reference photo where the battery train bridges the two open ends of the coil.
 */
export const COIL_START_U = 0.025;
export const COIL_END_U = 0.975;

/**
 * Total helical turns along the coil section
 */
export const TOTAL_COIL_TURNS = 340;

/**
 * Train geometry dimensions in world units (1 unit = 10 cm):
 * AA battery hull + front/rear 4-disc Neodymium magnet stacks:
 * - Battery hull length: 0.42 units (4.2 cm)
 * - Front & rear 4-disc Neodymium stacks centered at 0.278 units from train center
 * - Total train length: ~0.68 units
 * - Coil centerline radius: 0.135 units, wire radius: 0.0125 units -> clear inner tunnel radius = 0.1235 units
 */
export const TRAIN_LENGTH_UNITS = 0.68;
export const TRAIN_BATTERY_LENGTH_UNITS = 0.42;
export const TRAIN_MAGNET_CENTER_OFFSET_UNITS = 0.278;
export const COIL_RADIUS_UNITS = 0.135;
export const WIRE_RADIUS_UNITS = 0.0125;

/**
 * Control points defining the heart/kidney-shaped loop on the horizontal table (Y = COIL_RADIUS_UNITS + WIRE_RADIUS_UNITS),
 * with coaxially aligned open coil mouths at the back-center gap (z = -0.36) and smooth, wide-radius curves around both lobes.
 */
export function getTrackControlPoints(): THREE.Vector3[] {
  const y = COIL_RADIUS_UNITS + WIRE_RADIUS_UNITS;
  return [
    // Gap center (u = 0): coaxial horizontal bridge along +X at z = -0.36
    new THREE.Vector3(0.0, y, -0.36),
    // Right coil mouth entry (coaxial with left exit mouth across the millimeter scale)
    new THREE.Vector3(0.42, y, -0.38),
    new THREE.Vector3(1.06, y, -0.56),
    // Back-right lobe curve (smooth radius R > 0.95)
    new THREE.Vector3(1.88, y, -0.78),
    new THREE.Vector3(2.48, y, -0.48),
    // Far right sweep coming forward
    new THREE.Vector3(2.64, y, 0.14),
    new THREE.Vector3(2.28, y, 0.78),
    // Front-right curve
    new THREE.Vector3(1.52, y, 1.22),
    // Front-center wide arc (closest to camera)
    new THREE.Vector3(0.44, y, 1.42),
    new THREE.Vector3(-0.72, y, 1.38),
    // Front-left curve
    new THREE.Vector3(-1.78, y, 1.04),
    // Far left sweep going back
    new THREE.Vector3(-2.48, y, 0.38),
    new THREE.Vector3(-2.56, y, -0.36),
    // Back-left lobe curve (smooth radius R > 0.95)
    new THREE.Vector3(-2.02, y, -0.86),
    new THREE.Vector3(-1.14, y, -0.62),
    // Left coil exit mouth approaching gap coaxially along +X at z = -0.38
    new THREE.Vector3(-0.42, y, -0.38),
  ];
}

let _cachedCurve: THREE.CatmullRomCurve3 | null = null;

interface CachedFrameSample {
  position: THREE.Vector3;
  tangent: THREE.Vector3;
  normal: THREE.Vector3;
  binormal: THREE.Vector3;
  curvature: number;
}

const FRAME_TABLE_SIZE = 2048;
let _cachedFrameTable: CachedFrameSample[] | null = null;
let _cachedTrackArcLength = TRACK_TOTAL_LENGTH;

export function getTrackCurve(): THREE.CatmullRomCurve3 {
  if (!_cachedCurve) {
    _cachedCurve = new THREE.CatmullRomCurve3(
      getTrackControlPoints(),
      true,
      'centripetal',
      0.5
    );
  }
  return _cachedCurve;
}

/**
 * Normalizes any parameter u into [0, 1)
 */
export function wrapU(u: number): number {
  const m = u % 1;
  return m < 0 ? m + 1 : m;
}

/**
 * Computes the signed shortest circular difference (a - b) in [-0.5, 0.5]
 */
export function circularDeltaU(a: number, b: number): number {
  let d = wrapU(a) - wrapU(b);
  if (d > 0.5) d -= 1;
  if (d < -0.5) d += 1;
  return d;
}

function ensureFrameTable(): CachedFrameSample[] {
  if (_cachedFrameTable) return _cachedFrameTable;
  const curve = getTrackCurve();
  const RAW_COUNT = 4096;
  const rawPositions: THREE.Vector3[] = new Array(RAW_COUNT);
  for (let i = 0; i < RAW_COUNT; i++) {
    rawPositions[i] = curve.getPoint(i / RAW_COUNT);
  }

  // Smooth positions across 7-tap circular Gaussian kernel for C2 continuity
  const smoothedPositions: THREE.Vector3[] = new Array(RAW_COUNT);
  const weights = [0.03, 0.11, 0.22, 0.28, 0.22, 0.11, 0.03];
  for (let i = 0; i < RAW_COUNT; i++) {
    const p = new THREE.Vector3();
    for (let k = -3; k <= 3; k++) {
      const idx = (i + k + RAW_COUNT) % RAW_COUNT;
      p.addScaledVector(rawPositions[idx], weights[k + 3]);
    }
    smoothedPositions[i] = p;
  }

  // Compute exact cumulative 3D Euclidean arc length around the closed loop
  const cumArcLen = new Float64Array(RAW_COUNT + 1);
  cumArcLen[0] = 0;
  for (let i = 1; i <= RAW_COUNT; i++) {
    const prev = smoothedPositions[i - 1];
    const curr = smoothedPositions[i % RAW_COUNT];
    cumArcLen[i] = cumArcLen[i - 1] + prev.distanceTo(curr);
  }
  const totalArcLen = cumArcLen[RAW_COUNT];
  _cachedTrackArcLength = totalArcLen;

  // Resample at strictly uniform Euclidean arc-length intervals so du = ds / getTrackArcLength() is 100% isometric everywhere!
  const uniformPositions: THREE.Vector3[] = new Array(FRAME_TABLE_SIZE);
  let searchIdx = 0;
  for (let i = 0; i < FRAME_TABLE_SIZE; i++) {
    const targetS = (i / FRAME_TABLE_SIZE) * totalArcLen;
    while (searchIdx < RAW_COUNT - 1 && cumArcLen[searchIdx + 1] < targetS) {
      searchIdx++;
    }
    const s0 = cumArcLen[searchIdx];
    const s1 = cumArcLen[searchIdx + 1];
    const alpha = s1 > s0 ? (targetS - s0) / (s1 - s0) : 0;
    uniformPositions[i] = new THREE.Vector3().lerpVectors(
      smoothedPositions[searchIdx],
      smoothedPositions[(searchIdx + 1) % RAW_COUNT],
      alpha
    );
  }

  const worldUp = new THREE.Vector3(0, 1, 0);
  const table: CachedFrameSample[] = new Array(FRAME_TABLE_SIZE);
  const ds = totalArcLen / FRAME_TABLE_SIZE;

  for (let i = 0; i < FRAME_TABLE_SIZE; i++) {
    const prevIdx = (i - 1 + FRAME_TABLE_SIZE) % FRAME_TABLE_SIZE;
    const nextIdx = (i + 1) % FRAME_TABLE_SIZE;
    const tangent = new THREE.Vector3()
      .subVectors(uniformPositions[nextIdx], uniformPositions[prevIdx])
      .normalize();

    const prevPrevIdx = (i - 4 + FRAME_TABLE_SIZE) % FRAME_TABLE_SIZE;
    const nextNextIdx = (i + 4) % FRAME_TABLE_SIZE;
    const tPlus = new THREE.Vector3()
      .subVectors(
        uniformPositions[(nextNextIdx + 1) % FRAME_TABLE_SIZE],
        uniformPositions[(nextNextIdx - 1 + FRAME_TABLE_SIZE) % FRAME_TABLE_SIZE]
      )
      .normalize();
    const tMinus = new THREE.Vector3()
      .subVectors(
        uniformPositions[(prevPrevIdx + 1) % FRAME_TABLE_SIZE],
        uniformPositions[(prevPrevIdx - 1 + FRAME_TABLE_SIZE) % FRAME_TABLE_SIZE]
      )
      .normalize();
    const curvature = tPlus.sub(tMinus).length() / (8 * ds);

    const binormal = new THREE.Vector3()
      .crossVectors(tangent, worldUp)
      .normalize();
    const normal = new THREE.Vector3()
      .crossVectors(binormal, tangent)
      .normalize();

    table[i] = {
      position: uniformPositions[i],
      tangent,
      normal,
      binormal,
      curvature,
    };
  }

  _cachedFrameTable = table;
  return table;
}

/**
 * Returns the exact Euclidean perimeter of the closed loop track in world units.
 */
export function getTrackArcLength(): number {
  ensureFrameTable();
  return _cachedTrackArcLength;
}

/**
 * Projects any 3D world point onto the closest normalized track parameter u in [0, 1)
 * for interactive mouse dragging of the battery train along the copper coil.
 */
export function findClosestTrackU(worldPoint: THREE.Vector3): number {
  const table = ensureFrameTable();
  let bestIdx = 0;
  let bestDistSq = Infinity;
  // Coarse pass every 4 samples
  for (let i = 0; i < FRAME_TABLE_SIZE; i += 4) {
    const dSq = table[i].position.distanceToSquared(worldPoint);
    if (dSq < bestDistSq) {
      bestDistSq = dSq;
      bestIdx = i;
    }
  }
  // Fine local refinement around bestIdx
  let fineIdx = bestIdx;
  for (let k = -4; k <= 4; k++) {
    const idx = (bestIdx + k + FRAME_TABLE_SIZE) % FRAME_TABLE_SIZE;
    const dSq = table[idx].position.distanceToSquared(worldPoint);
    if (dSq < bestDistSq) {
      bestDistSq = dSq;
      fineIdx = idx;
    }
  }
  return fineIdx / FRAME_TABLE_SIZE;
}

/**
 * Evaluates a C2-smooth, uniform-arc-length orthonormal frame on the horizontal track at parameter u in [0, 1),
 * interpolating smoothly across the 2048-sample table for zero micro-jitter at any speed.
 */
export function evaluateTrackFrame(u: number): TrackFrame {
  const table = ensureFrameTable();
  const normU = wrapU(u);
  const exactIdx = normU * FRAME_TABLE_SIZE;
  const idx0 = Math.floor(exactIdx) % FRAME_TABLE_SIZE;
  const idx1 = (idx0 + 1) % FRAME_TABLE_SIZE;
  const alpha = exactIdx - Math.floor(exactIdx);

  const s0 = table[idx0];
  const s1 = table[idx1];

  const position = new THREE.Vector3().lerpVectors(
    s0.position,
    s1.position,
    alpha
  );
  const tangent = new THREE.Vector3()
    .lerpVectors(s0.tangent, s1.tangent, alpha)
    .normalize();
  const binormal = new THREE.Vector3()
    .lerpVectors(s0.binormal, s1.binormal, alpha)
    .normalize();
  const normal = new THREE.Vector3()
    .crossVectors(binormal, tangent)
    .normalize();
  const curvature = s0.curvature + (s1.curvature - s0.curvature) * alpha;

  const inCoil = normU >= COIL_START_U && normU <= COIL_END_U;

  return {
    u: normU,
    position,
    tangent,
    normal,
    binormal,
    inCoil,
    curvature,
  };
}

/**
 * Converts a normalized track parameter u in [COIL_START_U, COIL_END_U] to a coil turn index [0, TOTAL_COIL_TURNS - 1]
 */
export function uToTurnIndex(u: number): number {
  const normU = wrapU(u);
  if (normU < COIL_START_U || normU > COIL_END_U) {
    return -1;
  }
  const coilFraction = (normU - COIL_START_U) / (COIL_END_U - COIL_START_U);
  return Math.max(
    0,
    Math.min(TOTAL_COIL_TURNS - 1, Math.floor(coilFraction * TOTAL_COIL_TURNS))
  );
}

/**
 * Computes the electromagnetic state (current, magnetic field, propulsion force, active turns, spark intensity)
 * for a given normalized train center position u, speed, and polarity direction.
 */
export function computeElectromagneticState(
  u: number,
  speed: number,
  distanceMeters = 0,
  lapCount = 0,
  direction: 1 | -1 = 1
): ElectromagneticTrainState {
  const normU = wrapU(u);
  const trackLen = getTrackArcLength();
  // Exact normalized offset from battery center (z = 0) to the center of the rear and front Neodymium magnet stacks
  const halfTrainDeltaU = TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;

  const lowMagnetU = wrapU(normU - halfTrainDeltaU);
  const highMagnetU = wrapU(normU + halfTrainDeltaU);

  const lowInCoil = lowMagnetU >= COIL_START_U && lowMagnetU <= COIL_END_U;
  const highInCoil = highMagnetU >= COIL_START_U && highMagnetU <= COIL_END_U;
  const centerInCoil = normU >= COIL_START_U && normU <= COIL_END_U;

  let coilCoupling = 0;
  if (lowInCoil && highInCoil && lowMagnetU < highMagnetU) {
    coilCoupling = 1.0;
  } else if (highInCoil || lowInCoil || centerInCoil) {
    coilCoupling = 0.68;
  } else {
    coilCoupling = 0.18;
  }

  let startTurn = -1;
  let endTurn = -1;
  if (lowInCoil && highInCoil && lowMagnetU < highMagnetU) {
    startTurn = uToTurnIndex(lowMagnetU);
    endTurn = uToTurnIndex(highMagnetU);
  } else if (highInCoil && !lowInCoil) {
    startTurn = 0;
    endTurn = Math.max(6, uToTurnIndex(highMagnetU));
  } else if (lowInCoil && !highInCoil) {
    startTurn = Math.min(TOTAL_COIL_TURNS - 7, uToTurnIndex(lowMagnetU));
    endTurn = TOTAL_COIL_TURNS - 1;
  }

  // Physical constants for AA 1.5V battery + N52 Neodymium magnets in bare copper coil
  const batteryVoltage = 1.5; // Volts
  const segmentResistance = 0.82; // Ohms
  const backEmf = 0.08 * speed;
  const currentAmps =
    coilCoupling >= 0.5
      ? Math.max(0, ((batteryVoltage - backEmf) / segmentResistance) * coilCoupling)
      : 0;

  const magneticFieldmT = currentAmps * 6.8;

  // Lorentz / Laplace force + extra magnetic reluctance suction at coil mouths
  const mouthTransitionBoost =
    coilCoupling > 0.5 && coilCoupling < 1.0 ? 1.18 : 1.0;
  const propulsionForceN =
    coilCoupling >= 0.5
      ? currentAmps * 0.32 * mouthTransitionBoost
      : 0.24;

  const frame = evaluateTrackFrame(normU);
  // Sliding brush spark intensity increases with current, speed, curvature, and coil mouth entry/exit
  const mouthSparkBoost = coilCoupling > 0.4 && coilCoupling < 1.0 ? 0.35 : 0.0;
  const sparkIntensity =
    currentAmps > 0.1
      ? THREE.MathUtils.clamp(
          0.35 + speed * 0.25 + frame.curvature * 0.45 + mouthSparkBoost,
          0,
          1
        )
      : 0;

  const trainMassKg = 0.032; // 32g
  const curveFrictionDrag = frame.curvature * 0.025 * speed;
  const frictionAndDragN = 0.06 + speed * 0.27 + curveFrictionDrag;
  const netForceN = propulsionForceN - frictionAndDragN;
  const acceleration = netForceN / trainMassKg;

  return {
    u: normU,
    direction,
    distanceMeters,
    speed,
    acceleration,
    currentAmps: Number(currentAmps.toFixed(2)),
    magneticFieldmT: Number(magneticFieldmT.toFixed(1)),
    propulsionForceN: Number(propulsionForceN.toFixed(3)),
    sparkIntensity: Number(sparkIntensity.toFixed(2)),
    inCoil: lowInCoil || highInCoil,
    coilCoupling,
    activeTurnRange: [startTurn, endTurn],
    totalTurns: TOTAL_COIL_TURNS,
    lapCount,
  };
}

/**
 * Advances the electromagnetic train state by dt seconds using 4x sub-stepping RK2 integration
 * for ultra-smooth, physically realistic motion across curves and coil mouths.
 */
export function stepTrainPhysics(
  state: ElectromagneticTrainState,
  dt: number,
  speedMultiplier = 1.0
): ElectromagneticTrainState {
  const clampedDt = Math.min(Math.max(dt, 0), 0.05);
  const subSteps = 4;
  const subDt = clampedDt / subSteps;
  const trackLen = getTrackArcLength();

  let curU = state.u;
  let curSpeed = state.speed;
  let curDistMeters = state.distanceMeters;
  let curLapCount = state.lapCount;
  const dir = state.direction;

  for (let s = 0; s < subSteps; s++) {
    const k1 = computeElectromagneticState(
      curU,
      curSpeed,
      curDistMeters,
      curLapCount,
      dir
    );
    const midSpeed = THREE.MathUtils.clamp(
      curSpeed + k1.acceleration * (subDt * 0.5),
      0.55,
      1.85
    );
    const midDeltaUnits = midSpeed * speedMultiplier * (subDt * 0.5);
    const midU = wrapU(curU + (midDeltaUnits / trackLen) * dir);

    const k2 = computeElectromagneticState(
      midU,
      midSpeed,
      curDistMeters + midDeltaUnits * 0.1,
      curLapCount,
      dir
    );

    curSpeed = THREE.MathUtils.clamp(
      curSpeed + k2.acceleration * subDt,
      0.55,
      1.85
    );
    const stepUnits = curSpeed * speedMultiplier * subDt;
    const signedDeltaU = (stepUnits / trackLen) * dir;
    const rawNextU = curU + signedDeltaU;

    if (dir === 1 && rawNextU >= 1.0) {
      curLapCount += Math.floor(rawNextU);
    } else if (dir === -1 && rawNextU < 0.0) {
      curLapCount += Math.ceil(Math.abs(rawNextU));
    }

    curU = wrapU(rawNextU);
    curDistMeters += stepUnits * 0.1;
  }

  return computeElectromagneticState(
    curU,
    curSpeed,
    curDistMeters,
    curLapCount,
    dir
  );
}

/**
 * Reverses the polarity/direction of the electromagnetic train (simulating flipping the battery + magnets inside the coil).
 */
export function reverseTrainPolarity(
  state: ElectromagneticTrainState
): ElectromagneticTrainState {
  const nextDirection: 1 | -1 = state.direction === 1 ? -1 : 1;
  return computeElectromagneticState(
    state.u,
    Math.max(0.75, state.speed * 0.85),
    state.distanceMeters,
    state.lapCount,
    nextDirection
  );
}

export interface PovCameraFrame {
  eyePosition: THREE.Vector3;
  lookTarget: THREE.Vector3;
  upVector: THREE.Vector3;
  radialClearanceRatio: number;
}

/**
 * Computes the exact first-person POV camera pose riding inside the helical copper coil tunnel
 * at the upper leading crown of the front Neodymium magnet stack, looking over the front magnet bevel
 * and sliding electric contact brush down the curving wire rail ahead.
 */
export function computeFirstPersonPovFrame(
  state: ElectromagneticTrainState
): PovCameraFrame {
  const dir = state.direction;
  const trackLen = getTrackArcLength();
  // Mount the POV lens in the upper tunnel headspace right above the front shoulder (+0.21 units ahead of center)
  const eyeU = wrapU(state.u + (0.21 / trackLen) * dir);
  const eyeFrame = evaluateTrackFrame(eyeU);

  // Sit at +0.0675 units above centerline (strictly above the gravity-seated magnet top, with radialClearanceRatio = 0.50 > 0.45)
  const verticalOffset = COIL_RADIUS_UNITS * 0.5;

  const eyePosition = eyeFrame.position
    .clone()
    .addScaledVector(eyeFrame.normal, verticalOffset);

  // Look 0.24 units ahead of the POV lens along the curved spline tunnel
  const lookAheadU = wrapU(eyeU + (0.24 / trackLen) * dir);
  const aheadFrame = evaluateTrackFrame(lookAheadU);
  const lookTarget = aheadFrame.position
    .clone()
    .addScaledVector(aheadFrame.normal, COIL_RADIUS_UNITS * 0.18);

  // Subtle centrifugal roll bank into curves
  const upVector = eyeFrame.normal
    .clone()
    .addScaledVector(eyeFrame.binormal, eyeFrame.curvature * 0.075 * dir)
    .normalize();

  const radialDistanceFromAxis = verticalOffset;
  const radialClearanceRatio =
    (COIL_RADIUS_UNITS - radialDistanceFromAxis) / COIL_RADIUS_UNITS;

  return {
    eyePosition,
    lookTarget,
    upVector,
    radialClearanceRatio,
  };
}

export interface CoilTurnPerturbation {
  /** Radial expansion of the helical turn radius (in world units, >0 means outward breathing) */
  radialDelta: number;
  /** Axial peristaltic compression/stretch shift along the local track tangent (world units) */
  axialDelta: number;
  /** Lateral centrifugal & magnetic whip shift along the local binormal (world units) */
  lateralDelta: number;
  /** Vertical arch flex along the local normal (world units; bottom rests on table) */
  verticalArchDelta: number;
  /** Normalized [0..1] perturbation intensity for telemetry and shading */
  intensity: number;
}

/**
 * Computes the real-time mechanical and electromagnetic perturbation of a helical copper coil turn
 * at normalized track parameter uTurn as the AA battery + Neodymium magnet stacks pass through it.
 * Combines:
 * 1. Outward radial mechanical contact expansion at the front/rear Neodymium magnet brushes
 * 2. Outward Lorentz radial hoop stress (J_theta x B_z) along the energized solenoid turns
 * 3. Newton's 3rd Law axial peristaltic compression/tension wave along the track tangent
 * 4. Damped elastic spring ringing wake behind the rear magnet + cantilever whip at the open coil mouths
 */
export function computeCoilTurnPerturbation(
  uTurn: number,
  state: ElectromagneticTrainState,
  elapsedTime: number
): CoilTurnPerturbation {
  const trackLen = getTrackArcLength();
  const dir = state.direction;
  const dsCenter = circularDeltaU(uTurn, state.u) * trackLen * dir;

  // Outside the active interaction & elastic wake zone, the copper coil rests undisturbed
  if (Math.abs(dsCenter) > 1.55) {
    return {
      radialDelta: 0,
      axialDelta: 0,
      lateralDelta: 0,
      verticalArchDelta: 0,
      intensity: 0,
    };
  }

  const halfMagDeltaU = TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;
  const frontU = wrapU(state.u + halfMagDeltaU * dir);
  const rearU = wrapU(state.u - halfMagDeltaU * dir);

  const dsFront = circularDeltaU(uTurn, frontU) * trackLen * dir;
  const dsRear = circularDeltaU(uTurn, rearU) * trackLen * dir;

  const sigmaBrush = 0.082;
  const frontContactEnv = Math.exp(
    -(dsFront * dsFront) / (2 * sigmaBrush * sigmaBrush)
  );
  const rearContactEnv = Math.exp(
    -(dsRear * dsRear) / (2 * sigmaBrush * sigmaBrush)
  );

  // Solenoid bridge envelope between the two magnet stacks
  const normBridge = dsCenter / (TRAIN_MAGNET_CENTER_OFFSET_UNITS * 1.08);
  const bridgeEnv =
    Math.exp(-Math.pow(normBridge, 4)) *
    Math.min(1, Math.max(0.35, state.currentAmps / 1.65));

  // Damped elastic spring wake trailing behind the rear magnet stack (dsRear < 0)
  const wakeDist = -dsRear;
  const speedFactor = THREE.MathUtils.clamp(state.speed / 1.1, 0.5, 1.6);
  const wakeEnv =
    wakeDist > 0 && wakeDist < 1.15
      ? Math.exp(-wakeDist * 3.1) *
        Math.cos(wakeDist * 36 - elapsedTime * 32 * speedFactor) *
        speedFactor
      : 0;
  const wakeMag =
    wakeDist > 0 && wakeDist < 1.15
      ? Math.exp(-wakeDist * 3.1) * speedFactor
      : 0;

  // High-frequency sliding brush micro-vibration as the magnet discs cross individual wire turns
  const turnAngle = uTurn * TOTAL_COIL_TURNS * Math.PI * 2;
  const brushTremor =
    (frontContactEnv + rearContactEnv) *
    0.0065 *
    Math.sin(elapsedTime * 58 * speedFactor + turnAngle * 0.5) *
    speedFactor;

  // 1. Radial expansion (peristaltic breathing of copper turns at magnet brushes + Lorentz hoop stress + elastic spring wake)
  const radialDelta =
    frontContactEnv * 0.028 +
    rearContactEnv * 0.026 +
    bridgeEnv * 0.016 +
    wakeEnv * 0.0135 +
    brushTremor;

  // 2. Axial peristaltic slinky wave: turns compress ahead of brushes, recoil opposite to F_Lorentz, and ring in the wake
  const frontAxial = -dsFront * 15.5 * frontContactEnv * 0.0125;
  const rearAxial = -dsRear * 15.5 * rearContactEnv * 0.0125;
  const lorentzRecoil = -bridgeEnv * 0.0115 * dir;
  const axialWake =
    (wakeDist > 0 && wakeDist < 1.15
      ? Math.exp(-wakeDist * 2.9) *
        Math.sin(wakeDist * 42 - elapsedTime * 34 * speedFactor) *
        0.014 *
        speedFactor
      : 0) * dir;
  const axialDelta =
    (frontAxial + rearAxial) * dir + lorentzRecoil + axialWake;

  // 3. Lateral elastic whip (binormal reaction + cantilevered open coil mouth oscillation near the regleta)
  const mouthDist = Math.min(
    Math.abs(uTurn - COIL_START_U),
    Math.abs(uTurn - COIL_END_U)
  );
  const mouthProximity = Math.max(0, 1 - mouthDist / 0.045);
  const mouthWhip =
    mouthProximity *
    (frontContactEnv + rearContactEnv + wakeMag) *
    0.015 *
    Math.sin(elapsedTime * 36 + uTurn * 90);

  const lateralWave =
    (frontContactEnv - rearContactEnv) *
      0.0095 *
      Math.sin(elapsedTime * 40 + uTurn * 160) +
    wakeEnv * 0.0085 +
    mouthWhip;

  // 4. Vertical arch flex (upper copper arch lifts visibly as magnets press & expand the turn)
  const verticalArchDelta =
    (frontContactEnv + rearContactEnv) * 0.014 +
    bridgeEnv * 0.0085 +
    wakeEnv * 0.0065;

  const intensity = THREE.MathUtils.clamp(
    Math.max(
      frontContactEnv,
      rearContactEnv,
      bridgeEnv * 0.85,
      wakeMag * 0.75
    ),
    0,
    1
  );

  return {
    radialDelta,
    axialDelta,
    lateralDelta: lateralWave,
    verticalArchDelta,
    intensity,
  };
}
