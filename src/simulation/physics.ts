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
  /** Battery polarity: sign of the propulsion force along +u (+1 forward, -1 reversed) */
  direction: 1 | -1;
  /** Distance traveled in meters (always increasing) */
  distanceMeters: number;
  /** Signed velocity along +u in m/s (can oppose `direction` while braking after a reversal) */
  velocity: number;
  /** Speed magnitude in m/s */
  speed: number;
  /** Instantaneous tangential acceleration in m/s^2 */
  acceleration: number;
  /** Electric current in Amperes flowing through the bridged coil turns */
  currentAmps: number;
  /** Axial solenoid magnetic field B = mu0 * n * I inside bridged turns (mT) */
  magneticFieldmT: number;
  /** Electromagnetic propulsion force magnitude on the magnets (N) */
  propulsionForceN: number;
  /** Whether the battery circuit is closed through the coil (both magnets touch copper) */
  circuitClosed: boolean;
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
 * Physical model of the homopolar "magnet train" (SI units).
 *
 * The battery and the two neodymium stacks close a circuit through the copper turns that lie between
 * the magnets. That stretch of coil becomes a short solenoid with B = mu0 * n * I. Each magnet sits at
 * one end of that solenoid, where the field gradient pushes on its dipole moment; with like poles facing
 * outward both pushes point the same way. Using F ~ m * dB/dz over a transition length of about 2a gives
 * a force constant k = F / I. The same k sets the back-EMF (k * v), so energy is conserved.
 *
 * When a magnet rides over the open gap (insulated anodized ruler) the circuit opens: no current, no
 * force, and the train has to coast across on its own momentum.
 */
export const PHYSICS = {
  batteryEmfV: 1.5,
  /** Internal resistance of an alkaline AA under heavy load plus magnet-to-wire contact resistance */
  circuitResistanceOhm: 0.65,
  /** Bare copper wire resistance of the full coil (~29 m of 2.15 mm wire) */
  fullCoilResistanceOhm: 0.135,
  /** Dipole moment of one stack of four 17 x 2.6 mm N52 discs (M ~ 1.15e6 A/m) */
  magnetDipoleMomentAm2: 2.8,
  /** AA cell (23 g) plus two stacks of NdFeB discs (18 g each) */
  trainMassKg: 0.059,
  /** Sliding/rolling friction against the wire turns */
  kineticFriction: 0.05,
  staticFriction: 0.08,
  /** Velocity-proportional contact losses (bouncing on the turns, micro-arcing) per magnet on copper, N*s/m */
  contactDampingPerMagnet: 0.095,
  gravity: 9.81,
};

const MU0 = 4e-7 * Math.PI;
/** 1 world unit = 10 cm */
export const METERS_PER_UNIT = 0.1;

export function getCoilTurnsPerMeter(): number {
  const coilLengthM =
    (COIL_END_U - COIL_START_U) * getTrackArcLength() * METERS_PER_UNIT;
  return TOTAL_COIL_TURNS / coilLengthM;
}

/** Force constant k (N/A): F = k * I and back-EMF = k * v. */
export function getForceConstant(): number {
  const coilRadiusM = COIL_RADIUS_UNITS * METERS_PER_UNIT;
  return (
    (2 * PHYSICS.magnetDipoleMomentAm2 * MU0 * getCoilTurnsPerMeter()) /
    (2 * coilRadiusM)
  );
}

function isOnCopper(u: number): boolean {
  return u >= COIL_START_U && u <= COIL_END_U;
}

/** Sign of travel: follows the velocity, falls back to the polarity when (almost) at rest. */
export function travelSign(state: ElectromagneticTrainState): 1 | -1 {
  if (state.velocity > 0.01) return 1;
  if (state.velocity < -0.01) return -1;
  return state.direction;
}

/**
 * Computes the electromagnetic state (circuit, current, field, force, energized turns, sparks)
 * for a train centered at normalized position u moving with signed velocity (m/s).
 */
export function computeElectromagneticState(
  u: number,
  velocity: number,
  distanceMeters = 0,
  lapCount = 0,
  direction: 1 | -1 = 1
): ElectromagneticTrainState {
  const normU = wrapU(u);
  const trackLen = getTrackArcLength();
  const halfTrainDeltaU = TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;

  const lowMagnetU = wrapU(normU - halfTrainDeltaU);
  const highMagnetU = wrapU(normU + halfTrainDeltaU);
  const lowInCoil = isOnCopper(lowMagnetU);
  const highInCoil = isOnCopper(highMagnetU);

  const circuitClosed = lowInCoil && highInCoil;
  // Normally the energized stretch lies between the magnets. When the train straddles the gap with both
  // magnets on copper, the current has to go the long way round through the entire coil.
  const straddlesGap = circuitClosed && lowMagnetU > highMagnetU;

  let startTurn = -1;
  let endTurn = -1;
  if (circuitClosed && !straddlesGap) {
    startTurn = uToTurnIndex(lowMagnetU);
    endTurn = uToTurnIndex(highMagnetU);
  } else if (straddlesGap) {
    startTurn = 0;
    endTurn = TOTAL_COIL_TURNS - 1;
  }

  const k = getForceConstant();
  const resistance =
    PHYSICS.circuitResistanceOhm +
    (straddlesGap ? PHYSICS.fullCoilResistanceOhm : 0);
  // Back-EMF opposes the current only when the train moves along the force direction
  const backEmf = k * velocity * direction;
  const currentAmps = circuitClosed
    ? Math.max(0, (PHYSICS.batteryEmfV - backEmf) / resistance)
    : 0;
  const magneticFieldmT = MU0 * getCoilTurnsPerMeter() * currentAmps * 1000;
  const propulsionForceN = k * currentAmps;

  // Net tangential force (signed along +u)
  const magnetsOnCopper = (lowInCoil ? 1 : 0) + (highInCoil ? 1 : 0);
  const frame = evaluateTrackFrame(normU);
  const curvaturePerM = frame.curvature / METERS_PER_UNIT;
  const centripetal = velocity * velocity * curvaturePerM;
  const normalForce =
    PHYSICS.trainMassKg * Math.hypot(PHYSICS.gravity, centripetal);
  const drive = propulsionForceN * direction;
  const damping = PHYSICS.contactDampingPerMagnet * magnetsOnCopper * velocity;
  const friction =
    Math.abs(velocity) > 1e-4
      ? PHYSICS.kineticFriction * normalForce * Math.sign(velocity)
      : 0;
  const acceleration = (drive - damping - friction) / PHYSICS.trainMassKg;

  // Contact sparks grow with current and sliding speed; a magnet entering or leaving copper arcs hardest
  const atMouth = lowInCoil !== highInCoil || straddlesGap;
  const sparkIntensity =
    currentAmps > 0.05
      ? THREE.MathUtils.clamp(
          (currentAmps / 2.3) * (0.35 + 0.65 * Math.min(1, Math.abs(velocity) / 0.6)) +
            (atMouth ? 0.3 : 0),
          0,
          1
        )
      : 0;

  return {
    u: normU,
    direction,
    distanceMeters,
    velocity,
    speed: Math.abs(velocity),
    acceleration,
    currentAmps,
    magneticFieldmT,
    propulsionForceN,
    circuitClosed,
    sparkIntensity,
    inCoil: lowInCoil || highInCoil,
    coilCoupling: circuitClosed ? (straddlesGap ? 0.9 : 1) : 0,
    activeTurnRange: [startTurn, endTurn],
    totalTurns: TOTAL_COIL_TURNS,
    lapCount,
  };
}

/**
 * Advances the train by dt seconds of simulated time (semi-implicit Euler, 8 sub-steps) with
 * static friction: a train at rest only starts moving when the drive overcomes it.
 */
export function stepTrainPhysics(
  state: ElectromagneticTrainState,
  dt: number
): ElectromagneticTrainState {
  const clampedDt = Math.min(Math.max(dt, 0), 0.05);
  const subSteps = 8;
  const subDt = clampedDt / subSteps;
  const trackLenM = getTrackArcLength() * METERS_PER_UNIT;

  let cur = state;
  let u = state.u;
  let v = state.velocity;
  let dist = state.distanceMeters;
  let laps = state.lapCount;
  const dir = state.direction;

  for (let s = 0; s < subSteps; s++) {
    cur = computeElectromagneticState(u, v, dist, laps, dir);

    if (Math.abs(v) < 1e-4) {
      const drive = cur.propulsionForceN;
      const holding =
        PHYSICS.staticFriction * PHYSICS.trainMassKg * PHYSICS.gravity;
      if (drive <= holding) {
        v = 0;
        continue;
      }
    }

    const nextV = v + cur.acceleration * subDt;
    // Friction and damping can stop the train but never push it backwards on their own
    const drivePushesBack = cur.propulsionForceN * dir * Math.sign(v) < 0;
    v =
      v !== 0 && Math.sign(nextV) !== Math.sign(v) && !drivePushesBack
        ? 0
        : nextV;

    const rawNextU = u + (v * subDt) / trackLenM;
    if (rawNextU >= 1) laps += 1;
    else if (rawNextU < 0) laps += 1;
    u = wrapU(rawNextU);
    dist += Math.abs(v) * subDt;
  }

  return computeElectromagneticState(u, v, dist, laps, dir);
}

/**
 * Reverses the battery polarity. The train keeps its momentum, so it first brakes under the
 * reversed force and then accelerates the other way.
 */
export function reverseTrainPolarity(
  state: ElectromagneticTrainState
): ElectromagneticTrainState {
  const nextDirection: 1 | -1 = state.direction === 1 ? -1 : 1;
  return computeElectromagneticState(
    state.u,
    state.velocity,
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
  const dir = travelSign(state);
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
  const dir = travelSign(state);
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
    Math.exp(-Math.pow(normBridge, 4)) * Math.min(1, state.currentAmps / 2.0);

  // Damped elastic spring wake trailing behind the rear magnet stack (dsRear < 0)
  const wakeDist = -dsRear;
  const speedFactor = THREE.MathUtils.clamp(state.speed / 0.6, 0, 1.4);
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

  // The raw envelopes above describe the shape of the disturbance. A real coil barely moves (the train
  // weighs ~0.6 N), so displacements are scaled to roughly a millimetre: visible up close, not rubbery.
  const REALISM_SCALE = 0.14;
  return {
    radialDelta: radialDelta * REALISM_SCALE,
    axialDelta: axialDelta * REALISM_SCALE,
    lateralDelta: lateralWave * REALISM_SCALE,
    verticalArchDelta: verticalArchDelta * REALISM_SCALE,
    intensity,
  };
}
