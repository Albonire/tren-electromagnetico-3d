import * as THREE from 'three';
import {
  COIL_START_U,
  COIL_END_U,
  TOTAL_COIL_TURNS,
  COIL_RADIUS_UNITS,
  WIRE_RADIUS_UNITS,
  TRAIN_BATTERY_LENGTH_UNITS,
  TRAIN_MAGNET_CENTER_OFFSET_UNITS,
  getTrackArcLength,
  evaluateTrackFrame,
  wrapU,
  circularDeltaU,
  computeCoilTurnPerturbation,
  ElectromagneticTrainState,
} from './physics';
import {
  createBatteryWrapperTexture,
  createBatteryRoughnessMap,
  createMagnetPoleCapTexture,
  createBrushedMetalBumpTexture,
  createNaturalCopperAlbedoTexture,
  createTabletopAlbedoTexture,
  createTabletopMicroTexture,
  createCoilRibbonShadowTexture,
  createSoftShadowTexture,
  createStudioBackdropTexture,
  createWindowSunbeamTexture,
  createDiegeticCalloutSprite,
  createPovMicroLabelSprite,
  createPrecisionWorkbenchScaleTexture,
} from './textures';

export interface HelicalCoilRig {
  group: THREE.Group;
  coilMesh: THREE.Mesh;
  activeTurnsInstanced: THREE.InstancedMesh;
  currentParticles: THREE.InstancedMesh;
  solenoidCoreParticles: THREE.InstancedMesh;
  turnUValues: number[];
  thermalLevels: Float32Array;
  maxPerturbationMm: number;
  updateActiveTurns: (
    state: ElectromagneticTrainState,
    elapsedTime: number,
    dt: number,
    showElectronFlow?: boolean
  ) => void;
}

export interface TrainRig {
  rootGroup: THREE.Group;
  batteryGroup: THREE.Group;
  frontMagnetGroup: THREE.Group;
  rearMagnetGroup: THREE.Group;
  physicsOverlayGroup: THREE.Group;
  trainShadowMesh: THREE.Mesh;
  frontContactRing: THREE.Mesh;
  rearContactRing: THREE.Mesh;
  sparkInstanced: THREE.InstancedMesh;
  forceArrowGroup: THREE.Group;
  fieldLoopsGroup: THREE.Group;
  updateTrainPose: (
    state: ElectromagneticTrainState,
    elapsedTime: number,
    showExplicitFields: boolean
  ) => void;
}

// Pre-allocated scratch math objects (Zero GC in render loop)
const _dummyObj = new THREE.Object3D();
const _scratchPos = new THREE.Vector3();
const _scratchLook = new THREE.Vector3();
const _scratchColor = new THREE.Color();
const _inactiveScale = new THREE.Vector3(0.0001, 0.0001, 0.0001);

/**
 * Shared organic turn spacing mapping t in [0, 1] -> normalized track parameter u in [COIL_START_U, COIL_END_U]
 * used identically by both the 3D TubeGeometry helix and the per-turn InstancedMesh so they stay 100% in phase.
 */
function mapCoilFractionToTrackU(t: number): number {
  const organicT = THREE.MathUtils.clamp(
    t +
      Math.sin(t * Math.PI * 18) * 0.00075 +
      Math.cos(t * Math.PI * 43) * 0.00038 -
      Math.sin(t * Math.PI) * 0.0012,
    0,
    1
  );
  return COIL_START_U + organicT * (COIL_END_U - COIL_START_U);
}

/**
 * Deterministic organic spacing perturbation so the 340 wire turns mimic the hand-wound coil.
 */
export function getTurnNormalizedU(turnIndex: number): number {
  return mapCoilFractionToTrackU(turnIndex / TOTAL_COIL_TURNS);
}

/**
 * Builds the dark slate laboratory tabletop with diagonal window sunlight patch,
 * the 3D physical precision ruler ("regleta milimetrada") between the two open coil mouths,
 * subtle volumetric sunbeam shafts from the nearby upper-right window, and dark studio backdrop.
 */
export function buildLabEnvironment(
  scene: THREE.Scene,
  maxAnisotropy: number,
  isSoftwareGl: boolean
): THREE.Group {
  const envGroup = new THREE.Group();
  const tableAlbedo = createTabletopAlbedoTexture(maxAnisotropy);
  const tableBump = createTabletopMicroTexture(maxAnisotropy);
  const ribbonShadowTex = createCoilRibbonShadowTexture();

  // 1. Dark Slate / Obsidian Laboratory Workbench Surface with Diagonal Window Sunlight Patch
  const tableGeo = new THREE.BoxGeometry(16.0, 0.22, 10.5);
  const tableMat = new THREE.MeshStandardMaterial({
    map: tableAlbedo,
    color: 0xd8dee9,
    roughness: 0.76,
    metalness: 0.05,
    envMapIntensity: 0.05,
    bumpMap: tableBump,
    bumpScale: 0.0018,
  });
  const tableMesh = new THREE.Mesh(tableGeo, tableMat);
  tableMesh.position.set(0, -0.11, 2.05);
  tableMesh.receiveShadow = !isSoftwareGl;
  envGroup.add(tableMesh);

  // Dark anodized titanium back edge trim of the lab bench
  const edgeGeo = new THREE.BoxGeometry(16.02, 0.225, 0.045);
  const edgeMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.42,
    metalness: 0.55,
    envMapIntensity: 0.15,
  });
  const backEdge = new THREE.Mesh(edgeGeo, edgeMat);
  backEdge.position.set(0, -0.11, -3.2);
  envGroup.add(backEdge);

  // 2. Feathered Contact Shadow + Sunlit Copper Caustic Ribbon directly under the heart-shaped coil loop
  const shadowRibbonGeo = buildCoilShadowRibbonGeometry();
  const shadowRibbonMat = new THREE.MeshBasicMaterial({
    map: ribbonShadowTex,
    transparent: true,
    opacity: 0.92,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const shadowRibbon = new THREE.Mesh(shadowRibbonGeo, shadowRibbonMat);
  shadowRibbon.position.y = 0.003;
  envGroup.add(shadowRibbon);

  // 2b. 3D Physical Precision Ruler & Coaxial Alignment Guide ("Regleta Milimetrada") between the two open coil mouths
  const regletaGroup = new THREE.Group();
  regletaGroup.position.set(0.0, 0.0, -0.368);

  // 3D Machined dark-anodized aluminum ruler base plate
  const regletaBaseGeo = new THREE.BoxGeometry(1.42, 0.009, 0.56);
  const regletaBaseMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    metalness: 0.72,
    roughness: 0.32,
    envMapIntensity: 0.25,
  });
  const regletaBaseMesh = new THREE.Mesh(regletaBaseGeo, regletaBaseMat);
  regletaBaseMesh.position.set(0, 0.0045, 0);
  regletaBaseMesh.receiveShadow = !isSoftwareGl;
  regletaGroup.add(regletaBaseMesh);

  // Sleek low-profile beveled side rails framing the regleta
  const railTrimGeo = new THREE.BoxGeometry(1.425, 0.012, 0.014);
  const railTrimMat = new THREE.MeshStandardMaterial({
    color: 0xb45309,
    metalness: 0.85,
    roughness: 0.24,
    envMapIntensity: 0.35,
  });
  const frontRailTrim = new THREE.Mesh(railTrimGeo, railTrimMat);
  frontRailTrim.position.set(0, 0.006, 0.274);
  regletaGroup.add(frontRailTrim);

  const backRailTrim = new THREE.Mesh(railTrimGeo, railTrimMat);
  backRailTrim.position.set(0, 0.006, -0.274);
  regletaGroup.add(backRailTrim);

  // High-contrast engraved millimeter ruler top face (strictly above shadowRibbon at y = 0.0096)
  const scaleDecalTex = createPrecisionWorkbenchScaleTexture(maxAnisotropy);
  const scaleDecalGeo = new THREE.PlaneGeometry(1.39, 0.525);
  scaleDecalGeo.rotateX(-Math.PI / 2);
  const scaleDecalMat = new THREE.MeshStandardMaterial({
    map: scaleDecalTex,
    emissive: 0xffffff,
    emissiveMap: scaleDecalTex,
    emissiveIntensity: 0.32,
    roughness: 0.36,
    metalness: 0.2,
    envMapIntensity: 0.15,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  const scaleDecalMesh = new THREE.Mesh(scaleDecalGeo, scaleDecalMat);
  scaleDecalMesh.position.set(0.0, 0.0096, 0.0);
  scaleDecalMesh.receiveShadow = !isSoftwareGl;
  regletaGroup.add(scaleDecalMesh);

  envGroup.add(regletaGroup);

  // 3. Dark Studio Lab Backdrop with Nearby Side Window on Upper-Right
  const backdropTex = createStudioBackdropTexture(maxAnisotropy);
  const backdropGeo = new THREE.PlaneGeometry(14.5, 5.2);
  const backdropMat = new THREE.MeshBasicMaterial({
    map: backdropTex,
    fog: false,
  });
  const backdropMesh = new THREE.Mesh(backdropGeo, backdropMat);
  backdropMesh.position.set(0.15, 1.45, -3.26);
  envGroup.add(backdropMesh);

  // 4. Diagonal Volumetric-Style Sunlight Shafts entering from the nearby upper-right window into the dark room
  const sunbeamTex = createWindowSunbeamTexture();
  const sunbeamMat = new THREE.MeshBasicMaterial({
    map: sunbeamTex,
    transparent: true,
    opacity: 0.11,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const beamConfigs = [
    { pos: [1.55, 1.55, -0.55], rotZ: 0.56, rotY: -0.28, w: 2.6, h: 4.4 },
    { pos: [2.05, 1.72, -1.05], rotZ: 0.54, rotY: 0.22, w: 2.2, h: 4.6 },
    { pos: [0.95, 1.38, 0.15], rotZ: 0.58, rotY: -0.12, w: 2.0, h: 4.1 },
  ];
  for (const cfg of beamConfigs) {
    const beamGeo = new THREE.PlaneGeometry(cfg.w, cfg.h);
    const beamMesh = new THREE.Mesh(beamGeo, sunbeamMat);
    beamMesh.position.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
    beamMesh.rotation.set(0.18, cfg.rotY, cfg.rotZ);
    beamMesh.layers.set(1);
    envGroup.add(beamMesh);
  }

  scene.add(envGroup);
  return envGroup;
}

/**
 * Generates a flat ribbon geometry along the coil spline to cast a feathered contact shadow
 * onto the dark slate laboratory table.
 */
function buildCoilShadowRibbonGeometry(): THREE.BufferGeometry {
  const segments = 260;
  const halfWidth = COIL_RADIUS_UNITS * 2.15;
  const positions = new Float32Array((segments + 1) * 2 * 3);
  const uvs = new Float32Array((segments + 1) * 2 * 2);
  const indices: number[] = [];

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const u = COIL_START_U + t * (COIL_END_U - COIL_START_U);
    const frame = evaluateTrackFrame(u);

    const left = frame.position
      .clone()
      .addScaledVector(frame.binormal, -halfWidth);
    const right = frame.position
      .clone()
      .addScaledVector(frame.binormal, halfWidth);

    const vIdx = i * 6;
    positions[vIdx] = left.x;
    positions[vIdx + 1] = 0;
    positions[vIdx + 2] = left.z;

    positions[vIdx + 3] = right.x;
    positions[vIdx + 4] = 0;
    positions[vIdx + 5] = right.z;

    const uvIdx = i * 4;
    uvs[uvIdx] = 0;
    uvs[uvIdx + 1] = t;
    uvs[uvIdx + 2] = 1;
    uvs[uvIdx + 3] = t;

    if (i < segments) {
      const a = i * 2;
      const b = i * 2 + 1;
      const c = (i + 1) * 2;
      const d = (i + 1) * 2 + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Custom 3D Curve representing the 340-turn helical wire winding along the heart/kidney loop spline.
 * Overrides getPointAt(u) to call getPoint(u) directly because evaluateTrackFrame(u) is already
 * uniform-arc-length parameterized (avoiding Three.js's 200-division chord-length aliasing on 340 turns).
 */
class HelicalSolenoidCurve extends THREE.Curve<THREE.Vector3> {
  constructor() {
    super();
  }

  getPoint(t: number, optionalTarget = new THREE.Vector3()): THREE.Vector3 {
    const u = mapCoilFractionToTrackU(t);
    const frame = evaluateTrackFrame(u);

    const angle = t * TOTAL_COIL_TURNS * Math.PI * 2;
    const rJitter =
      1 +
      Math.sin(angle * 0.37) * 0.011 +
      Math.cos(angle * 0.13) * 0.007;
    const radius = COIL_RADIUS_UNITS * rJitter;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    optionalTarget
      .copy(frame.position)
      .addScaledVector(frame.binormal, cosA * radius)
      .addScaledVector(frame.normal, sinA * radius);

    if (optionalTarget.y < WIRE_RADIUS_UNITS) {
      optionalTarget.y = WIRE_RADIUS_UNITS;
    }
    return optionalTarget;
  }

  override getPointAt(
    u: number,
    optionalTarget = new THREE.Vector3()
  ): THREE.Vector3 {
    return this.getPoint(u, optionalTarget);
  }
}

/**
 * Builds the 3D helical bare copper wire coil track with real-time mechanical & electromagnetic
 * vertex perturbation as the battery passes, sheared wire end-caps, X-ray cutaway energized-turns
 * InstancedMesh, 180 animated electron flow particles (e-), and 36 axial solenoid magnetic field particles (B_bobina).
 */
export function buildHelicalCoilSystem(
  maxAnisotropy: number,
  isSoftwareGl: boolean
): HelicalCoilRig {
  const group = new THREE.Group();
  const metalBump = createBrushedMetalBumpTexture(maxAnisotropy);
  const copperAlbedo = createNaturalCopperAlbedoTexture(maxAnisotropy);

  // 1. Primary Continuous Helical Coil Mesh (Authentic drawn metallic copper C11000)
  const helixCurve = new HelicalSolenoidCurve();
  const tubularSegments = isSoftwareGl ? 3800 : 6800;
  const radialSegments = isSoftwareGl ? 7 : 10;
  const coilGeo = new THREE.TubeGeometry(
    helixCurve,
    tubularSegments,
    WIRE_RADIUS_UNITS * 0.86,
    radialSegments,
    false
  );

  // Precompute per-ring basis frames for real-time physical coil vertex perturbation when the train passes
  const coilPosAttr = coilGeo.attributes.position as THREE.BufferAttribute;
  const restCoilPos = new Float32Array(coilPosAttr.array);
  const ringCount = tubularSegments + 1;
  const ringVerts = radialSegments + 1;

  const ringU = new Float32Array(ringCount);
  const ringSinA = new Float32Array(ringCount);
  const ringRadX = new Float32Array(ringCount);
  const ringRadY = new Float32Array(ringCount);
  const ringRadZ = new Float32Array(ringCount);
  const ringTanX = new Float32Array(ringCount);
  const ringTanY = new Float32Array(ringCount);
  const ringTanZ = new Float32Array(ringCount);
  const ringBinX = new Float32Array(ringCount);
  const ringBinY = new Float32Array(ringCount);
  const ringBinZ = new Float32Array(ringCount);
  const ringNormX = new Float32Array(ringCount);
  const ringNormY = new Float32Array(ringCount);
  const ringNormZ = new Float32Array(ringCount);
  const ringWasPerturbed = new Uint8Array(ringCount);

  for (let s = 0; s < ringCount; s++) {
    const t = s / tubularSegments;
    const u = mapCoilFractionToTrackU(t);
    const frame = evaluateTrackFrame(u);
    const angle = t * TOTAL_COIL_TURNS * Math.PI * 2;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    ringU[s] = u;
    ringSinA[s] = sinA;

    ringRadX[s] = frame.binormal.x * cosA + frame.normal.x * sinA;
    ringRadY[s] = frame.binormal.y * cosA + frame.normal.y * sinA;
    ringRadZ[s] = frame.binormal.z * cosA + frame.normal.z * sinA;

    ringTanX[s] = frame.tangent.x;
    ringTanY[s] = frame.tangent.y;
    ringTanZ[s] = frame.tangent.z;

    ringBinX[s] = frame.binormal.x;
    ringBinY[s] = frame.binormal.y;
    ringBinZ[s] = frame.binormal.z;

    ringNormX[s] = frame.normal.x;
    ringNormY[s] = frame.normal.y;
    ringNormZ[s] = frame.normal.z;
  }

  const coilMat = new THREE.MeshPhysicalMaterial({
    map: copperAlbedo,
    color: 0xe8b298,
    metalness: 0.9,
    roughness: 0.21,
    clearcoat: 0.45,
    clearcoatRoughness: 0.15,
    bumpMap: metalBump,
    bumpScale: 0.0022,
  });

  const coilMesh = new THREE.Mesh(coilGeo, coilMat);
  coilMesh.castShadow = !isSoftwareGl;
  coilMesh.receiveShadow = !isSoftwareGl;
  coilMesh.frustumCulled = false;
  group.add(coilMesh);

  // 1b. Sheared Copper Wire Cross-Section End-Caps at both open mouths of the helical coil (t = 0 and t = 1)
  const wireCapGeo = new THREE.CircleGeometry(WIRE_RADIUS_UNITS * 0.86, 14);
  const wireCapMat = new THREE.MeshPhysicalMaterial({
    color: 0xf0c2a8,
    metalness: 0.94,
    roughness: 0.14,
  });
  const endCapMeshes: { mesh: THREE.Mesh; restPos: THREE.Vector3; ringIdx: number }[] = [];
  for (const endT of [0, 1]) {
    const capPos = helixCurve.getPoint(endT);
    const adjPos = helixCurve.getPoint(endT === 0 ? 0.0005 : 0.9995);
    const capDir = new THREE.Vector3().subVectors(capPos, adjPos).normalize();
    const capMesh = new THREE.Mesh(wireCapGeo, wireCapMat);
    capMesh.position.copy(capPos);
    capMesh.lookAt(capPos.clone().add(capDir));
    group.add(capMesh);
    endCapMeshes.push({
      mesh: capMesh,
      restPos: capPos.clone(),
      ringIdx: endT === 0 ? 0 : tubularSegments,
    });
  }

  // 2. Dynamic Energized X-Ray Conductor Sheath (240° arc, semi-translucent copper so internal electrons e- are visible)
  const singleTurnGeo = new THREE.TorusGeometry(
    COIL_RADIUS_UNITS * 1.008,
    WIRE_RADIUS_UNITS * 1.04,
    8,
    isSoftwareGl ? 28 : 48,
    Math.PI * 1.34
  );
  singleTurnGeo.rotateZ(Math.PI * 0.83);

  const activeTurnMat = new THREE.MeshStandardMaterial({
    color: 0xd98a64,
    emissive: 0x8f3d14,
    emissiveIntensity: 0.42,
    metalness: 0.86,
    roughness: 0.18,
    transparent: true,
    opacity: 0.46,
  });

  const activeTurnsInstanced = new THREE.InstancedMesh(
    singleTurnGeo,
    activeTurnMat,
    TOTAL_COIL_TURNS
  );
  activeTurnsInstanced.frustumCulled = false;

  const turnUValues: number[] = [];
  const thermalLevels = new Float32Array(TOTAL_COIL_TURNS);

  for (let i = 0; i < TOTAL_COIL_TURNS; i++) {
    const u = getTurnNormalizedU(i);
    turnUValues.push(u);
    _dummyObj.position.set(0, -10, 0);
    _dummyObj.scale.copy(_inactiveScale);
    _dummyObj.updateMatrix();
    activeTurnsInstanced.setMatrixAt(i, _dummyObj.matrix);
    activeTurnsInstanced.setColorAt(i, new THREE.Color(0xd98a64));
  }
  activeTurnsInstanced.instanceMatrix.needsUpdate = true;
  if (activeTurnsInstanced.instanceColor) {
    activeTurnsInstanced.instanceColor.needsUpdate = true;
  }
  group.add(activeTurnsInstanced);

  // 3. Animated Electron Flow Particles (e-) racing helically inside the energized copper wire conductor
  const particleCount = 180;
  const particleGeo = new THREE.SphereGeometry(WIRE_RADIUS_UNITS * 0.68, 10, 10);
  const particleMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
  });
  const currentParticles = new THREE.InstancedMesh(
    particleGeo,
    particleMat,
    particleCount
  );
  currentParticles.frustumCulled = false;
  for (let p = 0; p < particleCount; p++) {
    currentParticles.setColorAt(
      p,
      new THREE.Color(p % 3 === 0 ? 0xe0f2fe : p % 3 === 1 ? 0x38bdf8 : 0x7dd3fc)
    );
  }
  if (currentParticles.instanceColor) {
    currentParticles.instanceColor.needsUpdate = true;
  }
  group.add(currentParticles);

  // 4. 36 Axial Solenoid Magnetic Field Particles (B_bobina) flowing smoothly along the core of the energized coil
  const coreParticleCount = 36;
  const coreParticleGeo = new THREE.ConeGeometry(0.0085, 0.026, 8);
  coreParticleGeo.rotateX(Math.PI / 2);
  const coreParticleMat = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.85,
  });
  const solenoidCoreParticles = new THREE.InstancedMesh(
    coreParticleGeo,
    coreParticleMat,
    coreParticleCount
  );
  solenoidCoreParticles.layers.set(1);
  solenoidCoreParticles.frustumCulled = false;
  for (let cp = 0; cp < coreParticleCount; cp++) {
    solenoidCoreParticles.setColorAt(
      cp,
      new THREE.Color(cp % 2 === 0 ? 0xfbbf24 : 0xfde047)
    );
  }
  if (solenoidCoreParticles.instanceColor) {
    solenoidCoreParticles.instanceColor.needsUpdate = true;
  }
  group.add(solenoidCoreParticles);

  const rig: HelicalCoilRig = {
    group,
    coilMesh,
    activeTurnsInstanced,
    currentParticles,
    solenoidCoreParticles,
    turnUValues,
    thermalLevels,
    maxPerturbationMm: 0,
    updateActiveTurns: () => {},
  };

  const updateActiveTurns = (
    state: ElectromagneticTrainState,
    elapsedTime: number,
    dt = 1 / 60,
    showElectronFlow = true
  ) => {
    const [startTurn, endTurn] = state.activeTurnRange;
    const coolingFactor = Math.exp(-3.4 * Math.max(dt, 0.005));
    const trackLen = getTrackArcLength();

    // 1. Real-time physical vertex perturbation of the 3D continuous helical copper coil (coilMesh)
    const posArr = coilPosAttr.array as Float32Array;
    let maxDispUnits = 0;
    let anyVertexModified = false;

    for (let s = 0; s < ringCount; s++) {
      const u = ringU[s];
      const approxDist = Math.abs(circularDeltaU(u, state.u)) * trackLen;

      if (approxDist > 1.55) {
        if (ringWasPerturbed[s] === 1) {
          const baseIdx = s * ringVerts * 3;
          const endIdx = baseIdx + ringVerts * 3;
          for (let k = baseIdx; k < endIdx; k++) {
            posArr[k] = restCoilPos[k];
          }
          ringWasPerturbed[s] = 0;
          anyVertexModified = true;
        }
        continue;
      }

      const pert = computeCoilTurnPerturbation(u, state, elapsedTime);
      if (pert.intensity <= 0.0005) {
        if (ringWasPerturbed[s] === 1) {
          const baseIdx = s * ringVerts * 3;
          const endIdx = baseIdx + ringVerts * 3;
          for (let k = baseIdx; k < endIdx; k++) {
            posArr[k] = restCoilPos[k];
          }
          ringWasPerturbed[s] = 0;
          anyVertexModified = true;
        }
        continue;
      }

      // Upper arch of each turn flexes freely while the bottom wire contact against the dark slate bench stays grounded
      const archFactor = Math.max(0, 0.55 + 0.45 * ringSinA[s]);
      const radStep = pert.radialDelta * (0.35 + 0.65 * archFactor);
      const vertStep = pert.verticalArchDelta * archFactor;

      const dx =
        ringRadX[s] * radStep +
        ringTanX[s] * pert.axialDelta +
        ringBinX[s] * pert.lateralDelta +
        ringNormX[s] * vertStep;
      const dy =
        ringRadY[s] * radStep +
        ringTanY[s] * pert.axialDelta +
        ringBinY[s] * pert.lateralDelta +
        ringNormY[s] * vertStep;
      const dz =
        ringRadZ[s] * radStep +
        ringTanZ[s] * pert.axialDelta +
        ringBinZ[s] * pert.lateralDelta +
        ringNormZ[s] * vertStep;

      const dispMag = Math.hypot(dx, dy, dz);
      if (dispMag > maxDispUnits) {
        maxDispUnits = dispMag;
      }

      const baseIdx = s * ringVerts * 3;
      for (let r = 0; r < ringVerts; r++) {
        const i3 = baseIdx + r * 3;
        posArr[i3] = restCoilPos[i3] + dx;
        posArr[i3 + 1] = Math.max(
          WIRE_RADIUS_UNITS * 0.85,
          restCoilPos[i3 + 1] + dy
        );
        posArr[i3 + 2] = restCoilPos[i3 + 2] + dz;
      }
      ringWasPerturbed[s] = 1;
      anyVertexModified = true;
    }

    if (anyVertexModified) {
      coilPosAttr.needsUpdate = true;
    }
    // 1 world unit = 100 mm
    rig.maxPerturbationMm = Number((maxDispUnits * 100).toFixed(2));

    // Synchronize the two sheared copper wire end-caps at the coil mouths next to the regleta
    for (const cap of endCapMeshes) {
      const s = cap.ringIdx;
      const pert = computeCoilTurnPerturbation(ringU[s], state, elapsedTime);
      cap.mesh.position.set(
        cap.restPos.x +
          ringRadX[s] * pert.radialDelta +
          ringTanX[s] * pert.axialDelta +
          ringBinX[s] * pert.lateralDelta,
        Math.max(
          WIRE_RADIUS_UNITS,
          cap.restPos.y +
            ringRadY[s] * pert.radialDelta +
            ringNormY[s] * pert.verticalArchDelta
        ),
        cap.restPos.z +
          ringRadZ[s] * pert.radialDelta +
          ringTanZ[s] * pert.axialDelta +
          ringBinZ[s] * pert.lateralDelta
      );
    }

    // 2. Update energized X-ray conductor sheath turns in lockstep with the perturbed copper coil
    activeTurnMat.opacity = showElectronFlow ? 0.44 : 0.78;
    activeTurnMat.emissiveIntensity = showElectronFlow ? 0.38 : 0.25;

    for (let i = 0; i < TOTAL_COIL_TURNS; i++) {
      const isCurrentlyBridged =
        startTurn >= 0 &&
        i >= startTurn &&
        i <= endTurn &&
        state.currentAmps > 0.1;

      if (isCurrentlyBridged) {
        thermalLevels[i] = 1.0;
      } else {
        thermalLevels[i] *= coolingFactor;
      }

      const thermal = thermalLevels[i];
      if (isCurrentlyBridged || thermal > 0.08) {
        const u = turnUValues[i];
        const frame = evaluateTrackFrame(u);
        const pert = computeCoilTurnPerturbation(u, state, elapsedTime);

        const radialScale =
          1.0 + Math.max(0, pert.radialDelta) / COIL_RADIUS_UNITS;

        _dummyObj.position
          .copy(frame.position)
          .addScaledVector(frame.tangent, pert.axialDelta)
          .addScaledVector(frame.binormal, pert.lateralDelta)
          .addScaledVector(frame.normal, pert.verticalArchDelta * 0.5);
        _scratchLook.copy(_dummyObj.position).add(frame.tangent);
        _dummyObj.lookAt(_scratchLook);

        const thicknessScale = isCurrentlyBridged
          ? 1.04
          : 0.65 + 0.35 * thermal;
        _dummyObj.scale.set(radialScale, radialScale, thicknessScale);
        _dummyObj.updateMatrix();
        activeTurnsInstanced.setMatrixAt(i, _dummyObj.matrix);

        if (isCurrentlyBridged) {
          const normPos = (i - startTurn) / Math.max(1, endTurn - startTurn);
          const endProximity = Math.abs(normPos - 0.5) * 2;
          _scratchColor.setHSL(
            0.058 + endProximity * 0.025,
            0.66,
            0.56 + endProximity * 0.14
          );
        } else {
          _scratchColor.setHSL(0.048, 0.55, 0.42 + 0.14 * thermal);
        }
        activeTurnsInstanced.setColorAt(i, _scratchColor);
      } else {
        _dummyObj.position.set(0, -10, 0);
        _dummyObj.scale.copy(_inactiveScale);
        _dummyObj.updateMatrix();
        activeTurnsInstanced.setMatrixAt(i, _dummyObj.matrix);
      }
    }
    activeTurnsInstanced.instanceMatrix.needsUpdate = true;
    if (activeTurnsInstanced.instanceColor) {
      activeTurnsInstanced.instanceColor.needsUpdate = true;
    }

    // 3. Update 180 animated electron flow particles (e-) along the perturbed copper wire turns [startTurn..endTurn]
    const hasBridgedTurns =
      showElectronFlow &&
      state.currentAmps > 0.1 &&
      startTurn >= 0 &&
      endTurn > startTurn;
    const tStart = hasBridgedTurns ? startTurn / TOTAL_COIL_TURNS : 0;
    const tSpan = hasBridgedTurns
      ? (endTurn - startTurn) / TOTAL_COIL_TURNS
      : 0;

    for (let p = 0; p < particleCount; p++) {
      if (!hasBridgedTurns) {
        _dummyObj.position.set(0, -10, 0);
        _dummyObj.scale.copy(_inactiveScale);
        _dummyObj.updateMatrix();
        currentParticles.setMatrixAt(p, _dummyObj.matrix);
        continue;
      }

      const rawPhase =
        ((p / particleCount) + elapsedTime * 0.68 * state.direction) % 1;
      const phase = rawPhase < 0 ? rawPhase + 1 : rawPhase;
      const coilT = THREE.MathUtils.clamp(tStart + phase * tSpan, 0, 1);

      helixCurve.getPoint(coilT, _scratchPos);
      const nextCoilT = THREE.MathUtils.clamp(
        coilT + 0.0004 * state.direction,
        0,
        1
      );
      helixCurve.getPoint(nextCoilT, _scratchLook);

      const sIdx = Math.min(
        tubularSegments,
        Math.round(coilT * tubularSegments)
      );
      const pert = computeCoilTurnPerturbation(ringU[sIdx], state, elapsedTime);
      _scratchPos.x +=
        ringRadX[sIdx] * pert.radialDelta * 0.7 +
        ringTanX[sIdx] * pert.axialDelta +
        ringBinX[sIdx] * pert.lateralDelta;
      _scratchPos.y +=
        ringRadY[sIdx] * pert.radialDelta * 0.7 +
        ringNormY[sIdx] * pert.verticalArchDelta * 0.5;
      _scratchPos.z +=
        ringRadZ[sIdx] * pert.radialDelta * 0.7 +
        ringTanZ[sIdx] * pert.axialDelta +
        ringBinZ[sIdx] * pert.lateralDelta;

      _dummyObj.position.copy(_scratchPos);
      _dummyObj.lookAt(_scratchLook);
      const pulseScale = 0.82 + 0.25 * Math.sin(elapsedTime * 10 + p * 0.9);
      _dummyObj.scale.set(
        pulseScale * 0.85,
        pulseScale * 0.85,
        pulseScale * 1.85
      );
      _dummyObj.updateMatrix();
      currentParticles.setMatrixAt(p, _dummyObj.matrix);
    }
    currentParticles.instanceMatrix.needsUpdate = true;

    // 4. Update 36 Axial Solenoid Magnetic Field Particles (B_bobina) streaming strictly inside the energized coil turns
    const startU = hasBridgedTurns ? turnUValues[startTurn] : state.u;
    const spanU = hasBridgedTurns
      ? Math.max(0.005, turnUValues[endTurn] - turnUValues[startTurn])
      : 0.01;

    for (let cp = 0; cp < coreParticleCount; cp++) {
      if (!hasBridgedTurns) {
        _dummyObj.position.set(0, -10, 0);
        _dummyObj.scale.copy(_inactiveScale);
        _dummyObj.updateMatrix();
        solenoidCoreParticles.setMatrixAt(cp, _dummyObj.matrix);
        continue;
      }

      const streamIndex = cp % 3;
      const rawCorePhase =
        ((Math.floor(cp / 3) / 12) + elapsedTime * 0.55 * state.direction) % 1;
      const corePhase = rawCorePhase < 0 ? rawCorePhase + 1 : rawCorePhase;
      const coreU = wrapU(startU + corePhase * spanU);
      const frame = evaluateTrackFrame(coreU);

      const binormalOffset = (streamIndex - 1) * (COIL_RADIUS_UNITS * 0.42);
      const normalOffset = COIL_RADIUS_UNITS * 0.68;

      _scratchPos
        .copy(frame.position)
        .addScaledVector(frame.binormal, binormalOffset)
        .addScaledVector(frame.normal, normalOffset);
      _scratchLook
        .copy(_scratchPos)
        .addScaledVector(frame.tangent, state.direction);

      _dummyObj.position.copy(_scratchPos);
      _dummyObj.lookAt(_scratchLook);
      _dummyObj.scale.setScalar(0.88 + 0.2 * Math.sin(elapsedTime * 8 + cp));
      _dummyObj.updateMatrix();
      solenoidCoreParticles.setMatrixAt(cp, _dummyObj.matrix);
    }
    solenoidCoreParticles.instanceMatrix.needsUpdate = true;
  };

  rig.updateActiveTurns = updateActiveTurns;
  return rig;
}


/**
 * Helper to create a 3D leader line from a component anchor point up to its 3D callout sprite.
 */
function createLeaderLine(
  start: THREE.Vector3,
  end: THREE.Vector3,
  colorHex: number
): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
  const mat = new THREE.LineBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.78,
  });
  return new THREE.Line(geo, mat);
}

/**
 * Builds the composite 12-part Electromagnetic Train (AA Battery + Stacked Neodymium Magnets +
 * Laser-Engraved N/S Pole Caps + Sliding Brush Micro-Sparks + 3D Magnetic Flux Loops B + Lorentz Force Vector F).
 */
export function buildElectromagneticTrainRig(
  maxAnisotropy: number,
  isSoftwareGl: boolean
): TrainRig {
  const rootGroup = new THREE.Group();
  const batteryGroup = new THREE.Group();
  const frontMagnetGroup = new THREE.Group();
  const rearMagnetGroup = new THREE.Group();
  const physicsOverlayGroup = new THREE.Group();

  rootGroup.add(batteryGroup);
  rootGroup.add(frontMagnetGroup);
  rootGroup.add(rearMagnetGroup);
  rootGroup.add(physicsOverlayGroup);

  const batteryTex = createBatteryWrapperTexture(maxAnisotropy);
  const batteryRoughTex = createBatteryRoughnessMap(maxAnisotropy);
  const northCapTex = createMagnetPoleCapTexture('N', maxAnisotropy);
  const southCapTex = createMagnetPoleCapTexture('S', maxAnisotropy);
  const metalBump = createBrushedMetalBumpTexture(maxAnisotropy);
  const softShadowTex = createSoftShadowTexture();

  const batteryRadius = COIL_RADIUS_UNITS * 0.56; // ~0.0756 units
  const batteryLength = TRAIN_BATTERY_LENGTH_UNITS; // 0.42 units
  const magnetRadius = COIL_RADIUS_UNITS * 0.64; // ~0.0864 units
  const singleMagnetThickness = 0.026; // 4 discs = 0.104 units total stack length
  const coilInnerRadius = COIL_RADIUS_UNITS - WIRE_RADIUS_UNITS * 0.9; // ~0.12375 units
  // Physical gravity offset: Neodymium magnets rest on the bottom inner copper turns (-0.0317 units)
  // Max radial extent of magnet = 0.0317 + 0.0864 = 0.1181 < coilInnerRadius (0.12375) -> 100% inside copper tunnel!
  const gravityNormalOffset = -(coilInnerRadius - magnetRadius) * 0.85;

  // Dynamic contact shadow plane gliding on the table right beneath the train
  const trainShadowGeo = new THREE.PlaneGeometry(0.95, 0.36);
  trainShadowGeo.rotateX(-Math.PI / 2);
  trainShadowGeo.rotateY(Math.PI / 2);
  const trainShadowMat = new THREE.MeshBasicMaterial({
    map: softShadowTex,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
  });
  const trainShadowMesh = new THREE.Mesh(trainShadowGeo, trainShadowMat);
  rootGroup.add(trainShadowMesh);

  // --- PART 1: Spine-Conformal AA Battery Hull (28 axial rings so it conforms smoothly to coil curvature with zero wall clipping) ---
  const hullGeo = new THREE.CylinderGeometry(
    batteryRadius,
    batteryRadius,
    batteryLength,
    36,
    28,
    false
  );
  hullGeo.rotateX(Math.PI / 2);

  const restHullPos = new Float32Array(
    (hullGeo.attributes.position as THREE.BufferAttribute).array
  );
  const restHullNorm = new Float32Array(
    (hullGeo.attributes.normal as THREE.BufferAttribute).array
  );

  const wrapperMat = new THREE.MeshPhysicalMaterial({
    map: batteryTex,
    roughnessMap: batteryRoughTex,
    color: 0xffffff,
    roughness: 0.75,
    metalness: 0.48,
    clearcoat: 0.72,
    clearcoatRoughness: 0.12,
  });
  const crimpSteelMat = new THREE.MeshPhysicalMaterial({
    color: 0xe2e8f0,
    metalness: 0.95,
    roughness: 0.16,
    bumpMap: metalBump,
    bumpScale: 0.002,
  });

  const hullMesh = new THREE.Mesh(hullGeo, [
    wrapperMat,
    crimpSteelMat,
    crimpSteelMat,
  ]);
  hullMesh.castShadow = !isSoftwareGl;
  hullMesh.frustumCulled = false;
  batteryGroup.add(hullMesh);

  // --- PART 2 & 3: Metallic Crimped Shoulder Rings, Insulator Washer & Positive Button Terminal ---
  const shoulderGeo = new THREE.TorusGeometry(
    batteryRadius * 0.96,
    0.0065,
    12,
    32
  );
  const frontShoulder = new THREE.Mesh(shoulderGeo, crimpSteelMat);
  batteryGroup.add(frontShoulder);

  const rearShoulder = new THREE.Mesh(shoulderGeo, crimpSteelMat);
  batteryGroup.add(rearShoulder);

  // Dark dielectric fiber insulator ring under positive terminal button
  const insulatorMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.72,
    metalness: 0.05,
  });
  const insulatorGeo = new THREE.RingGeometry(
    batteryRadius * 0.39,
    batteryRadius * 0.88,
    28
  );
  const frontInsulatorMesh = new THREE.Mesh(insulatorGeo, insulatorMat);
  batteryGroup.add(frontInsulatorMesh);

  const posButtonGeo = new THREE.CylinderGeometry(
    batteryRadius * 0.36,
    batteryRadius * 0.4,
    0.028,
    24
  );
  posButtonGeo.rotateX(Math.PI / 2);
  const posButtonMesh = new THREE.Mesh(posButtonGeo, crimpSteelMat);
  batteryGroup.add(posButtonMesh);

  // --- PART 4 & 5: Front & Rear Neodymium (NdFeB N52) Magnet Stacks with Engraved N/S Pole Caps ---
  const nickelMagnetMat = new THREE.MeshPhysicalMaterial({
    color: 0xf1f5f9,
    metalness: 0.97,
    roughness: 0.09,
    clearcoat: 0.85,
    clearcoatRoughness: 0.07,
    bumpMap: metalBump,
    bumpScale: 0.0018,
  });

  const northCapMat = new THREE.MeshStandardMaterial({
    map: northCapTex,
    roughness: 0.22,
    metalness: 0.78,
  });
  const southCapMat = new THREE.MeshStandardMaterial({
    map: southCapTex,
    roughness: 0.22,
    metalness: 0.78,
  });

  const northPoleMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0xef4444,
    emissiveIntensity: 0.45,
    roughness: 0.25,
    metalness: 0.35,
  });
  const southPoleMat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    emissive: 0x3b82f6,
    emissiveIntensity: 0.45,
    roughness: 0.25,
    metalness: 0.35,
  });

  const discGeo = new THREE.CylinderGeometry(
    magnetRadius,
    magnetRadius,
    singleMagnetThickness * 0.9,
    32
  );
  discGeo.rotateX(Math.PI / 2);

  // Machined bevel ring between adjacent neodymium discs
  const bevelRingGeo = new THREE.TorusGeometry(
    magnetRadius * 0.985,
    0.0028,
    8,
    32
  );

  // Copper-beryllium sliding wear band on outer circumference of magnet discs
  const wearBandGeo = new THREE.TorusGeometry(
    magnetRadius * 1.003,
    0.0016,
    8,
    32
  );
  const copperWearMat = new THREE.MeshStandardMaterial({
    color: 0xd98a64,
    metalness: 0.92,
    roughness: 0.18,
  });

  const poleBandGeo = new THREE.CylinderGeometry(
    magnetRadius * 1.012,
    magnetRadius * 1.012,
    singleMagnetThickness * 0.38,
    32,
    1,
    true
  );
  poleBandGeo.rotateX(Math.PI / 2);

  for (let m = 0; m < 4; m++) {
    const zOffset = (m - 1.5) * singleMagnetThickness;
    const fDisc = new THREE.Mesh(discGeo, [
      nickelMagnetMat,
      northCapMat,
      southCapMat,
    ]);
    fDisc.position.z = zOffset;
    fDisc.castShadow = !isSoftwareGl;
    frontMagnetGroup.add(fDisc);

    const fBevel = new THREE.Mesh(bevelRingGeo, nickelMagnetMat);
    fBevel.position.z = zOffset + singleMagnetThickness * 0.45;
    frontMagnetGroup.add(fBevel);

    const fWear = new THREE.Mesh(wearBandGeo, copperWearMat);
    fWear.position.z = zOffset;
    frontMagnetGroup.add(fWear);

    const rDisc = new THREE.Mesh(discGeo, [
      nickelMagnetMat,
      southCapMat,
      northCapMat,
    ]);
    rDisc.position.z = zOffset;
    rDisc.castShadow = !isSoftwareGl;
    rearMagnetGroup.add(rDisc);

    const rBevel = new THREE.Mesh(bevelRingGeo, nickelMagnetMat);
    rBevel.position.z = zOffset - singleMagnetThickness * 0.45;
    rearMagnetGroup.add(rBevel);

    const rWear = new THREE.Mesh(wearBandGeo, copperWearMat);
    rWear.position.z = zOffset;
    rearMagnetGroup.add(rWear);
  }

  // Explicit North (Red) / South (Blue) pole indicator bands on the magnet stacks
  const frontNorthBand = new THREE.Mesh(poleBandGeo, northPoleMat);
  frontNorthBand.position.z = 1.65 * singleMagnetThickness;
  frontMagnetGroup.add(frontNorthBand);

  const frontSouthBand = new THREE.Mesh(poleBandGeo, southPoleMat);
  frontSouthBand.position.z = -1.65 * singleMagnetThickness;
  frontMagnetGroup.add(frontSouthBand);

  const rearNorthBand = new THREE.Mesh(poleBandGeo, northPoleMat);
  rearNorthBand.position.z = -1.65 * singleMagnetThickness;
  rearMagnetGroup.add(rearNorthBand);

  const rearSouthBand = new THREE.Mesh(poleBandGeo, southPoleMat);
  rearSouthBand.position.z = 1.65 * singleMagnetThickness;
  rearMagnetGroup.add(rearSouthBand);

  // --- PART 6: Sliding Electrical Contact Glow Rings + Dynamic Micro-Sparks ---
  const contactRingGeo = new THREE.TorusGeometry(
    magnetRadius * 1.045,
    0.0068,
    12,
    32
  );
  const contactRingMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.92,
  });
  const frontContactRing = new THREE.Mesh(contactRingGeo, contactRingMat);
  frontContactRing.layers.set(1);
  frontMagnetGroup.add(frontContactRing);

  const rearContactRing = new THREE.Mesh(
    contactRingGeo,
    contactRingMat.clone()
  );
  rearContactRing.layers.set(1);
  rearMagnetGroup.add(rearContactRing);

  // 24 Electric Micro-Spark Filaments (12 at front contact brush, 12 at rear contact brush, in world space)
  const sparkCount = 24;
  const sparkGeo = new THREE.ConeGeometry(0.0052, 0.035, 6);
  const sparkMat = new THREE.MeshBasicMaterial({
    color: 0xe0f2fe,
  });
  const sparkInstanced = new THREE.InstancedMesh(
    sparkGeo,
    sparkMat,
    sparkCount
  );
  sparkInstanced.layers.set(1);
  sparkInstanced.frustumCulled = false;
  for (let s = 0; s < sparkCount; s++) {
    sparkInstanced.setColorAt(
      s,
      new THREE.Color(s % 2 === 0 ? 0x7dd3fc : 0xfde047)
    );
  }
  if (sparkInstanced.instanceColor) {
    sparkInstanced.instanceColor.needsUpdate = true;
  }
  rootGroup.add(sparkInstanced);

  // --- PART 7: Multi-Shell Vertex-Colored 3D Analytical Dipole Magnetic Field Streamlines (B_magnet) ---
  const fieldLoopsGroup = new THREE.Group();
  const frontFieldLoopsSubgroup = new THREE.Group();
  const rearFieldLoopsSubgroup = new THREE.Group();
  frontMagnetGroup.add(frontFieldLoopsSubgroup);
  rearMagnetGroup.add(rearFieldLoopsSubgroup);

  const forceArrowGroup = new THREE.Group();
  physicsOverlayGroup.add(fieldLoopsGroup);
  physicsOverlayGroup.add(forceArrowGroup);

  const innerLoopMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
  });
  const outerLoopMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.68,
  });

  // Rotate cone +X/+Z so lookAt(+tangent) points the arrowhead cone tip along the exact streamline tangent
  const loopConeGeo = new THREE.ConeGeometry(0.012, 0.034, 12);
  loopConeGeo.rotateX(Math.PI / 2);
  const northConeMat = new THREE.MeshBasicMaterial({
    color: 0x0284c7,
    transparent: true,
    opacity: 0.95,
  });
  const southConeMat = new THREE.MeshBasicMaterial({
    color: 0x2563eb,
    transparent: true,
    opacity: 0.92,
  });

  const colorNorth = new THREE.Color(0xe11d48); // Rich crimson-rose at North pole exit
  const colorRadialPeak = new THREE.Color(0x0284c7); // Vivid deep cyan-cobalt at copper coil intersection
  const colorSouth = new THREE.Color(0x1d4ed8); // Deep royal dipole blue at South pole entry
  const tmpVertColor = new THREE.Color();

  // Store 28 3D analytical dipole streamline curves in magnet-local space (8 inner + 6 outer per stack)
  const dipoleLoopCurves: {
    curve: THREE.CatmullRomCurve3;
    phi: number;
    stackSign: number;
    shell: number;
  }[] = [];

  for (const stackSign of [-1, 1]) {
    const targetSubgroup =
      stackSign > 0 ? frontFieldLoopsSubgroup : rearFieldLoopsSubgroup;

    // Dual-shell configuration: shell 0 = high-intensity inner radial coil shell (8 loops),
    // shell 1 = wide volumetric outer dipole return shell (6 loops)
    const shells = [
      {
        shellIdx: 0,
        count: 8,
        rPeak: COIL_RADIUS_UNITS * 1.52,
        zSpan: 0.072,
        tubeRadius: 0.0033,
        phiOffset: 0,
        mat: innerLoopMat,
      },
      {
        shellIdx: 1,
        count: 6,
        rPeak: COIL_RADIUS_UNITS * 1.98,
        zSpan: 0.105,
        tubeRadius: 0.0024,
        phiOffset: Math.PI / 12,
        mat: outerLoopMat,
      },
    ];

    for (const sh of shells) {
      for (let k = 0; k < sh.count; k++) {
        const phi = (k / sh.count) * Math.PI * 2 + sh.phiOffset;
        const singleLoopGroup = new THREE.Group();
        singleLoopGroup.position.set(0, -gravityNormalOffset, 0);
        singleLoopGroup.rotation.z = phi;

        // Sample 16 analytical dipole streamline control points (N pole -> radial peak -> S pole -> core return)
        const pts: THREE.Vector3[] = [];
        const numSamples = 16;
        for (let sIdx = 0; sIdx < numSamples; sIdx++) {
          const tau = sIdx / numSamples;
          if (tau <= 0.82) {
            // Exterior dipole streamline lobe from North face (+stackSign) to South face (-stackSign)
            const uLobe = tau / 0.82; // 0..1 from N to S
            const theta = uLobe * Math.PI; // 0..pi
            const sinT = Math.sin(theta);
            const cosT = Math.cos(theta);
            // Analytical dipole profile r ~ sin^1.65(theta) with flattened polar entry
            const r =
              magnetRadius * 0.76 +
              (sh.rPeak - magnetRadius * 0.76) * Math.pow(sinT, 1.45);
            const z =
              stackSign *
              (sh.zSpan * cosT + 0.024 * Math.sin(theta * 2) * sinT);
            pts.push(new THREE.Vector3(r, 0, z));
          } else {
            // Smooth internal core closure inside the neodymium magnet stack from S back to N
            const uCore = (tau - 0.82) / 0.18;
            const zCore = stackSign * sh.zSpan * (-1 + 2 * uCore);
            const rCore = magnetRadius * (0.64 - 0.12 * Math.sin(uCore * Math.PI));
            pts.push(new THREE.Vector3(rCore, 0, zCore));
          }
        }

        const loopCurve = new THREE.CatmullRomCurve3(pts, true, 'centripetal');
        dipoleLoopCurves.push({
          curve: loopCurve,
          phi,
          stackSign,
          shell: sh.shellIdx,
        });

        const tubularSegs = 44;
        const radialSegs = 7;
        const loopGeo = new THREE.TubeGeometry(
          loopCurve,
          tubularSegs,
          sh.tubeRadius,
          radialSegs,
          true
        );

        // Apply smooth North -> Radial Cyan -> South vertex color gradient along the streamline
        const vCount = loopGeo.attributes.position.count;
        const colors = new Float32Array(vCount * 3);
        const ringVerts = radialSegs + 1;
        for (let tStep = 0; tStep <= tubularSegs; tStep++) {
          const frac = tStep / tubularSegs;
          if (frac < 0.25) {
            tmpVertColor
              .copy(colorNorth)
              .lerp(colorRadialPeak, frac / 0.25);
          } else if (frac < 0.58) {
            tmpVertColor.copy(colorRadialPeak);
          } else if (frac < 0.85) {
            tmpVertColor
              .copy(colorRadialPeak)
              .lerp(colorSouth, (frac - 0.58) / 0.27);
          } else {
            tmpVertColor
              .copy(colorSouth)
              .lerp(colorNorth, (frac - 0.85) / 0.15);
          }
          for (let rStep = 0; rStep < ringVerts; rStep++) {
            const vIdx = (tStep * ringVerts + rStep) * 3;
            if (vIdx + 2 < colors.length) {
              colors[vIdx] = tmpVertColor.r;
              colors[vIdx + 1] = tmpVertColor.g;
              colors[vIdx + 2] = tmpVertColor.b;
            }
          }
        }
        loopGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const loopMesh = new THREE.Mesh(loopGeo, sh.mat);
        loopMesh.layers.set(1);
        singleLoopGroup.add(loopMesh);

        // Place tangent-aligned directional arrowheads along the outward (N -> coil) and inward (coil -> S) branches
        const coneFractions = sh.shellIdx === 0 ? [0.22, 0.62] : [0.41];
        for (let cIdx = 0; cIdx < coneFractions.length; cIdx++) {
          const cf = coneFractions[cIdx];
          const cPos = loopCurve.getPointAt(cf);
          const cTan = loopCurve.getTangentAt(cf);
          const coneMesh = new THREE.Mesh(
            loopConeGeo,
            cIdx === 0 ? northConeMat : southConeMat
          );
          coneMesh.position.copy(cPos);
          coneMesh.lookAt(cPos.clone().add(cTan));
          if (sh.shellIdx === 1) {
            coneMesh.scale.setScalar(0.82);
          }
          coneMesh.layers.set(1);
          singleLoopGroup.add(coneMesh);
        }

        targetSubgroup.add(singleLoopGroup);
      }
    }
  }

  // 112 Animated Tangent-Aligned Magnetic Flux Needles gliding continuously along all 28 dipole loops
  const needlesPerLoop = 4;
  const fluxParticleCount = dipoleLoopCurves.length * needlesPerLoop; // 28 * 4 = 112
  const fluxParticleGeo = new THREE.ConeGeometry(0.0068, 0.028, 8);
  fluxParticleGeo.rotateX(Math.PI / 2); // Point cone tip along +Z so lookAt aligns with field tangent
  const fluxParticleMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
  });
  const magneticFluxInstanced = new THREE.InstancedMesh(
    fluxParticleGeo,
    fluxParticleMat,
    fluxParticleCount
  );
  magneticFluxInstanced.layers.set(1);
  magneticFluxInstanced.frustumCulled = false;
  for (let fp = 0; fp < fluxParticleCount; fp++) {
    const phase = fp % needlesPerLoop;
    magneticFluxInstanced.setColorAt(
      fp,
      new THREE.Color(
        phase === 0
          ? 0xfda4af
          : phase === 1
            ? 0x7dd3fc
            : phase === 2
              ? 0x38bdf8
              : 0x93c5fd
      )
    );
  }
  if (magneticFluxInstanced.instanceColor) {
    magneticFluxInstanced.instanceColor.needsUpdate = true;
  }
  rootGroup.add(magneticFluxInstanced);

  // --- PART 7B: Spine-Conformal 3D Solenoidal Core Magnetic Field Streamlines (B_bobina) ---
  // 5 dynamic curved field lines that bend along the exact curved coil spine and flare outward at the ends
  const solenoidLineCount = 5;
  const solenoidLineSegs = 28;
  const solenoidLineMat = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.78,
  });
  const solenoidHeadGeo = new THREE.ConeGeometry(0.013, 0.038, 12);
  solenoidHeadGeo.rotateX(Math.PI / 2);
  const solenoidHeadMat = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
  });

  const solenoidStreams: {
    line: THREE.Line;
    positions: Float32Array;
    cone: THREE.Mesh;
    localX: number;
    localY: number;
  }[] = [];

  const solenoidOffsets = [
    { x: 0.0, y: COIL_RADIUS_UNITS * 1.18 },
    { x: COIL_RADIUS_UNITS * 0.72, y: COIL_RADIUS_UNITS * 0.48 },
    { x: -COIL_RADIUS_UNITS * 0.72, y: COIL_RADIUS_UNITS * 0.48 },
    { x: COIL_RADIUS_UNITS * 0.56, y: -COIL_RADIUS_UNITS * 0.32 },
    { x: -COIL_RADIUS_UNITS * 0.56, y: -COIL_RADIUS_UNITS * 0.32 },
  ];

  for (let sIdx = 0; sIdx < solenoidLineCount; sIdx++) {
    const positions = new Float32Array((solenoidLineSegs + 1) * 3);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const line = new THREE.Line(lineGeo, solenoidLineMat);
    line.layers.set(1);
    line.frustumCulled = false;
    rootGroup.add(line);

    const cone = new THREE.Mesh(solenoidHeadGeo, solenoidHeadMat);
    cone.layers.set(1);
    cone.frustumCulled = false;
    rootGroup.add(cone);

    solenoidStreams.push({
      line,
      positions,
      cone,
      localX: solenoidOffsets[sIdx].x,
      localY: solenoidOffsets[sIdx].y,
    });
  }

  // Create Explicit 3D Propulsion Force Vector (F_Lorentz)
  const directionalVectorsSubgroup = new THREE.Group();
  forceArrowGroup.add(directionalVectorsSubgroup);

  const forceMat = new THREE.MeshBasicMaterial({
    color: 0x10b981,
  });
  const forceShaftGeo = new THREE.CylinderGeometry(0.013, 0.013, 0.34, 16);
  forceShaftGeo.rotateX(Math.PI / 2);
  const forceShaft = new THREE.Mesh(forceShaftGeo, forceMat);
  forceShaft.position.set(0, COIL_RADIUS_UNITS * 1.65, 0.38);
  directionalVectorsSubgroup.add(forceShaft);

  const forceHeadGeo = new THREE.ConeGeometry(0.034, 0.09, 18);
  forceHeadGeo.rotateX(Math.PI / 2);
  const forceHead = new THREE.Mesh(forceHeadGeo, forceMat);
  forceHead.position.set(0, COIL_RADIUS_UNITS * 1.65, 0.58);
  directionalVectorsSubgroup.add(forceHead);

  // Solenoid Axial Magnetic Field Vector (B_bobina) primary indicator arrow
  const bFieldMat = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
  });
  const bShaftGeo = new THREE.CylinderGeometry(0.0095, 0.0095, 0.38, 12);
  bShaftGeo.rotateX(Math.PI / 2);
  const bShaft = new THREE.Mesh(bShaftGeo, bFieldMat);
  bShaft.position.set(0, COIL_RADIUS_UNITS * 1.25, 0);
  directionalVectorsSubgroup.add(bShaft);

  const bHeadGeo = new THREE.ConeGeometry(0.026, 0.07, 14);
  bHeadGeo.rotateX(Math.PI / 2);
  const bHead = new THREE.Mesh(bHeadGeo, bFieldMat);
  bHead.position.set(0, COIL_RADIUS_UNITS * 1.25, 0.22);
  directionalVectorsSubgroup.add(bHead);

  // --- PART 8: Staggered, Non-Overlapping 3D World-Space Callout Sprites with Leader Lines (Layer 1: External Views) ---
  const forceCalloutPos = new THREE.Vector3(
    0.06,
    COIL_RADIUS_UNITS * 3.95,
    0.72
  );
  const forceCallout = createDiegeticCalloutSprite(
    'F',
    'Fuerza de Lorentz (Propulsión)',
    'F = I · L × B_radial  ->  Impulso',
    '#10b981'
  );
  forceCallout.position.copy(forceCalloutPos);
  forceCallout.layers.set(1);
  forceArrowGroup.add(forceCallout);
  const forceLeader = createLeaderLine(
    new THREE.Vector3(0, COIL_RADIUS_UNITS * 1.65, 0.48),
    new THREE.Vector3(
      forceCalloutPos.x,
      forceCalloutPos.y - 0.055,
      forceCalloutPos.z
    ),
    0x10b981
  );
  forceLeader.layers.set(1);
  forceArrowGroup.add(forceLeader);

  const currentCalloutPos = new THREE.Vector3(
    -0.02,
    COIL_RADIUS_UNITS * 2.75,
    -0.02
  );
  const currentCallout = createDiegeticCalloutSprite(
    'e-',
    'Flujo de Electrones en el Cobre',
    'Partículas e- (- a +1.5V) y Campo B',
    '#38bdf8'
  );
  currentCallout.position.copy(currentCalloutPos);
  currentCallout.layers.set(1);
  forceArrowGroup.add(currentCallout);
  const currentLeader = createLeaderLine(
    new THREE.Vector3(0, COIL_RADIUS_UNITS * 1.25, 0.0),
    new THREE.Vector3(
      currentCalloutPos.x,
      currentCalloutPos.y - 0.055,
      currentCalloutPos.z
    ),
    0x38bdf8
  );
  currentLeader.layers.set(1);
  forceArrowGroup.add(currentLeader);

  const polesCalloutPos = new THREE.Vector3(
    -0.08,
    COIL_RADIUS_UNITS * 2.05,
    -0.62
  );
  const polesCallout = createDiegeticCalloutSprite(
    'N·S',
    'Campo Magnético Dipolar NdFeB',
    'Doble capa 3D de líneas B (N -> S)',
    '#3b82f6'
  );
  polesCallout.position.copy(polesCalloutPos);
  polesCallout.layers.set(1);
  forceArrowGroup.add(polesCallout);
  const polesLeader = createLeaderLine(
    new THREE.Vector3(0, COIL_RADIUS_UNITS * 0.9, -0.28),
    new THREE.Vector3(
      polesCalloutPos.x,
      polesCalloutPos.y - 0.055,
      polesCalloutPos.z
    ),
    0x3b82f6
  );
  polesLeader.layers.set(1);
  forceArrowGroup.add(polesLeader);

  // Move external high-mounted arrows to layer 1 so POV inside tunnel has an unobstructed view
  directionalVectorsSubgroup.traverse((child) => {
    child.layers.set(1);
  });

  // --- PART 9: First-Person POV Interior Tunnel Interaction Rig (Layer 2: Inside-the-Coil POV) ---
  const tunnelCenterY = -gravityNormalOffset;
  const povInteriorGroup = new THREE.Group();
  frontMagnetGroup.add(povInteriorGroup);

  // Warm natural copper interior PointLight illuminating the inside of the bare copper wire turns ahead
  const interiorLight = new THREE.PointLight(0xf6ad85, 1.85, 1.6, 1.25);
  interiorLight.position.set(0, tunnelCenterY + 0.025, 0.18);
  frontMagnetGroup.add(interiorLight);

  // Secondary cool electric-arc fill light near the bottom sliding brush contact
  const brushArcLight = new THREE.PointLight(0x38bdf8, 1.35, 0.65, 1.4);
  brushArcLight.position.set(0, -magnetRadius * 0.72, 0.09);
  frontMagnetGroup.add(brushArcLight);

  // 8 Radial Magnetic Flux Lines (B_radial) shooting outward from the leading NdFeB magnet into the copper coil turns ahead of the POV lens
  const radialBMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.88,
  });
  const radialShaftGeo = new THREE.CylinderGeometry(0.0018, 0.0018, 0.036, 10);
  const radialConeGeo = new THREE.ConeGeometry(0.005, 0.014, 12);

  for (let rIdx = 0; rIdx < 8; rIdx++) {
    const theta = (rIdx / 8) * Math.PI * 2;
    const radialArrow = new THREE.Group();
    radialArrow.position.set(0, tunnelCenterY, 0.16);
    radialArrow.rotation.z = theta;

    const rShaft = new THREE.Mesh(radialShaftGeo, radialBMat);
    rShaft.position.set(0, COIL_RADIUS_UNITS * 0.72, 0);
    rShaft.layers.set(2);
    radialArrow.add(rShaft);

    const rCone = new THREE.Mesh(radialConeGeo, radialBMat);
    rCone.position.set(0, COIL_RADIUS_UNITS * 0.9, 0);
    rCone.layers.set(2);
    radialArrow.add(rCone);

    povInteriorGroup.add(radialArrow);
  }

  // Smooth 360-degree Semi-Translucent Natural Copper Turns ahead of the POV camera so internal electrons e- are visible
  const povTurnRingGeo = new THREE.TorusGeometry(
    COIL_RADIUS_UNITS * 0.97,
    WIRE_RADIUS_UNITS * 0.45,
    10,
    isSoftwareGl ? 36 : 64
  );
  const povTurnRingMat = new THREE.MeshStandardMaterial({
    color: 0xd48c6a,
    emissive: 0x7c3410,
    emissiveIntensity: 0.32,
    metalness: 0.88,
    roughness: 0.18,
    transparent: true,
    opacity: 0.34,
  });
  for (let ringIdx = 0; ringIdx <= 8; ringIdx++) {
    const tRing = new THREE.Mesh(povTurnRingGeo, povTurnRingMat);
    tRing.position.set(0, tunnelCenterY, 0.14 + ringIdx * 0.026);
    tRing.layers.set(2);
    povInteriorGroup.add(tRing);
  }

  // 45 Animated POV Electron Flow Particles (e-) racing around the interior of the copper wire turns in front of the POV lens
  const povElectronCount = 45;
  const povElectronGeo = new THREE.SphereGeometry(0.0046, 8, 8);
  const povElectronMat = new THREE.MeshBasicMaterial({
    color: 0x7dd3fc,
  });
  const povElectronsInstanced = new THREE.InstancedMesh(
    povElectronGeo,
    povElectronMat,
    povElectronCount
  );
  povElectronsInstanced.layers.set(2);
  povElectronsInstanced.frustumCulled = false;
  for (let pe = 0; pe < povElectronCount; pe++) {
    povElectronsInstanced.setColorAt(
      pe,
      new THREE.Color(pe % 2 === 0 ? 0xe0f2fe : 0x38bdf8)
    );
  }
  if (povElectronsInstanced.instanceColor) {
    povElectronsInstanced.instanceColor.needsUpdate = true;
  }
  povInteriorGroup.add(povElectronsInstanced);

  // Bottom Copper Sliding Contact Wear Strip & Electric Arc Bridge where the magnet brushes the bottom rail
  const contactBridgeGeo = new THREE.CylinderGeometry(
    0.007,
    0.007,
    0.22,
    14
  );
  contactBridgeGeo.rotateX(Math.PI / 2);
  const contactBridgeMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.82,
  });
  const contactBridgeMesh = new THREE.Mesh(contactBridgeGeo, contactBridgeMat);
  contactBridgeMesh.position.set(0, -magnetRadius * 0.94, 0.18);
  contactBridgeMesh.layers.set(2);
  povInteriorGroup.add(contactBridgeMesh);

  // Forward Axial Lorentz Force Vector (F_Lorentz) inside the center of the copper tunnel
  const povForceShaftGeo = new THREE.CylinderGeometry(0.004, 0.004, 0.12, 14);
  povForceShaftGeo.rotateX(Math.PI / 2);
  const povForceShaft = new THREE.Mesh(povForceShaftGeo, forceMat);
  povForceShaft.position.set(0, tunnelCenterY + 0.002, 0.24);
  povForceShaft.layers.set(2);
  povInteriorGroup.add(povForceShaft);

  const povForceHeadGeo = new THREE.ConeGeometry(0.0105, 0.03, 16);
  povForceHeadGeo.rotateX(Math.PI / 2);
  const povForceHead = new THREE.Mesh(povForceHeadGeo, forceMat);
  povForceHead.position.set(0, tunnelCenterY + 0.002, 0.31);
  povForceHead.layers.set(2);
  povInteriorGroup.add(povForceHead);

  // Micro-scale 3D POV Labels inside the coil tunnel (Layer 2) - staggered cleanly ahead of the POV lens (z = 0.23..0.27)
  const povLabelFlux = createPovMicroLabelSprite(
    'B',
    'Campo Radial B_r (Iman -> Cobre)',
    '#38bdf8'
  );
  povLabelFlux.position.set(-0.014, tunnelCenterY + 0.062, 0.24);
  povInteriorGroup.add(povLabelFlux);

  const povLabelForce = createPovMicroLabelSprite(
    'F',
    'F_Lorentz = I x B_radial (Empuje)',
    '#10b981'
  );
  povLabelForce.position.set(0.014, tunnelCenterY + 0.024, 0.26);
  povInteriorGroup.add(povLabelForce);

  const povLabelContact = createPovMicroLabelSprite(
    'e-',
    'Flujo de Electrones e- en Cobre',
    '#38bdf8'
  );
  povLabelContact.position.set(-0.012, tunnelCenterY - 0.022, 0.23);
  povInteriorGroup.add(povLabelContact);

  // Pre-allocated helpers for Spine-Conformal Battery Vertex Deformation & Tangent-Aligned Flux Needles
  const _spineCenter = new THREE.Vector3();
  const _localVec = new THREE.Vector3();
  const _localTargetVec = new THREE.Vector3();
  const _scratchTangent = new THREE.Vector3();

  const placeMeshOnSpine = (
    mesh: THREE.Object3D,
    axialZ: number,
    state: ElectromagneticTrainState,
    trackLen: number
  ) => {
    const u = wrapU(state.u + (axialZ / trackLen) * state.direction);
    const frame = evaluateTrackFrame(u);
    const ripple = frame.inCoil
      ? Math.sin(u * TOTAL_COIL_TURNS * Math.PI * 2) * 0.0009
      : 0;
    mesh.position
      .copy(frame.position)
      .addScaledVector(frame.normal, gravityNormalOffset + ripple);
    _scratchLook
      .copy(mesh.position)
      .addScaledVector(frame.tangent, state.direction);
    mesh.lookAt(_scratchLook);
    return frame;
  };

  const updateTrainPose = (
    state: ElectromagneticTrainState,
    elapsedTime: number,
    showExplicitFields: boolean
  ) => {
    const trackLen = getTrackArcLength();
    const centerFrame = evaluateTrackFrame(state.u);

    // 1. Place Front & Rear Neodymium Magnet Stacks on the exact arc-length parameterized spine
    placeMeshOnSpine(
      frontMagnetGroup,
      TRAIN_MAGNET_CENTER_OFFSET_UNITS,
      state,
      trackLen
    );
    placeMeshOnSpine(
      rearMagnetGroup,
      -TRAIN_MAGNET_CENTER_OFFSET_UNITS,
      state,
      trackLen
    );

    // 2. Place Battery Shoulders, Dielectric Insulator Washer & Positive Button Terminal on the exact spine
    placeMeshOnSpine(frontShoulder, batteryLength * 0.5, state, trackLen);
    placeMeshOnSpine(rearShoulder, -batteryLength * 0.5, state, trackLen);
    placeMeshOnSpine(
      frontInsulatorMesh,
      batteryLength * 0.5 + 0.002,
      state,
      trackLen
    );
    placeMeshOnSpine(
      posButtonMesh,
      batteryLength * 0.5 + 0.012,
      state,
      trackLen
    );

    // 3. Spine-Conformal AA Battery Hull Deformation: every axial cross-section z0 in [-L/2, +L/2]
    // sits at the exact center of the coil cross-section at u(z0) -> 0.000 mm lateral sagitta on all curves!
    const rollAngle =
      state.distanceMeters * 3.8 * state.direction +
      Math.sin(elapsedTime * 22) * 0.012;
    const cosRoll = Math.cos(rollAngle);
    const sinRoll = Math.sin(rollAngle);

    const posAttr = hullGeo.attributes.position as THREE.BufferAttribute;
    const normAttr = hullGeo.attributes.normal as THREE.BufferAttribute;
    const posArr = posAttr.array as Float32Array;
    const normArr = normAttr.array as Float32Array;
    const vCount = posAttr.count;

    for (let v = 0; v < vCount; v++) {
      const i3 = v * 3;
      const x0 = restHullPos[i3];
      const y0 = restHullPos[i3 + 1];
      const z0 = restHullPos[i3 + 2];

      const nx0 = restHullNorm[i3];
      const ny0 = restHullNorm[i3 + 1];
      const nz0 = restHullNorm[i3 + 2];

      // Rotate local cross-section by axial rolling spin
      const xr = x0 * cosRoll - y0 * sinRoll;
      const yr = x0 * sinRoll + y0 * cosRoll;
      const nxr = nx0 * cosRoll - ny0 * sinRoll;
      const nyr = nx0 * sinRoll + ny0 * cosRoll;

      const uSlice = wrapU(state.u + (z0 / trackLen) * state.direction);
      const sliceFrame = evaluateTrackFrame(uSlice);
      const sliceRipple = sliceFrame.inCoil
        ? Math.sin(uSlice * TOTAL_COIL_TURNS * Math.PI * 2) * 0.0009
        : 0;

      _spineCenter
        .copy(sliceFrame.position)
        .addScaledVector(sliceFrame.normal, gravityNormalOffset + sliceRipple);

      const bx = -xr * state.direction;
      const bnX = -nxr * state.direction;
      const tz = nz0 * state.direction;

      posArr[i3] =
        _spineCenter.x +
        sliceFrame.binormal.x * bx +
        sliceFrame.normal.x * yr;
      posArr[i3 + 1] =
        _spineCenter.y +
        sliceFrame.binormal.y * bx +
        sliceFrame.normal.y * yr;
      posArr[i3 + 2] =
        _spineCenter.z +
        sliceFrame.binormal.z * bx +
        sliceFrame.normal.z * yr;

      normArr[i3] =
        sliceFrame.binormal.x * bnX +
        sliceFrame.normal.x * nyr +
        sliceFrame.tangent.x * tz;
      normArr[i3 + 1] =
        sliceFrame.binormal.y * bnX +
        sliceFrame.normal.y * nyr +
        sliceFrame.tangent.y * tz;
      normArr[i3 + 2] =
        sliceFrame.binormal.z * bnX +
        sliceFrame.normal.z * nyr +
        sliceFrame.tangent.z * tz;
    }
    posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;
    hullGeo.computeBoundingSphere();

    // 4. Update 5 Spine-Conformal 3D Solenoidal Core Field Streamlines (B_bobina) along the curved coil
    const solenoidActive = showExplicitFields && state.currentAmps > 0.1;
    for (let sIdx = 0; sIdx < solenoidStreams.length; sIdx++) {
      const stream = solenoidStreams[sIdx];
      stream.line.visible = solenoidActive;
      stream.cone.visible = solenoidActive;
      if (!solenoidActive) continue;

      const { positions, localX, localY } = stream;
      for (let step = 0; step <= solenoidLineSegs; step++) {
        const tau = step / solenoidLineSegs; // 0..1 from rear to front of energized coil zone
        const zLocal = (tau - 0.5) * 0.64;
        // Solenoid field lines flare outward near the magnet ends (|tau - 0.5| > 0.28)
        const endDist = Math.max(0, Math.abs(tau - 0.5) - 0.24) / 0.26;
        const flare = 1 + endDist * endDist * 0.42;

        const uStep = wrapU(state.u + (zLocal / trackLen) * state.direction);
        const stepFrame = evaluateTrackFrame(uStep);

        const px =
          stepFrame.position.x +
          stepFrame.binormal.x * (localX * flare) +
          stepFrame.normal.x * (localY * flare);
        const py =
          stepFrame.position.y +
          stepFrame.binormal.y * (localX * flare) +
          stepFrame.normal.y * (localY * flare);
        const pz =
          stepFrame.position.z +
          stepFrame.binormal.z * (localX * flare) +
          stepFrame.normal.z * (localY * flare);

        const p3 = step * 3;
        positions[p3] = px;
        positions[p3 + 1] = py;
        positions[p3 + 2] = pz;

        if (step === solenoidLineSegs) {
          stream.cone.position.set(px, py, pz);
          _scratchLook
            .set(px, py, pz)
            .addScaledVector(stepFrame.tangent, state.direction);
          stream.cone.lookAt(_scratchLook);
        }
      }
      const linePosAttr = stream.line.geometry.attributes
        .position as THREE.BufferAttribute;
      linePosAttr.needsUpdate = true;
    }

    // Position and orient dynamic contact shadow on the tabletop
    trainShadowMesh.position.set(
      centerFrame.position.x,
      0.004,
      centerFrame.position.z
    );
    trainShadowMesh.rotation.y = Math.atan2(
      centerFrame.tangent.x,
      centerFrame.tangent.z
    );

    // Position and orient 3D Physics Overlay (High-mounted arrows and 3D callout sprites)
    physicsOverlayGroup.position.copy(centerFrame.position);
    _scratchLook.copy(centerFrame.position).add(centerFrame.tangent);
    physicsOverlayGroup.lookAt(_scratchLook);
    physicsOverlayGroup.visible = showExplicitFields;
    frontFieldLoopsSubgroup.visible = showExplicitFields;
    rearFieldLoopsSubgroup.visible = showExplicitFields;
    magneticFluxInstanced.visible = showExplicitFields;
    povInteriorGroup.visible = showExplicitFields;

    // Flip force & field arrows when polarity direction reverses
    directionalVectorsSubgroup.scale.z = state.direction;

    // Update world matrices of front and rear magnet groups so flux needles and sparks lock onto them on curves
    frontMagnetGroup.updateMatrixWorld();
    rearMagnetGroup.updateMatrixWorld();

    const loopSpin = elapsedTime * 0.38 * state.direction;
    frontFieldLoopsSubgroup.rotation.z = loopSpin;
    rearFieldLoopsSubgroup.rotation.z = loopSpin;

    // Animate 112 tangent-aligned magnetic flux needles flowing along all 28 dipole loops (N -> S)
    for (let lIdx = 0; lIdx < dipoleLoopCurves.length; lIdx++) {
      const { curve, phi, stackSign, shell } = dipoleLoopCurves[lIdx];
      const totalPhi = phi + loopSpin;
      const cosPhi = Math.cos(totalPhi);
      const sinPhi = Math.sin(totalPhi);
      const parentMatWorld =
        stackSign > 0
          ? frontMagnetGroup.matrixWorld
          : rearMagnetGroup.matrixWorld;
      const speedRate = shell === 0 ? 0.34 : 0.25;

      for (let p = 0; p < needlesPerLoop; p++) {
        const instanceIdx = lIdx * needlesPerLoop + p;
        const rawT =
          ((p / needlesPerLoop) + elapsedTime * speedRate) % 1;
        const t = rawT < 0 ? rawT + 1 : rawT;
        curve.getPointAt(t, _scratchPos);
        curve.getTangentAt(t, _scratchTangent);

        const rx = _scratchPos.x * cosPhi - _scratchPos.y * sinPhi;
        const ry =
          _scratchPos.x * sinPhi +
          _scratchPos.y * cosPhi -
          gravityNormalOffset;
        const rz = _scratchPos.z;

        const tx = _scratchTangent.x * cosPhi - _scratchTangent.y * sinPhi;
        const ty = _scratchTangent.x * sinPhi + _scratchTangent.y * cosPhi;
        const tz = _scratchTangent.z;

        _localVec.set(rx, ry, rz).applyMatrix4(parentMatWorld);
        _localTargetVec
          .set(rx + tx * 0.05, ry + ty * 0.05, rz + tz * 0.05)
          .applyMatrix4(parentMatWorld);

        _dummyObj.position.copy(_localVec);
        _dummyObj.lookAt(_localTargetVec);
        const fScale =
          (shell === 0 ? 0.92 : 0.74) *
          (0.84 + 0.22 * Math.sin(elapsedTime * 5 + instanceIdx));
        _dummyObj.scale.setScalar(fScale);
        _dummyObj.updateMatrix();
        magneticFluxInstanced.setMatrixAt(instanceIdx, _dummyObj.matrix);
      }
    }
    magneticFluxInstanced.instanceMatrix.needsUpdate = true;

    // Animate 45 POV electron particles spiraling smoothly inside the semi-translucent copper wire turns ahead of the POV camera
    for (let pe = 0; pe < povElectronCount; pe++) {
      if (!showExplicitFields || state.currentAmps <= 0.1) {
        _dummyObj.position.set(0, -10, 0);
        _dummyObj.scale.copy(_inactiveScale);
        _dummyObj.updateMatrix();
        povElectronsInstanced.setMatrixAt(pe, _dummyObj.matrix);
        continue;
      }
      const ringSlot = pe % 9;
      const zRing = 0.14 + ringSlot * 0.026;
      const theta =
        ((pe / povElectronCount) * Math.PI * 2 +
          elapsedTime * 2.2 * state.direction +
          ringSlot * 0.55) %
        (Math.PI * 2);
      const rWire = COIL_RADIUS_UNITS * 0.968;
      _dummyObj.position.set(
        Math.cos(theta) * rWire,
        tunnelCenterY + Math.sin(theta) * rWire,
        zRing
      );
      _dummyObj.scale.setScalar(
        0.86 + 0.24 * Math.sin(elapsedTime * 9 + pe)
      );
      _dummyObj.updateMatrix();
      povElectronsInstanced.setMatrixAt(pe, _dummyObj.matrix);
    }
    povElectronsInstanced.instanceMatrix.needsUpdate = true;

    // Animate contact spark rings & 24 electric micro-spark filaments locked to the bottom sliding contact of front/rear magnets
    const contactActive = state.currentAmps > 0.1;
    frontContactRing.visible = contactActive && showExplicitFields;
    rearContactRing.visible = contactActive && showExplicitFields;

    if (contactActive && showExplicitFields) {
      const pulse = 0.94 + 0.12 * Math.sin(elapsedTime * 18);
      frontContactRing.scale.setScalar(pulse);
      rearContactRing.scale.setScalar(pulse);

      for (let s = 0; s < sparkCount; s++) {
        const isFront = s < sparkCount / 2;
        const idx = s % (sparkCount / 2);
        const baseAngle =
          -Math.PI * 0.5 +
          Math.sin(idx * 1.7 + elapsedTime * (isFront ? 9 : -9)) *
            (Math.PI * 0.42);
        const flicker = Math.sin(elapsedTime * 34 + s * 3.7);

        if (flicker > -0.1) {
          const r = magnetRadius * 1.01;
          const parentMat = isFront
            ? frontMagnetGroup.matrixWorld
            : rearMagnetGroup.matrixWorld;
          _localVec
            .set(Math.cos(baseAngle) * r, Math.sin(baseAngle) * r, 0)
            .applyMatrix4(parentMat);
          _dummyObj.position.copy(_localVec);
          _dummyObj.rotation.set(
            Math.sin(baseAngle) * 0.8,
            Math.cos(baseAngle) * 0.8,
            baseAngle + Math.PI / 2
          );
          const sScale =
            state.sparkIntensity * (0.65 + 0.55 * Math.abs(flicker));
          _dummyObj.scale.setScalar(sScale);
          _dummyObj.updateMatrix();
          sparkInstanced.setMatrixAt(s, _dummyObj.matrix);
        } else {
          _dummyObj.position.set(0, -10, 0);
          _dummyObj.scale.copy(_inactiveScale);
          _dummyObj.updateMatrix();
          sparkInstanced.setMatrixAt(s, _dummyObj.matrix);
        }
      }
    } else {
      for (let s = 0; s < sparkCount; s++) {
        _dummyObj.position.set(0, -10, 0);
        _dummyObj.scale.copy(_inactiveScale);
        _dummyObj.updateMatrix();
        sparkInstanced.setMatrixAt(s, _dummyObj.matrix);
      }
    }
    sparkInstanced.instanceMatrix.needsUpdate = true;
  };

  return {
    rootGroup,
    batteryGroup,
    frontMagnetGroup,
    rearMagnetGroup,
    physicsOverlayGroup,
    trainShadowMesh,
    frontContactRing,
    rearContactRing,
    sparkInstanced,
    forceArrowGroup,
    fieldLoopsGroup,
    updateTrainPose,
  };
}
