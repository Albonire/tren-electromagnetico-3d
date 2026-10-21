import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  COIL_START_U,
  COIL_RADIUS_UNITS,
  WIRE_RADIUS_UNITS,
  computeElectromagneticState,
  stepTrainPhysics,
  reverseTrainPolarity,
  evaluateTrackFrame,
  findClosestTrackU,
  travelSign,
  circularDeltaU,
  getTrackArcLength,
  METERS_PER_UNIT,
  computeFirstPersonPovFrame,
  ElectromagneticTrainState,
} from './physics';
import {
  buildLabEnvironment,
  buildHelicalCoilSystem,
  buildElectromagneticTrainRig,
} from './models';
import {
  CalloutLabels,
  SimulationUi,
  SPEED_PRESETS,
  nearestSpeedPresetIndex,
  type CalloutInfo,
  type CameraMode,
  type SimulationUiActions,
  type SimulationUiState,
} from '../ui/SimulationUi';
import { computePipLayout, isCompactWidth } from '../ui/layout';

export type { CameraMode };

interface SimControlApi extends SimulationUiActions {
  setTrainU: (u: number) => void;
}

function computeResponsiveOverviewPos(
  aspect: number,
  target = new THREE.Vector3()
): THREE.Vector3 {
  const scale =
    aspect < 1.35 ? Math.min(2.7, 1.55 / Math.max(aspect, 0.45)) : 1.0;
  return target.set(0.0, 3.05 * scale, 4.45 * scale);
}

export default function ElectromagneticTrainCanvas() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<SimControlApi | null>(null);

  const calloutElsRef = useRef<Record<string, HTMLDivElement | null>>({});
  const toggleHelpRef = useRef<(open?: boolean) => void>(() => {});

  const [uiState, setUiState] = useState<SimulationUiState>({
    isPaused: false,
    cameraMode: 'overview',
    showExplicitFields: true,
    showPip: true,
    direction: 1,
    speedMultiplier: 0.1,
    currentAmps: 0,
    magneticFieldmT: 0,
    propulsionForceN: 0,
    speedCmPerS: 0,
    lapCount: 0,
  });
  const [callouts, setCallouts] = useState<CalloutInfo[]>([]);
  const [viewport, setViewport] = useState(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }));
  const [helpOpen, setHelpOpen] = useState(false);
  toggleHelpRef.current = (open?: boolean) =>
    setHelpOpen((prev) => (open === undefined ? !prev : open));

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // 1. Detect software WebGL (SwiftShader / llvmpipe) per Three.js skill guidelines
    const isSoftwareGl = (() => {
      const probeCanvas = document.createElement('canvas');
      const gl = probeCanvas.getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    // 2. Initialize WebGLRenderer with ACESFilmicToneMapping and SRGBColorSpace
    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.setPixelRatio(
      isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2)
    );
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.autoClear = false;

    // Clear container and mount canvas
    container.innerHTML = '';
    renderer.domElement.setAttribute('data-testid', 'em-train-canvas');
    renderer.domElement.className =
      'block w-full h-full cursor-grab active:cursor-grabbing';
    container.appendChild(renderer.domElement);

    // 3. Dark Atmospheric Laboratory Scene with Focused Nearby-Window Sunlight
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x06090f);
    scene.fog = new THREE.FogExp2(0x06090f, 0.011);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(
      new RoomEnvironment(),
      0.035
    ).texture;
    pmremGenerator.dispose();

    // 4. Primary Camera & Dedicated First-Person POV / PiP Secondary Camera
    const initialAspect = window.innerWidth / window.innerHeight;
    const camera = new THREE.PerspectiveCamera(35, initialAspect, 0.04, 50);
    camera.layers.enable(0);
    camera.layers.enable(1); // External 3D callouts & macro force vectors

    const pipCamera = new THREE.PerspectiveCamera(62, 1.6, 0.008, 22);
    pipCamera.layers.enable(0);
    pipCamera.layers.enable(2); // Interior POV radial flux, copper conduction rings & micro-labels

    const overviewCamPos = computeResponsiveOverviewPos(initialAspect);
    const overviewTarget = new THREE.Vector3(0.0, 0.06, 0.05);
    camera.position.copy(overviewCamPos);
    camera.lookAt(overviewTarget);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(overviewTarget);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.485;
    controls.minDistance = 0.35;
    controls.maxDistance = 16.0;
    controls.update();

    // 5. Dark Laboratory + Nearby Window Sunlight Rig ("oscuro y que solo haya sol como si entrara desde una ventana cercana")
    // Low moody dark-room ambient so unlit areas stay rich and dark
    const hemiLight = new THREE.HemisphereLight(0x1e293b, 0x030509, 0.14);
    scene.add(hemiLight);

    // Subtle global twilight fill so the shadow side of the coil remains legible in the dark room
    const sunLight = new THREE.DirectionalLight(0xffcf86, 0.42);
    sunLight.position.set(3.8, 4.2, 1.2);
    scene.add(sunLight);

    // Focused Window Sunbeam Spotlight entering from the nearby upper-right window and casting crisp shadows onto the dark workbench
    const windowSpotLight = new THREE.SpotLight(
      0xffd68a,
      7.2,
      15.0,
      0.46,
      0.54,
      1.1
    );
    windowSpotLight.position.set(3.9, 3.7, -0.8);
    windowSpotLight.target.position.set(0.42, 0.0, 0.05);
    windowSpotLight.castShadow = !isSoftwareGl;
    const shadowRes = isSoftwareGl ? 1024 : 2048;
    windowSpotLight.shadow.mapSize.width = shadowRes;
    windowSpotLight.shadow.mapSize.height = shadowRes;
    windowSpotLight.shadow.camera.near = 0.8;
    windowSpotLight.shadow.camera.far = 14;
    windowSpotLight.shadow.bias = -0.0003;
    windowSpotLight.shadow.normalBias = 0.015;
    scene.add(windowSpotLight.target);
    scene.add(windowSpotLight);

    // Subtle cool dark-room rim light on the shadow side so metallic edges stay crisp
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.28);
    rimLight.position.set(-3.4, 2.6, -2.4);
    scene.add(rimLight);

    // Subtle warm copper bounce fill from the sunlit patch on the workbench
    const bounceFillLight = new THREE.DirectionalLight(0xf59e0b, 0.26);
    bounceFillLight.position.set(1.2, -1.5, 1.4);
    scene.add(bounceFillLight);

    // 6. Build 3D Laboratory Environment, Helical Wire Coil, and Electromagnetic Train Rig
    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    const labEnvironment = buildLabEnvironment(scene, maxAnisotropy, isSoftwareGl);
    const coilRig = buildHelicalCoilSystem(maxAnisotropy, isSoftwareGl);
    scene.add(coilRig.group);

    const trainRig = buildElectromagneticTrainRig(maxAnisotropy, isSoftwareGl);
    scene.add(trainRig.rootGroup);
    setCallouts(
      trainRig.calloutAnchors.map(({ id, symbol, title, subtitle, accent }) => ({
        id,
        symbol,
        title,
        subtitle,
        accent,
      }))
    );

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    // 7. Simulation State & Interactive Camera / Field Controls
    // The train starts at rest just past the gap and accelerates on its own
    let simState: ElectromagneticTrainState = computeElectromagneticState(
      COIL_START_U + 0.05,
      0,
      0,
      0,
      1
    );
    let isPaused = false;
    let showExplicitFields = true;
    let showPipViewport = true;
    // Time scale of the simulation (1 = real time); the real train is too fast to follow at 1x
    let speedMultiplier = 0.1;
    let cameraMode: CameraMode = 'overview';
    let userInteractingWithOrbit = false;
    let isDraggingTrain = false;
    let elapsedTime = 0;

    coilRig.updateActiveTurns(simState, 0, 1 / 60, showExplicitFields);
    trainRig.updateTrainPose(simState, 0, showExplicitFields);

    const syncBodyDataset = () => {
      document.body.dataset.simulationState = isPaused ? 'paused' : 'running';
      document.body.dataset.cameraMode = cameraMode;
      document.body.dataset.pipViewport = showPipViewport
        ? 'visible'
        : 'hidden';
      document.body.dataset.explicitFields = showExplicitFields
        ? 'visible'
        : 'hidden';
      document.body.dataset.electronFlow = showExplicitFields
        ? 'active'
        : 'hidden';
      document.body.dataset.polarity =
        simState.direction === 1 ? 'forward' : 'reversed';
      document.body.dataset.trainU = simState.u.toFixed(3);
      document.body.dataset.currentAmps = simState.currentAmps.toFixed(2);
      document.body.dataset.sparkIntensity = simState.sparkIntensity.toFixed(2);
      document.body.dataset.coilPerturbationMm =
        coilRig.maxPerturbationMm.toFixed(2);
      document.body.dataset.lapCount = String(simState.lapCount);

      setUiState({
        isPaused,
        cameraMode,
        showExplicitFields,
        showPip: showPipViewport,
        direction: simState.direction,
        speedMultiplier,
        currentAmps: simState.currentAmps,
        magneticFieldmT: simState.magneticFieldmT,
        propulsionForceN: simState.propulsionForceN,
        // 1 world unit = 10 cm; displacement is scaled by the playback multiplier
        speedCmPerS: simState.speed * 100,
        lapCount: simState.lapCount,
      });
    };
    syncBodyDataset();

    const configureCameraLayersAndFov = (mode: CameraMode) => {
      if (mode === 'pov') {
        camera.fov = 62;
        camera.near = 0.008;
        camera.far = 22;
        camera.layers.disable(1);
        camera.layers.enable(2);
        camera.updateProjectionMatrix();
        controls.enabled = false;
        renderer.domElement.style.cursor = 'default';
      } else {
        renderer.domElement.style.cursor = '';
        camera.fov = 35;
        camera.near = 0.04;
        camera.far = 50;
        camera.up.set(0, 1, 0);
        camera.layers.enable(1);
        camera.layers.disable(2);
        camera.updateProjectionMatrix();
        controls.enabled = !isDraggingTrain;
      }
    };

    const applyPovCameraImmediate = (targetCam: THREE.PerspectiveCamera) => {
      const pov = computeFirstPersonPovFrame(simState);
      targetCam.position.copy(pov.eyePosition);
      targetCam.up.copy(pov.upVector);
      targetCam.lookAt(pov.lookTarget);
    };

    const applyFollowCameraImmediate = () => {
      const frame = evaluateTrackFrame(simState.u);
      controls.target
        .copy(frame.position)
        .addScaledVector(frame.normal, 0.24);
      camera.position
        .copy(frame.position)
        .addScaledVector(frame.binormal, 1.72)
        .addScaledVector(frame.normal, 1.05)
        .addScaledVector(frame.tangent, -0.45 * travelSign(simState));
      controls.update();
    };

    const applyTunnelCameraImmediate = () => {
      const frame = evaluateTrackFrame(simState.u);
      controls.target
        .copy(frame.position)
        .addScaledVector(frame.normal, 0.16);
      camera.position
        .copy(frame.position)
        .addScaledVector(frame.tangent, 1.15 * travelSign(simState))
        .addScaledVector(frame.binormal, 1.05)
        .addScaledVector(frame.normal, 0.58);
      controls.update();
    };

    const setCameraMode = (nextMode: CameraMode) => {
      cameraMode = nextMode;
      configureCameraLayersAndFov(nextMode);
      if (nextMode === 'overview') {
        computeResponsiveOverviewPos(
          window.innerWidth / window.innerHeight,
          overviewCamPos
        );
        camera.position.copy(overviewCamPos);
        controls.target.copy(overviewTarget);
        controls.update();
      } else if (nextMode === 'pov') {
        applyPovCameraImmediate(camera);
      } else if (nextMode === 'follow') {
        applyFollowCameraImmediate();
      } else if (nextMode === 'top') {
        camera.position.set(0, 5.8, 0.01);
        controls.target.set(0, 0, 0);
        controls.update();
      } else if (nextMode === 'tunnel') {
        applyTunnelCameraImmediate();
      }
      syncBodyDataset();
    };

    const toggleFieldAndElectronVisualization = () => {
      showExplicitFields = !showExplicitFields;
      coilRig.updateActiveTurns(
        simState,
        elapsedTime,
        1 / 60,
        showExplicitFields
      );
      trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
      syncBodyDataset();
    };

    const setTrainUImmediate = (targetU: number, velocity = 0) => {
      simState = computeElectromagneticState(
        targetU,
        velocity,
        simState.distanceMeters,
        simState.lapCount,
        simState.direction
      );
      coilRig.updateActiveTurns(
        simState,
        elapsedTime,
        1 / 60,
        showExplicitFields
      );
      trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
      if (cameraMode === 'pov') {
        applyPovCameraImmediate(camera);
      } else if (cameraMode === 'follow') {
        applyFollowCameraImmediate();
      }
      syncBodyDataset();
      renderDualViewportFrame();
    };

    const reversePolarity = () => {
      simState = reverseTrainPolarity(simState);
      trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
      syncBodyDataset();
    };

    const stepSpeedPreset = (delta: 1 | -1) => {
      const idx = THREE.MathUtils.clamp(
        nearestSpeedPresetIndex(speedMultiplier) + delta,
        0,
        SPEED_PRESETS.length - 1
      );
      speedMultiplier = SPEED_PRESETS[idx].value;
      syncBodyDataset();
    };

    apiRef.current = {
      togglePause: () => {
        isPaused = !isPaused;
        syncBodyDataset();
      },
      setCameraMode,
      toggleFields: toggleFieldAndElectronVisualization,
      togglePip: () => {
        showPipViewport = !showPipViewport;
        syncBodyDataset();
      },
      reversePolarity,
      setSpeed: (mult: number) => {
        speedMultiplier = mult;
        syncBodyDataset();
      },
      setTrainU: setTrainUImmediate,
    };

    (window as unknown as Record<string, unknown>).__electromagneticTrainSim = {
      setTrainU: setTrainUImmediate,
      setCameraMode,
      toggleFields: toggleFieldAndElectronVisualization,
      reversePolarity,
      getState: () => simState,
    };

    const raycaster = new THREE.Raycaster();
    const pointerNdc = new THREE.Vector2();
    const dragPlane = new THREE.Plane(
      new THREE.Vector3(0, 1, 0),
      -(COIL_RADIUS_UNITS + WIRE_RADIUS_UNITS)
    );
    const dragHitPoint = new THREE.Vector3();
    // Hand velocity while dragging (simulated m/s), so releasing the train throws it
    let dragLastU = 0;
    let dragLastTime = 0;
    let dragVelocity = 0;

    const onPointerDown = (e: PointerEvent) => {
      if (cameraMode === 'pov') return;

      const rect = renderer.domElement.getBoundingClientRect();
      pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);

      const trainHits = raycaster.intersectObjects(
        [
          trainRig.batteryGroup,
          trainRig.frontMagnetGroup,
          trainRig.rearMagnetGroup,
        ],
        true
      );
      if (trainHits.length > 0 || e.shiftKey) {
        isDraggingTrain = true;
        controls.enabled = false;
        renderer.domElement.style.cursor = 'grabbing';
        dragVelocity = 0;
        dragLastU = simState.u;
        dragLastTime = performance.now();
        if (e.shiftKey && raycaster.ray.intersectPlane(dragPlane, dragHitPoint)) {
          dragLastU = findClosestTrackU(dragHitPoint);
          setTrainUImmediate(dragLastU);
        }
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (isDraggingTrain) {
        raycaster.setFromCamera(pointerNdc, camera);
        if (raycaster.ray.intersectPlane(dragPlane, dragHitPoint)) {
          const nextU = findClosestTrackU(dragHitPoint);
          const now = performance.now();
          const elapsedMs = Math.max(now - dragLastTime, 1);
          const handVelocity =
            (circularDeltaU(nextU, dragLastU) * getTrackArcLength() * METERS_PER_UNIT) /
            (elapsedMs / 1000) /
            Math.max(speedMultiplier, 0.01);
          const blend = 1 - Math.exp(-elapsedMs / 60);
          dragVelocity = THREE.MathUtils.clamp(
            dragVelocity + (handVelocity - dragVelocity) * blend,
            -1.2,
            1.2
          );
          dragLastU = nextU;
          dragLastTime = now;
          setTrainUImmediate(nextU, dragVelocity);
        }
        return;
      }

      // Update hover cursor feedback over train or PiP
      if (cameraMode !== 'pov') {
        raycaster.setFromCamera(pointerNdc, camera);
        const trainHits = raycaster.intersectObjects(
          [
            trainRig.batteryGroup,
            trainRig.frontMagnetGroup,
            trainRig.rearMagnetGroup,
          ],
          true
        );
        if (trainHits.length > 0) {
          renderer.domElement.style.cursor = 'grab';
          return;
        }
      }
      renderer.domElement.style.cursor = '';
    };

    const onPointerUp = () => {
      if (!isDraggingTrain) return;
      isDraggingTrain = false;
      // A hand that stopped before letting go does not throw the train
      const heldStill = performance.now() - dragLastTime > 120;
      setTrainUImmediate(simState.u, heldStill ? 0 : dragVelocity);
      if (cameraMode !== 'pov') {
        controls.enabled = true;
      }
      renderer.domElement.style.cursor = '';
    };

    const onDoubleClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);

      const hits = raycaster.intersectObjects(
        [
          trainRig.batteryGroup,
          trainRig.frontMagnetGroup,
          trainRig.rearMagnetGroup,
        ],
        true
      );
      if (hits.length > 0 || cameraMode === 'pov') {
        reversePolarity();
      }
    };

    const onControlsStart = () => {
      userInteractingWithOrbit = true;
    };
    const onControlsEnd = () => {
      userInteractingWithOrbit = false;
    };

    controls.addEventListener('start', onControlsStart);
    controls.addEventListener('end', onControlsEnd);
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('dblclick', onDoubleClick);

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const onControl = !!target?.closest('button, input, select, textarea, [role="dialog"]');
      // Let focused buttons keep their native Space/Enter activation
      if (onControl && (e.code === 'Space' || e.key === 'Enter')) return;

      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (key === 'Escape') {
        toggleHelpRef.current(false);
      } else if (key === '?' || key === 'h') {
        toggleHelpRef.current();
      } else if (e.code === 'Space') {
        e.preventDefault();
        isPaused = !isPaused;
        syncBodyDataset();
      } else if (key === 'f' || key === 'e') {
        toggleFieldAndElectronVisualization();
      } else if (key === 'v') {
        showPipViewport = !showPipViewport;
        syncBodyDataset();
      } else if (key === 'r') {
        reversePolarity();
      } else if (key === 'ArrowUp' || key === '+') {
        e.preventDefault();
        stepSpeedPreset(1);
      } else if (key === 'ArrowDown' || key === '-') {
        e.preventDefault();
        stepSpeedPreset(-1);
      } else if (key === '1') {
        setCameraMode('overview');
      } else if (key === '2' || key === 'c') {
        setCameraMode('follow');
      } else if (key === '3') {
        setCameraMode('top');
      } else if (key === '4') {
        setCameraMode('tunnel');
      } else if (key === '5' || key === 'p') {
        setCameraMode(cameraMode === 'pov' ? 'overview' : 'pov');
      }
    };
    window.addEventListener('keydown', onKeyDown);

    const onResize = () => {
      const aspect = window.innerWidth / window.innerHeight;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      setViewport({ width: window.innerWidth, height: window.innerHeight });
      if (cameraMode === 'overview' && !userInteractingWithOrbit) {
        computeResponsiveOverviewPos(aspect, overviewCamPos);
        camera.position.copy(overviewCamPos);
        controls.target.copy(overviewTarget);
        controls.update();
      }
    };
    window.addEventListener('resize', onResize);

    // 8. Render Loop with Dual-Viewport WebGL Scissor PiP & mandatory software-WebGL afterDraw() pacing
    let frameId = 0;
    let disposed = false;
    let lastTime = performance.now();
    let datasetTickCounter = 0;

    const desiredCamPos = new THREE.Vector3();
    const desiredTarget = new THREE.Vector3();
    const calloutWorldPos = new THREE.Vector3();

    // Projects each 3D callout anchor to screen space and moves its HTML label (no React re-render)
    const updateCalloutLabels = () => {
      const fullW = window.innerWidth;
      const fullH = window.innerHeight;
      const labelsVisible =
        cameraMode !== 'pov' && trainRig.physicsOverlayGroup.visible;
      for (const anchor of trainRig.calloutAnchors) {
        const el = calloutElsRef.current[anchor.id];
        if (!el) continue;
        anchor.object.getWorldPosition(calloutWorldPos).project(camera);
        const onScreen =
          labelsVisible &&
          calloutWorldPos.z < 1 &&
          Math.abs(calloutWorldPos.x) < 1.1 &&
          Math.abs(calloutWorldPos.y) < 1.1;
        el.style.opacity = onScreen ? '1' : '0';
        if (!onScreen) continue;
        const x = (calloutWorldPos.x * 0.5 + 0.5) * fullW;
        const y = (-calloutWorldPos.y * 0.5 + 0.5) * fullH;
        el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -100%)`;
      }
    };

    function afterDraw(cb: () => void) {
      if (disposed) return;
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync
        ? gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
        : null;
      gl.flush();
      if (!sync) {
        frameId = window.setTimeout(cb, 33);
        return;
      }
      const poll = () => {
        if (disposed) {
          gl.deleteSync(sync);
          return;
        }
        const state = gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 4);
        } else {
          gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 33);
        }
      };
      frameId = window.setTimeout(poll, 4);
    }

    function renderDualViewportFrame() {
      const fullW = window.innerWidth;
      const fullH = window.innerHeight;

      // Pass 1: Full-screen Primary Camera
      renderer.setViewport(0, 0, fullW, fullH);
      renderer.setScissor(0, 0, fullW, fullH);
      renderer.setScissorTest(true);
      renderer.clear();
      renderer.render(scene, camera);

      // Pass 2: Live WebGL Picture-in-Picture (PiP) viewport; the HTML bezel frames it
      if (showPipViewport) {
        const pip = computePipLayout(fullW, fullH);

        // Configure PiP camera pose & layers
        pipCamera.aspect = pip.width / pip.height;
        if (cameraMode === 'pov') {
          pipCamera.fov = 35;
          pipCamera.near = 0.05;
          pipCamera.far = 40;
          pipCamera.layers.disable(2);
          pipCamera.layers.enable(1);
          pipCamera.updateProjectionMatrix();
          computeResponsiveOverviewPos(pipCamera.aspect, pipCamera.position);
          pipCamera.up.set(0, 1, 0);
          pipCamera.lookAt(overviewTarget);
        } else {
          pipCamera.fov = 62;
          pipCamera.near = 0.008;
          pipCamera.far = 22;
          pipCamera.layers.disable(1);
          pipCamera.layers.disable(2);
          pipCamera.updateProjectionMatrix();
          applyPovCameraImmediate(pipCamera);
        }

        renderer.setViewport(pip.left, pip.glY, pip.width, pip.height);
        renderer.setScissor(pip.left, pip.glY, pip.width, pip.height);
        renderer.clear();
        renderer.render(scene, pipCamera);
      }

      renderer.setScissorTest(false);
      updateCalloutLabels();
    }

    function tick() {
      if (disposed) return;
      const now = performance.now();
      const rawDt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;
      const dt = isSoftwareGl ? 1 / 30 : rawDt;

      // Everything physical (motion, sparks, electrons) runs in simulated time
      const simDt = dt * speedMultiplier;
      labEnvironment.update(dt, now / 1000);

      if (!isPaused && !isDraggingTrain) {
        elapsedTime += simDt;
        simState = stepTrainPhysics(simState, simDt);
        coilRig.updateActiveTurns(
          simState,
          elapsedTime,
          simDt,
          showExplicitFields
        );
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);

        datasetTickCounter++;
        if (datasetTickCounter % 8 === 0) {
          syncBodyDataset();
        }
      } else if (isDraggingTrain) {
        elapsedTime += simDt;
        coilRig.updateActiveTurns(
          simState,
          elapsedTime,
          simDt,
          showExplicitFields
        );
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
      }

      if (cameraMode === 'pov') {
        applyPovCameraImmediate(camera);
      } else if (cameraMode === 'follow' && !userInteractingWithOrbit) {
        const frame = evaluateTrackFrame(simState.u);
        desiredTarget.copy(frame.position).addScaledVector(frame.normal, 0.24);
        desiredCamPos
          .copy(frame.position)
          .addScaledVector(frame.binormal, 1.72)
          .addScaledVector(frame.normal, 1.05)
          .addScaledVector(frame.tangent, -0.45 * travelSign(simState));

        const dampFactor = 1 - Math.exp(-6.0 * dt);
        camera.position.lerp(desiredCamPos, dampFactor);
        controls.target.lerp(desiredTarget, dampFactor);
        controls.update();
      } else if (cameraMode === 'tunnel' && !userInteractingWithOrbit) {
        const frame = evaluateTrackFrame(simState.u);
        desiredTarget
          .copy(frame.position)
          .addScaledVector(frame.normal, 0.16);
        desiredCamPos
          .copy(frame.position)
          .addScaledVector(frame.tangent, 1.15 * travelSign(simState))
          .addScaledVector(frame.binormal, 1.05)
          .addScaledVector(frame.normal, 0.58);

        const dampFactor = 1 - Math.exp(-7.0 * dt);
        camera.position.lerp(desiredCamPos, dampFactor);
        controls.target.lerp(desiredTarget, dampFactor);
        controls.update();
      } else {
        controls.update();
      }

      // Keep the subject crisp at any zoom while the far end of the room dissolves into darkness
      if (scene.fog instanceof THREE.Fog) {
        const focusDist =
          cameraMode === 'pov' ? 1.5 : camera.position.distanceTo(controls.target);
        scene.fog.near = Math.max(2.5, focusDist * 1.15);
        scene.fog.far = Math.max(9, focusDist * 3.2);
      }

      renderDualViewportFrame();

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    }

    renderDualViewportFrame();
    if (isSoftwareGl) {
      afterDraw(tick);
    } else {
      frameId = requestAnimationFrame(tick);
    }

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('pointerup', onPointerUp);
      controls.removeEventListener('start', onControlsStart);
      controls.removeEventListener('end', onControlsEnd);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('dblclick', onDoubleClick);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  const actions = useMemo<SimulationUiActions>(
    () => ({
      togglePause: () => apiRef.current?.togglePause(),
      reversePolarity: () => apiRef.current?.reversePolarity(),
      toggleFields: () => apiRef.current?.toggleFields(),
      togglePip: () => apiRef.current?.togglePip(),
      setCameraMode: (mode) => apiRef.current?.setCameraMode(mode),
      setSpeed: (mult) => apiRef.current?.setSpeed(mult),
    }),
    []
  );
  const registerCalloutRef = useCallback(
    (id: string, el: HTMLDivElement | null) => {
      calloutElsRef.current[id] = el;
    },
    []
  );

  const compact = isCompactWidth(viewport.width);
  const pipLayout = computePipLayout(viewport.width, viewport.height);

  return (
    <main
      aria-label="Simulación 3D del tren electromagnético homopolar"
      className="relative m-0 h-dvh w-screen select-none overflow-hidden bg-[#06090f] p-0"
    >
      <div ref={mountRef} className="h-full w-full" />
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <CalloutLabels
          callouts={callouts}
          registerRef={registerCalloutRef}
          compact={compact}
        />
      </div>
      <SimulationUi
        ui={uiState}
        actions={actions}
        compact={compact}
        wide={viewport.width >= 1100}
        pipLayout={pipLayout}
        helpOpen={helpOpen}
        setHelpOpen={setHelpOpen}
      />
    </main>
  );
}
