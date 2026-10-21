import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  Play,
  Pause,
  ArrowLeftRight,
  Magnet,
  Eye,
  Navigation,
  ScanEye,
  Compass,
  Gauge,
  MoveHorizontal,
  Sun,
} from 'lucide-react';
import {
  COIL_START_U,
  COIL_RADIUS_UNITS,
  WIRE_RADIUS_UNITS,
  computeElectromagneticState,
  stepTrainPhysics,
  reverseTrainPolarity,
  evaluateTrackFrame,
  findClosestTrackU,
  computeFirstPersonPovFrame,
  ElectromagneticTrainState,
} from './physics';
import {
  buildLabEnvironment,
  buildHelicalCoilSystem,
  buildElectromagneticTrainRig,
} from './models';

export type CameraMode = 'overview' | 'pov' | 'follow' | 'top' | 'tunnel';

interface SimControlApi {
  togglePause: () => void;
  setCameraMode: (mode: CameraMode) => void;
  toggleFields: () => void;
  reversePolarity: () => void;
  setSpeed: (mult: number) => void;
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

/**
 * Computes the bottom-right WebGL Picture-in-Picture (PiP) viewport rectangle in CSS pixels.
 */
function computePipRect(width: number, height: number) {
  const isMobile = width < 640;
  const pipW = isMobile
    ? Math.round(Math.min(168, width * 0.44))
    : Math.round(Math.min(360, width * 0.26));
  const pipH = Math.round(pipW * 0.62);
  const margin = isMobile ? 14 : 22;
  return {
    x: width - pipW - margin,
    y: margin, // WebGL viewport Y is measured from bottom edge
    width: pipW,
    height: pipH,
    margin,
  };
}

export default function ElectromagneticTrainCanvas() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<SimControlApi | null>(null);

  const [uiState, setUiState] = useState({
    isPaused: false,
    cameraMode: 'overview' as CameraMode,
    showExplicitFields: true,
    direction: 1 as 1 | -1,
    speedMultiplier: 0.55,
    currentAmps: 1.71,
    magneticFieldmT: 11.6,
    propulsionForceN: 0.547,
    inCoil: true,
  });

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
    buildLabEnvironment(scene, maxAnisotropy, isSoftwareGl);
    const coilRig = buildHelicalCoilSystem(maxAnisotropy, isSoftwareGl);
    scene.add(coilRig.group);

    const trainRig = buildElectromagneticTrainRig(maxAnisotropy, isSoftwareGl);
    scene.add(trainRig.rootGroup);

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    // 7. Simulation State & Interactive Camera / Field Controls
    let simState: ElectromagneticTrainState = computeElectromagneticState(
      COIL_START_U + 0.03,
      1.22,
      0,
      0,
      1
    );
    let isPaused = false;
    let showExplicitFields = true;
    let showPipViewport = true;
    let speedMultiplier = 0.55;
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
        direction: simState.direction,
        speedMultiplier,
        currentAmps: simState.currentAmps,
        magneticFieldmT: simState.magneticFieldmT,
        propulsionForceN: simState.propulsionForceN,
        inCoil: simState.inCoil,
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
      } else {
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
        .addScaledVector(frame.tangent, -0.45 * simState.direction);
      controls.update();
    };

    const applyTunnelCameraImmediate = () => {
      const frame = evaluateTrackFrame(simState.u);
      controls.target
        .copy(frame.position)
        .addScaledVector(frame.normal, 0.16);
      camera.position
        .copy(frame.position)
        .addScaledVector(frame.tangent, 1.15 * simState.direction)
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

    const setTrainUImmediate = (targetU: number) => {
      simState = computeElectromagneticState(
        targetU,
        simState.speed,
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

    apiRef.current = {
      togglePause: () => {
        isPaused = !isPaused;
        syncBodyDataset();
      },
      setCameraMode,
      toggleFields: toggleFieldAndElectronVisualization,
      reversePolarity: () => {
        simState = reverseTrainPolarity(simState);
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
        syncBodyDataset();
      },
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
      reversePolarity: () => apiRef.current?.reversePolarity(),
      getState: () => simState,
    };

    const raycaster = new THREE.Raycaster();
    const pointerNdc = new THREE.Vector2();
    const dragPlane = new THREE.Plane(
      new THREE.Vector3(0, 1, 0),
      -(COIL_RADIUS_UNITS + WIRE_RADIUS_UNITS)
    );
    const dragHitPoint = new THREE.Vector3();
    let pointerDownPos = { x: 0, y: 0 };

    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
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
        if (e.shiftKey && raycaster.ray.intersectPlane(dragPlane, dragHitPoint)) {
          setTrainUImmediate(findClosestTrackU(dragHitPoint));
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
          setTrainUImmediate(nextU);
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

    const onPointerUp = (e: PointerEvent) => {
      const wasDraggingTrain = isDraggingTrain;
      if (isDraggingTrain) {
        isDraggingTrain = false;
        if (cameraMode !== 'pov') {
          controls.enabled = true;
        }
        renderer.domElement.style.cursor = '';
      }

      const dist = Math.hypot(
        e.clientX - pointerDownPos.x,
        e.clientY - pointerDownPos.y
      );
      if (dist > 6 || wasDraggingTrain) return;

      const rect = renderer.domElement.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const localYFromBottom = rect.height - (e.clientY - rect.top);

      // Check if user clicked inside the 3D Picture-in-Picture (PiP) viewport in the bottom-right corner
      if (showPipViewport) {
        const pip = computePipRect(rect.width, rect.height);
        if (
          localX >= pip.x &&
          localX <= pip.x + pip.width &&
          localYFromBottom >= pip.y &&
          localYFromBottom <= pip.y + pip.height
        ) {
          setCameraMode(cameraMode === 'pov' ? 'overview' : 'pov');
          return;
        }
      }

      // If currently in full-screen First-Person POV mode, clicking anywhere in the tunnel cycles to follow/overview
      if (cameraMode === 'pov') {
        setCameraMode('follow');
        return;
      }

      pointerNdc.x = (localX / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);

      // Clicking on the copper coil rail toggles the Magnetic Field Lines + Electron Flow (e-) visualization
      const coilHits = raycaster.intersectObject(coilRig.coilMesh, false);
      if (coilHits.length > 0) {
        toggleFieldAndElectronVisualization();
      }
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
        simState = reverseTrainPolarity(simState);
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
        syncBodyDataset();
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
      if (e.code === 'Space') {
        e.preventDefault();
        isPaused = !isPaused;
        syncBodyDataset();
      } else if (
        e.key === 'f' ||
        e.key === 'F' ||
        e.key === 'e' ||
        e.key === 'E'
      ) {
        toggleFieldAndElectronVisualization();
      } else if (e.key === 'v' || e.key === 'V') {
        showPipViewport = !showPipViewport;
        syncBodyDataset();
      } else if (e.key === 'r' || e.key === 'R') {
        simState = reverseTrainPolarity(simState);
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);
        syncBodyDataset();
      } else if (e.key === 'ArrowUp' || e.key === '+') {
        speedMultiplier = Math.min(2.2, speedMultiplier + 0.2);
        syncBodyDataset();
      } else if (e.key === 'ArrowDown' || e.key === '-') {
        speedMultiplier = Math.max(0.25, speedMultiplier - 0.2);
        syncBodyDataset();
      } else if (e.key === '1') {
        setCameraMode('overview');
      } else if (e.key === '2' || e.key === 'c' || e.key === 'C') {
        setCameraMode('follow');
      } else if (e.key === '3') {
        setCameraMode('top');
      } else if (e.key === '4') {
        setCameraMode('tunnel');
      } else if (e.key === '5' || e.key === 'p' || e.key === 'P') {
        setCameraMode(cameraMode === 'pov' ? 'overview' : 'pov');
      }
    };
    window.addEventListener('keydown', onKeyDown);

    const onResize = () => {
      const aspect = window.innerWidth / window.innerHeight;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
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
    const pipBorderColor = new THREE.Color(0xc98769);

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

      // Pass 2: Live WebGL Picture-in-Picture (PiP) Viewport in bottom-right corner
      if (showPipViewport) {
        const pip = computePipRect(fullW, fullH);
        const border = 2;

        // Draw subtle natural copper border frame via WebGL scissor clear
        renderer.setViewport(
          pip.x - border,
          pip.y - border,
          pip.width + border * 2,
          pip.height + border * 2
        );
        renderer.setScissor(
          pip.x - border,
          pip.y - border,
          pip.width + border * 2,
          pip.height + border * 2
        );
        const prevClearColor = renderer.getClearColor(new THREE.Color());
        const prevClearAlpha = renderer.getClearAlpha();
        renderer.setClearColor(pipBorderColor, 1);
        renderer.clearColor();
        renderer.setClearColor(prevClearColor, prevClearAlpha);

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

        renderer.setViewport(pip.x, pip.y, pip.width, pip.height);
        renderer.setScissor(pip.x, pip.y, pip.width, pip.height);
        renderer.clear();
        renderer.render(scene, pipCamera);
      }

      renderer.setScissorTest(false);
    }

    function tick() {
      if (disposed) return;
      const now = performance.now();
      const rawDt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;
      const dt = isSoftwareGl ? 1 / 30 : rawDt;

      if (!isPaused && !isDraggingTrain) {
        elapsedTime += dt;
        simState = stepTrainPhysics(simState, dt, speedMultiplier);
        coilRig.updateActiveTurns(
          simState,
          elapsedTime,
          dt,
          showExplicitFields
        );
        trainRig.updateTrainPose(simState, elapsedTime, showExplicitFields);

        datasetTickCounter++;
        if (datasetTickCounter % 8 === 0) {
          syncBodyDataset();
        }
      } else if (isDraggingTrain) {
        elapsedTime += dt;
        coilRig.updateActiveTurns(
          simState,
          elapsedTime,
          dt,
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
          .addScaledVector(frame.tangent, -0.45 * simState.direction);

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
          .addScaledVector(frame.tangent, 1.15 * simState.direction)
          .addScaledVector(frame.binormal, 1.05)
          .addScaledVector(frame.normal, 0.58);

        const dampFactor = 1 - Math.exp(-7.0 * dt);
        camera.position.lerp(desiredCamPos, dampFactor);
        controls.target.lerp(desiredTarget, dampFactor);
        controls.update();
      } else {
        controls.update();
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

  return (
    <main
      aria-label="Simulación 3D del tren electromagnético homopolar"
      className="relative w-screen h-screen overflow-hidden bg-[#06090f] m-0 p-0 select-none"
    >
      <div ref={mountRef} className="w-full h-full" />

      {/* iOS / visionOS Translucent Frosted-Glass Control Bar (Bottom-Left) */}
      <section
        aria-label="Controles de laboratorio del tren electromagnético"
        className="fixed bottom-4 left-4 z-10 flex flex-col gap-2 bg-slate-900/72 backdrop-blur-2xl border border-white/25 rounded-[22px] px-3.5 py-2.5 text-white shadow-[0_16px_40px_rgba(15,23,42,0.28),inset_0_1px_0_rgba(255,255,255,0.28)] max-w-[calc(100vw-205px)] sm:max-w-[630px]"
      >
        {/* Top Row: Clean Unboxed Live Telemetry + Interactive Drag Hint */}
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] border-b border-white/12 pb-1.5 px-0.5">
          <div className="flex items-center gap-3 font-mono tracking-tight">
            <span className="inline-flex items-center gap-1.5 text-sky-300">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              <span>
                I = <strong className="font-semibold text-white">{uiState.currentAmps.toFixed(2)} A</strong>
              </span>
            </span>
            <span className="text-white/25" aria-hidden="true">·</span>
            <span className="inline-flex items-center gap-1.5 text-amber-300">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span>
                B = <strong className="font-semibold text-white">{uiState.magneticFieldmT.toFixed(1)} mT</strong>
              </span>
            </span>
            <span className="text-white/25" aria-hidden="true">·</span>
            <span className="inline-flex items-center gap-1.5 text-emerald-300">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>
                F = <strong className="font-semibold text-white">{uiState.propulsionForceN.toFixed(2)} N</strong>
              </span>
            </span>
          </div>

          <span className="hidden sm:inline-flex items-center gap-1.5 text-[10.5px] text-white/70 font-medium">
            <Sun className="w-3 h-3 text-amber-300 stroke-[2.2]" />
            <MoveHorizontal className="w-3.5 h-3.5 text-sky-300 stroke-[2]" />
            <span>Arrastra la pila · Shift+Clic en carril</span>
          </span>
        </div>

        {/* Bottom Row: iOS Tactile Buttons & Segmented Controls (Icons Only + Text, Zero Emojis) */}
        <div className="flex flex-wrap items-center gap-1.5 text-[11.5px]">
          {/* Play / Pause Action Pill */}
          <button
            type="button"
            onClick={() => apiRef.current?.togglePause()}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[12px] font-medium transition-all duration-150 cursor-pointer ${
              uiState.isPaused
                ? 'bg-white text-slate-900 shadow-sm'
                : 'bg-white/14 hover:bg-white/22 text-white border border-white/15'
            }`}
          >
            {uiState.isPaused ? (
              <Play className="w-3.5 h-3.5 fill-current stroke-[2]" />
            ) : (
              <Pause className="w-3.5 h-3.5 fill-current stroke-[2]" />
            )}
            <span>{uiState.isPaused ? 'Reanudar' : 'Pausa'}</span>
          </button>

          {/* Polarity Inversion Button */}
          <button
            type="button"
            onClick={() => apiRef.current?.reversePolarity()}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[12px] bg-white/14 hover:bg-white/22 border border-white/15 text-amber-200 font-medium transition-all duration-150 cursor-pointer"
            title="Invertir polaridad de la pila AA (±1.5V)"
          >
            <ArrowLeftRight className="w-3.5 h-3.5 stroke-[2]" />
            <span>{uiState.direction === 1 ? '+1.5V' : '-1.5V'}</span>
          </button>

          {/* Magnetic Fields & Electron Flow iOS Toggle */}
          <button
            type="button"
            onClick={() => apiRef.current?.toggleFields()}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[12px] font-medium transition-all duration-150 cursor-pointer ${
              uiState.showExplicitFields
                ? 'bg-sky-500/90 hover:bg-sky-500 text-white shadow-[0_2px_10px_rgba(14,165,233,0.4)] border border-sky-300/50'
                : 'bg-white/10 hover:bg-white/18 border border-white/12 text-white/65'
            }`}
          >
            <Magnet className="w-3.5 h-3.5 stroke-[2]" />
            <span>{uiState.showExplicitFields ? 'Campos y e- ON' : 'Campos y e- OFF'}</span>
          </button>

          {/* iOS Segmented Control: Camera Perspectives */}
          <div
            role="group"
            aria-label="Perspectiva de cámara"
            className="inline-flex items-center bg-black/30 p-0.5 rounded-[12px] border border-white/10"
          >
            {(
              [
                { mode: 'overview', label: 'General', Icon: Eye },
                { mode: 'follow', label: 'Seguir', Icon: Navigation },
                { mode: 'pov', label: '1ª Persona POV', Icon: ScanEye },
                { mode: 'top', label: 'Cenital', Icon: Compass },
              ] as const
            ).map(({ mode, label, Icon }) => {
              const active = uiState.cameraMode === mode;
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => apiRef.current?.setCameraMode(mode)}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-[9px] text-[11px] transition-all duration-150 cursor-pointer ${
                    active
                      ? 'bg-white text-slate-900 font-semibold shadow-sm'
                      : 'text-white/75 hover:text-white font-medium'
                  }`}
                >
                  <Icon className="w-3 h-3 stroke-[2]" />
                  <span>{label}</span>
                </button>
              );
            })}
          </div>

          {/* iOS Segmented Control: Simulation Speed */}
          <div
            role="group"
            aria-label="Velocidad de simulación"
            className="inline-flex items-center bg-black/30 p-0.5 rounded-[12px] border border-white/10"
          >
            <span className="pl-1.5 pr-1 text-white/55" aria-hidden="true">
              <Gauge className="w-3 h-3 stroke-[2]" />
            </span>
            {[
              { label: '0.35×', val: 0.35 },
              { label: '0.55× Real', val: 0.55 },
              { label: '1.0×', val: 1.0 },
            ].map((s) => {
              const active = Math.abs(uiState.speedMultiplier - s.val) < 0.05;
              return (
                <button
                  key={s.label}
                  type="button"
                  onClick={() => apiRef.current?.setSpeed(s.val)}
                  className={`px-2 py-1 rounded-[9px] text-[10.5px] transition-all duration-150 cursor-pointer ${
                    active
                      ? 'bg-emerald-400 text-slate-950 font-semibold shadow-sm'
                      : 'text-white/70 hover:text-white font-medium'
                  }`}
                >
                  {s.label}
                </button>
              );
            })}
          </div>
        </div>
      </section>
    </main>
  );
}
