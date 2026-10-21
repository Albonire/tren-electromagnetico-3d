import * as THREE from 'three';

/**
 * Generates a 2K high-resolution AA/AAA battery wrapper albedo texture matching the black,
 * brushed silver, and copper-gold battery in the reference photo.
 */
export function createBatteryWrapperTexture(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d')!;

  // Base deep satin carbon-black wrapper
  const blackGrad = ctx.createLinearGradient(0, 0, 2048, 0);
  blackGrad.addColorStop(0, '#090b0e');
  blackGrad.addColorStop(0.5, '#14171f');
  blackGrad.addColorStop(1, '#090b0e');
  ctx.fillStyle = blackGrad;
  ctx.fillRect(0, 0, 2048, 1024);

  // Brushed silver-platinum metallic foil band across middle-upper body
  const silverGrad = ctx.createLinearGradient(0, 190, 2048, 190);
  silverGrad.addColorStop(0, '#b8c0cc');
  silverGrad.addColorStop(0.2, '#f1f5f9');
  silverGrad.addColorStop(0.45, '#94a3b8');
  silverGrad.addColorStop(0.72, '#f8fafc');
  silverGrad.addColorStop(1, '#b8c0cc');
  ctx.fillStyle = silverGrad;
  ctx.fillRect(0, 190, 2048, 490);

  // Metallic copper-gold accent trim rings
  const goldGrad = ctx.createLinearGradient(0, 0, 2048, 0);
  goldGrad.addColorStop(0, '#b45309');
  goldGrad.addColorStop(0.25, '#fbbf24');
  goldGrad.addColorStop(0.5, '#d97706');
  goldGrad.addColorStop(0.75, '#fde68a');
  goldGrad.addColorStop(1, '#b45309');
  ctx.fillStyle = goldGrad;
  ctx.fillRect(0, 160, 2048, 30);
  ctx.fillRect(0, 680, 2048, 24);

  // Fine anisotropic circumferential foil grain lines
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1.5;
  for (let y = 196; y < 675; y += 8) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(2048, y);
    ctx.stroke();
  }

  // Draw crisp technical typography on 3 radial sides so readable from every angle
  for (const centerX of [360, 1024, 1688]) {
    ctx.save();
    ctx.translate(centerX, 512);
    ctx.rotate(-Math.PI / 2);

    // Positive '+' terminal badge near front end
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(395, 0, 42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#090b0e';
    ctx.font = 'bold 64px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('+', 395, 3);

    // Negative '−' terminal badge near rear end
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(-395, 0, 42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#090b0e';
    ctx.font = 'bold 68px sans-serif';
    ctx.fillText('−', -395, 3);

    // Primary specification header on silver band
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 48px sans-serif';
    ctx.fillText('ALKALINE 1.5V LR6', 36, -64);

    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 32px monospace';
    ctx.fillText('MOTOR LINEAL HOMOPOLAR', 36, -6);

    // Framed technical spec table
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 3.5;
    ctx.strokeRect(-175, 34, 420, 68);
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 28px monospace';
    ctx.fillText('I = 1.8A • NdFeB N52', 35, 68);

    // Crossed-out wheeled bin icon on the black lower band (matching reference photo)
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 4.5;
    ctx.strokeRect(-310, -44, 64, 80);
    ctx.beginPath();
    ctx.moveTo(-324, -52);
    ctx.lineTo(-232, 48);
    ctx.moveTo(-232, -52);
    ctx.lineTo(-324, 48);
    ctx.stroke();

    ctx.restore();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a matching PBR roughness map for the battery wrapper so the metallic silver band
 * and gold rings have much sharper specular highlights than the satin black sleeve.
 */
export function createBatteryRoughnessMap(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Satin black sleeve (roughness ~ 0.42 -> gray 107)
  ctx.fillStyle = 'rgb(107, 107, 107)';
  ctx.fillRect(0, 0, 512, 512);

  // Brushed silver band (roughness ~ 0.16 -> gray 41)
  ctx.fillStyle = 'rgb(41, 41, 41)';
  ctx.fillRect(0, 95, 512, 245);

  // Polished gold trim rings (roughness ~ 0.08 -> gray 20)
  ctx.fillStyle = 'rgb(20, 20, 20)';
  ctx.fillRect(0, 80, 512, 15);
  ctx.fillRect(0, 340, 512, 12);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a circular laser-engraved Neodymium magnet pole cap texture ('N' or 'S')
 * with concentric lathe-machined nickel rings.
 */
export function createMagnetPoleCapTexture(
  pole: 'N' | 'S',
  maxAnisotropy = 4
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  const radGrad = ctx.createRadialGradient(256, 256, 10, 256, 256, 256);
  radGrad.addColorStop(0, '#f8fafc');
  radGrad.addColorStop(0.7, '#e2e8f0');
  radGrad.addColorStop(0.88, pole === 'N' ? '#ef4444' : '#3b82f6');
  radGrad.addColorStop(1, '#cbd5e1');
  ctx.fillStyle = radGrad;
  ctx.fillRect(0, 0, 512, 512);

  // Concentric micro-lathe rings
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.12)';
  ctx.lineWidth = 2;
  for (let r = 24; r < 220; r += 14) {
    ctx.beginPath();
    ctx.arc(256, 256, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Engraved pole letter in center
  ctx.fillStyle = pole === 'N' ? '#dc2626' : '#2563eb';
  ctx.font = 'bold 170px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(pole, 256, 264);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a fine procedural bump/roughness map for brushed nickel magnets and drawn bare copper/silver wire.
 */
export function createBrushedMetalBumpTexture(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 512, 512);

  // Circumferential + longitudinal wire drawing micro-striations
  for (let y = 0; y < 512; y += 2) {
    const lum = 110 + Math.floor(Math.sin(y * 0.85) * 22 + ((y * 19) % 23));
    ctx.fillStyle = `rgb(${lum},${lum},${lum})`;
    ctx.fillRect(0, y, 512, 2);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  for (let x = 0; x < 512; x += 4) {
    ctx.fillRect(x, 0, 1, 512);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a natural, non-saturated drawn electrolytic copper albedo texture (C11000 copper tone)
 * with subtle longitudinal drawing striations and realistic metallic rose-amber warmth.
 */
export function createNaturalCopperAlbedoTexture(
  maxAnisotropy = 4
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createLinearGradient(0, 0, 512, 512);
  grad.addColorStop(0, '#c98769');
  grad.addColorStop(0.28, '#dda185');
  grad.addColorStop(0.55, '#be7a5b');
  grad.addColorStop(0.82, '#e2ab90');
  grad.addColorStop(1, '#c98769');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);

  // Subtle drawn-copper micro-striations
  for (let y = 0; y < 512; y += 3) {
    const alpha = 0.03 + 0.03 * Math.sin(y * 0.7);
    ctx.fillStyle =
      y % 6 === 0
        ? `rgba(255, 235, 220, ${alpha})`
        : `rgba(90, 45, 25, ${alpha})`;
    ctx.fillRect(0, y, 512, 1.5);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(12, 2);
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}


/**
 * Generates a 2K dark slate/obsidian laboratory workbench texture with a dramatic diagonal
 * patch of warm golden sunlight streaming through a nearby window (complete with window-pane
 * mullions and soft penumbra) while keeping the surrounding workbench deep and dark.
 */
export function createTabletopAlbedoTexture(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 2048;
  const ctx = canvas.getContext('2d')!;

  // 1. Deep dark obsidian-slate laboratory surface base
  const baseGrad = ctx.createRadialGradient(1140, 960, 80, 1024, 1024, 1400);
  baseGrad.addColorStop(0, '#101622');
  baseGrad.addColorStop(0.48, '#0a0e17');
  baseGrad.addColorStop(1, '#05070c');
  ctx.fillStyle = baseGrad;
  ctx.fillRect(0, 0, 2048, 2048);

  // 2. Subtle precision laboratory drafting grid (5 cm squares) on the dark slate bench
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.048)';
  ctx.lineWidth = 1.2;
  const step = 64;
  for (let x = 0; x <= 2048; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 2048);
    ctx.stroke();
  }
  for (let y = 0; y <= 2048; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(2048, y);
    ctx.stroke();
  }

  // 3. Localized diagonal window-pane sunlight gobo streaming from the nearby upper-right window
  ctx.save();
  ctx.translate(1290, 860);
  ctx.rotate(-0.46);
  ctx.filter = 'blur(28px)';

  // Soft golden ambient glow around the window projection on the dark slate table
  const windowHalo = ctx.createRadialGradient(0, 0, 30, 0, 0, 620);
  windowHalo.addColorStop(0, 'rgba(255, 190, 105, 0.16)');
  windowHalo.addColorStop(0.55, 'rgba(245, 158, 66, 0.06)');
  windowHalo.addColorStop(1, 'rgba(6, 9, 15, 0.0)');
  ctx.fillStyle = windowHalo;
  ctx.fillRect(-700, -700, 1400, 1400);

  // 4 distinct sunlit window panes cast diagonally across the right/center of the dark workbench with soft optical penumbra
  const panes = [
    { x: -390, y: -320, w: 345, h: 275 },
    { x: 15, y: -320, w: 345, h: 275 },
    { x: -390, y: 15, w: 345, h: 290 },
    { x: 15, y: 15, w: 345, h: 290 },
  ];

  for (const pane of panes) {
    const paneGrad = ctx.createLinearGradient(
      pane.x,
      pane.y,
      pane.x + pane.w,
      pane.y + pane.h
    );
    paneGrad.addColorStop(0, 'rgba(255, 208, 132, 0.21)');
    paneGrad.addColorStop(0.5, 'rgba(251, 178, 88, 0.13)');
    paneGrad.addColorStop(1, 'rgba(217, 119, 46, 0.05)');
    ctx.fillStyle = paneGrad;
    ctx.beginPath();
    ctx.roundRect(pane.x, pane.y, pane.w, pane.h, 24);
    ctx.fill();
  }
  ctx.filter = 'none';
  ctx.restore();

  // 4. Deep dark vignette around the outer perimeter of the workbench so the room stays dark & moody
  const edgeVig = ctx.createRadialGradient(1160, 920, 280, 1024, 1024, 1280);
  edgeVig.addColorStop(0, 'rgba(5, 7, 12, 0.0)');
  edgeVig.addColorStop(0.62, 'rgba(5, 7, 12, 0.48)');
  edgeVig.addColorStop(1, 'rgba(3, 5, 9, 0.88)');
  ctx.fillStyle = edgeVig;
  ctx.fillRect(0, 0, 2048, 2048);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a subtle matte micro-grain bump texture for the dark slate laboratory tabletop.
 */
export function createTabletopMicroTexture(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 512, 512);

  const imgData = ctx.getImageData(0, 0, 512, 512);
  const data = imgData.data;
  let seed = 1337;
  for (let i = 0; i < data.length; i += 4) {
    seed = (seed * 16807) % 2147483647;
    const delta = (seed & 15) - 7;
    const v = 128 + delta;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(16, 16);
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a high-resolution contact shadow + warm sunlit copper caustic reflection ribbon
 * under the 340-turn helical coil on the dark slate workbench.
 */
export function createCoilRibbonShadowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, 256, 1024);

  const imgData = ctx.getImageData(0, 0, 256, 1024);
  const data = imgData.data;

  for (let y = 0; y < 1024; y++) {
    const v = y / 1023;
    const endFade = Math.min(1, v / 0.025) * Math.min(1, (1 - v) / 0.025);
    const turnWave = 0.72 + 0.28 * Math.cos(v * Math.PI * 340 * 2);

    for (let x = 0; x < 256; x++) {
      // Shift shadow toward -u to match diagonal window sunlight from upper-right
      const u = (x / 255) * 2 - 1;
      const shadowCenter = u + 0.16;
      const coreShadow = Math.exp(-shadowCenter * shadowCenter * 5.8);
      const ambientShadow = Math.exp(-u * u * 2.2) * 0.42;

      // Warm copper caustic reflection band on the dark table
      const causticOffset = u - 0.24;
      const copperCaustic =
        Math.exp(-causticOffset * causticOffset * 14.0) * turnWave;

      const shadowAlpha = Math.min(
        1,
        (coreShadow * turnWave * 0.75 + ambientShadow) * endFade
      );
      const causticWeight = Math.min(1, copperCaustic * 0.34 * endFade);

      const idx = (y * 256 + x) * 4;
      data[idx] = Math.round(4 * (1 - causticWeight) + 218 * causticWeight);
      data[idx + 1] = Math.round(6 * (1 - causticWeight) + 130 * causticWeight);
      data[idx + 2] = Math.round(10 * (1 - causticWeight) + 78 * causticWeight);
      data[idx + 3] = Math.round(
        255 * Math.max(shadowAlpha, causticWeight * 0.62)
      );
    }
  }

  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates a soft radial contact shadow texture for grounding the battery train on the dark table.
 */
export function createSoftShadowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createRadialGradient(128, 128, 8, 128, 128, 124);
  grad.addColorStop(0, 'rgba(2, 4, 8, 0.78)');
  grad.addColorStop(0.5, 'rgba(4, 7, 13, 0.36)');
  grad.addColorStop(1, 'rgba(4, 7, 13, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a dark, moody physics laboratory backdrop where the room is in deep shadow
 * except for a soft out-of-focus side window glow on the upper-right.
 */
export function createStudioBackdropTexture(maxAnisotropy = 4): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // 1. Deep dark obsidian-slate laboratory wall
  const wallGrad = ctx.createLinearGradient(0, 0, 1024, 512);
  wallGrad.addColorStop(0, '#04060a');
  wallGrad.addColorStop(0.55, '#070b12');
  wallGrad.addColorStop(0.85, '#0d1320');
  wallGrad.addColorStop(1, '#060910');
  ctx.fillStyle = wallGrad;
  ctx.fillRect(0, 0, 1024, 512);

  // 2. Soft out-of-focus warm golden window glow on the upper-right of the dark laboratory wall
  const windowGlow = ctx.createRadialGradient(920, 110, 10, 820, 180, 520);
  windowGlow.addColorStop(0, 'rgba(255, 208, 135, 0.28)');
  windowGlow.addColorStop(0.38, 'rgba(245, 158, 66, 0.10)');
  windowGlow.addColorStop(1, 'rgba(6, 9, 15, 0.0)');
  ctx.fillStyle = windowGlow;
  ctx.fillRect(0, 0, 1024, 512);

  // 3. Dark horizon shadow along the bottom edge where the wall meets the back of the bench
  const bottomVig = ctx.createLinearGradient(0, 380, 0, 512);
  bottomVig.addColorStop(0, 'rgba(5, 7, 12, 0.0)');
  bottomVig.addColorStop(1, 'rgba(4, 6, 10, 0.88)');
  ctx.fillStyle = bottomVig;
  ctx.fillRect(0, 380, 1024, 132);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a soft feathered volumetric sunbeam shaft texture for diagonal window rays in the dark room.
 */
export function createWindowSunbeamTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, 512, 512);

  const imgData = ctx.getImageData(0, 0, 512, 512);
  const data = imgData.data;

  for (let y = 0; y < 512; y++) {
    const v = y / 511; // 0 = window top, 1 = table surface
    const longitudinalFade =
      Math.pow(Math.sin(v * Math.PI), 0.85) * (1 - v * 0.35);

    for (let x = 0; x < 512; x++) {
      const u = (x / 511) * 2 - 1; // -1 .. +1 across beam width
      // Multi-shaft window mullion modulation across the sunbeam
      const mullionDip =
        1 - 0.55 * Math.exp(-Math.pow((u - 0.05) * 9.5, 2));
      const transverseProfile = Math.exp(-u * u * 3.8) * mullionDip;
      const alpha = longitudinalFade * transverseProfile;

      const idx = (y * 512 + x) * 4;
      data[idx] = 255;
      data[idx + 1] = 212;
      data[idx + 2] = 142;
      data[idx + 3] = Math.round(255 * Math.min(1, alpha));
    }
  }

  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates a compact micro-scale 3D label sprite for the First-Person POV camera inside the
 * helical coil tunnel (layer 2), styled to match the HTML glass labels of the external views.
 */
export function createPovMicroLabelSprite(
  badge: string,
  label: string,
  accentHex: string
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 680;
  canvas.height = 116;
  const ctx = canvas.getContext('2d')!;
  const fontStack =
    '-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", sans-serif';

  ctx.clearRect(0, 0, 680, 116);

  // Neutral dark glass capsule with a single hairline border
  ctx.fillStyle = 'rgba(28, 28, 30, 0.82)';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(8, 8, 664, 100, 50);
  ctx.fill();
  ctx.stroke();

  // Accent symbol rendered as tinted text, no filled badge
  ctx.fillStyle = accentHex;
  ctx.font = `600 34px ${fontStack}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(badge, 62, 59);

  ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
  ctx.font = `500 30px ${fontStack}`;
  ctx.textAlign = 'left';
  ctx.fillText(label, 104, 59);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });

  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.155, 0.0265, 1);
  sprite.layers.set(2);
  sprite.renderOrder = 30;
  return sprite;
}

/**
 * Creates a high-contrast, precision-engraved laboratory millimeter ruler ("Regleta Milimetrada")
 * texture mounted on the 3D physical regleta plate between the two open mouths of the copper coil.
 */
export function createPrecisionWorkbenchScaleTexture(
  maxAnisotropy: number
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, 1024, 512);

  // 1. Machined dark-anodized titanium/slate ruler plate with metallic sheen
  const bodyGrad = ctx.createLinearGradient(0, 28, 0, 484);
  bodyGrad.addColorStop(0, '#1e293b');
  bodyGrad.addColorStop(0.18, '#0f172a');
  bodyGrad.addColorStop(0.5, '#172234');
  bodyGrad.addColorStop(0.82, '#0f172a');
  bodyGrad.addColorStop(1, '#1e293b');
  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.roundRect(16, 28, 992, 456, 22);
  ctx.fill();

  // Outer machined brass/amber bezel + inner cyan calibration hairline
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.72)';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(18, 30, 988, 452, 20);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(28, 40, 968, 432, 14);
  ctx.stroke();

  // 2. Central Coaxial Guide Channel between the two copper coil mouths
  const channelGrad = ctx.createLinearGradient(0, 180, 0, 332);
  channelGrad.addColorStop(0, 'rgba(2, 6, 23, 0.75)');
  channelGrad.addColorStop(0.5, 'rgba(15, 23, 42, 0.55)');
  channelGrad.addColorStop(1, 'rgba(2, 6, 23, 0.75)');
  ctx.fillStyle = channelGrad;
  ctx.fillRect(36, 180, 952, 152);

  // Dashed cyan coaxial centerline (y = 256)
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.78)';
  ctx.lineWidth = 2.5;
  ctx.setLineDash([14, 10]);
  ctx.beginPath();
  ctx.moveTo(54, 256);
  ctx.lineTo(970, 256);
  ctx.stroke();
  ctx.restore();

  // Left & Right Coil Mouth Alignment Reticles (Boquilla Salida at -30mm, Boquilla Entrada at +30mm)
  for (const side of [-1, 1]) {
    const mx = 512 + side * 300;
    ctx.strokeStyle = side < 0 ? '#38bdf8' : '#fbbf24';
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(mx, 165);
    ctx.lineTo(mx, 347);
    ctx.stroke();

    ctx.fillStyle = side < 0 ? 'rgba(56, 189, 248, 0.22)' : 'rgba(251, 191, 36, 0.22)';
    ctx.beginPath();
    ctx.arc(mx, 256, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // 3. High-contrast Millimeter & Centimeter Graduation Ticks along Top & Bottom Ruler Edges (-40 mm to +40 mm)
  ctx.textAlign = 'center';
  for (let mm = -40; mm <= 40; mm += 1) {
    const x = 512 + mm * 11.2;
    const isMajor = mm % 10 === 0;
    const isMid = mm % 5 === 0;

    ctx.strokeStyle = isMajor
      ? '#fbbf24'
      : isMid
        ? '#e2e8f0'
        : 'rgba(148, 163, 184, 0.65)';
    ctx.lineWidth = isMajor ? 3.2 : isMid ? 2.2 : 1.3;

    const topTickLen = isMajor ? 36 : isMid ? 24 : 13;
    // Top edge ticks
    ctx.beginPath();
    ctx.moveTo(x, 42);
    ctx.lineTo(x, 42 + topTickLen);
    ctx.stroke();

    // Bottom edge ticks
    ctx.beginPath();
    ctx.moveTo(x, 470);
    ctx.lineTo(x, 470 - topTickLen);
    ctx.stroke();

    if (isMajor) {
      ctx.fillStyle = mm === 0 ? '#38bdf8' : '#f8fafc';
      ctx.font = '700 22px ui-monospace, SFMono-Regular, Menlo, monospace';
      const label = mm === 0 ? '0 mm' : `${mm > 0 ? '+' : ''}${mm}`;
      ctx.fillText(label, x, 106);
    }
  }

  // 4. Center Bridge Status & Technical Specification Engraving
  ctx.fillStyle = '#38bdf8';
  ctx.font = '700 21px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(
    '◀ BOQUILLA SALIDA          PUENTE COAXIAL (GAP)          BOQUILLA ENTRADA ▶',
    512,
    222
  );

  ctx.fillStyle = '#fde68a';
  ctx.font = '600 19px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(
    'REGLETA MILIMETRADA 1:1  ·  COBRE CU-ETP 99.9%  ·  V = 1.5V DC  ·  NdFeB N52',
    512,
    302
  );

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}
