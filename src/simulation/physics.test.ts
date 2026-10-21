import test from 'node:test';
import assert from 'node:assert/strict';
import {
  COIL_START_U,
  TOTAL_COIL_TURNS,
  TRAIN_MAGNET_CENTER_OFFSET_UNITS,
  getTrackArcLength,
  evaluateTrackFrame,
  uToTurnIndex,
  computeElectromagneticState,
  stepTrainPhysics,
  reverseTrainPolarity,
  computeFirstPersonPovFrame,
  computeCoilTurnPerturbation,
} from './physics';

test('track spline evaluates orthonormal frames and positive curvature around the entire loop', () => {
  for (let i = 0; i <= 20; i++) {
    const u = i / 20;
    const frame = evaluateTrackFrame(u);
    assert.ok(Math.abs(frame.tangent.length() - 1) < 1e-4, 'tangent is unit length');
    assert.ok(Math.abs(frame.normal.length() - 1) < 1e-4, 'normal is unit length');
    assert.ok(Math.abs(frame.binormal.length() - 1) < 1e-4, 'binormal is unit length');
    assert.ok(Math.abs(frame.tangent.dot(frame.normal)) < 1e-4, 'tangent and normal are orthogonal');
    assert.ok(frame.curvature >= 0, 'curvature is non-negative');
  }
});

test('coil gap vs active coil region is accurately identified', () => {
  const gapFrame = evaluateTrackFrame(0.005);
  assert.equal(gapFrame.inCoil, false);
  assert.equal(uToTurnIndex(0.005), -1);
  const midFrame = evaluateTrackFrame(0.5);
  assert.equal(midFrame.inCoil, true);
  const midTurn = uToTurnIndex(0.5);
  assert.ok(midTurn > 150 && midTurn < 190);
});

test('electromagnetic state energizes only the coil turns between the rear and front magnets and computes spark intensity', () => {
  const stateInside = computeElectromagneticState(0.5, 1.1);
  assert.equal(stateInside.inCoil, true);
  assert.equal(stateInside.coilCoupling, 1);
  assert.ok(stateInside.currentAmps > 1.2, 'current flows when inside coil');
  assert.ok(stateInside.magneticFieldmT > 8, 'magnetic field is generated inside bridged turns');
  assert.ok(stateInside.propulsionForceN > 0.25, 'Lorentz propulsion force is positive');
  assert.ok(stateInside.sparkIntensity > 0.3, 'sliding brush spark intensity is active inside coil');
  const [startTurn, endTurn] = stateInside.activeTurnRange;
  assert.ok(startTurn >= 0 && endTurn > startTurn && endTurn < TOTAL_COIL_TURNS);
  const activeSpan = endTurn - startTurn;
  assert.ok(activeSpan >= 14 && activeSpan <= 32, `expected ~18-24 active turns, got ${activeSpan}`);
});

test('uniform Euclidean arc-length parameterization preserves exact magnet-to-battery spacing on tight curves', () => {
  const firstFrame = evaluateTrackFrame(0);
  const secondFrame = evaluateTrackFrame(0.01);
  const refStepLen = firstFrame.position.distanceTo(secondFrame.position);
  for (let i = 0; i < 50; i++) {
    const u0 = i / 50;
    const u1 = u0 + 0.01;
    const f0 = evaluateTrackFrame(u0);
    const f1 = evaluateTrackFrame(u1);
    const stepLen = f0.position.distanceTo(f1.position);
    assert.ok(
      Math.abs(stepLen - refStepLen) / refStepLen < 0.03,
      `non-uniform arc length at u=${u0}: got ${stepLen} vs ${refStepLen}`
    );
  }
});

test('stepping train physics advances position smoothly, supports polarity reversal, and increments lap count', () => {
  let state = computeElectromagneticState(COIL_START_U, 1);
  for (let step = 0; step < 300; step++) {
    state = stepTrainPhysics(state, 1 / 60, 1.5);
  }
  assert.ok(state.distanceMeters > 0.5, 'train advances along track');
  assert.ok(state.speed > 0.8 && state.speed <= 1.85, 'speed stabilizes at terminal velocity');
  const reversed = reverseTrainPolarity(state);
  assert.equal(reversed.direction, -1);
  const afterReverseStep = stepTrainPhysics(reversed, 0.05, 1);
  assert.equal(afterReverseStep.direction, -1);
  const nearEnd = computeElectromagneticState(0.995, 1.4, 0.9, 0, 1);
  const wrapped = stepTrainPhysics(nearEnd, 0.05, 2);
  assert.equal(wrapped.lapCount, 1);
  assert.ok(wrapped.u < 0.1);
});

test('first-person POV camera frame stays inside the helical coil tunnel and steers into curves', () => {
  for (let i = 0; i < 16; i++) {
    const u = i / 16;
    const state = computeElectromagneticState(u, 1.25, 0, 0, 1);
    const pov = computeFirstPersonPovFrame(state);
    assert.ok(pov.radialClearanceRatio > 0.45, 'POV camera maintains safe radial clearance from inner wire walls');
    const lookDir = pov.lookTarget.clone().sub(pov.eyePosition).normalize();
    const frame = evaluateTrackFrame(u);
    assert.ok(lookDir.dot(frame.tangent) > 0.65, 'POV camera looks forward along the coil tunnel');
  }
});

test('copper coil turns physically react with radial expansion and axial peristaltic wave as battery passes', () => {
  const state = computeElectromagneticState(0.5, 1.35, 1.2, 0, 1);
  const trackLen = getTrackArcLength();
  const frontMagnetU = 0.5 + TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;

  const atFrontBrush = computeCoilTurnPerturbation(frontMagnetU, state, 0.4);
  const atBridgeCenter = computeCoilTurnPerturbation(0.5, state, 0.4);
  const farAway = computeCoilTurnPerturbation(0.15, state, 0.4);

  assert.ok(atFrontBrush.radialDelta > 0.01, 'front magnet brush expands copper turn radially');
  assert.ok(atFrontBrush.intensity > 0.9, 'perturbation intensity peaks at magnet brush');
  assert.ok(atBridgeCenter.radialDelta > 0.005, 'energized solenoid bridge expands under Lorentz hoop stress');
  assert.ok(atBridgeCenter.axialDelta < 0, 'Newton 3rd law recoil shifts bridged turns opposite to forward propulsion');
  assert.equal(farAway.intensity, 0, 'distant coil turns remain unperturbed');
  assert.equal(farAway.radialDelta, 0, 'distant coil turns have zero radial displacement');
});
