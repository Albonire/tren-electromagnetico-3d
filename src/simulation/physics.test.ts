import test from 'node:test';
import assert from 'node:assert/strict';
import {
  COIL_START_U,
  COIL_RADIUS_UNITS,
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
  getCoilTurnsPerMeter,
  getForceConstant,
  PHYSICS,
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
  const stateInside = computeElectromagneticState(0.5, 0.4);
  assert.equal(stateInside.inCoil, true);
  assert.equal(stateInside.circuitClosed, true);
  assert.equal(stateInside.coilCoupling, 1);
  assert.ok(stateInside.currentAmps > 1.5 && stateInside.currentAmps < 2.5, `realistic AA short-circuit current, got ${stateInside.currentAmps}`);
  assert.ok(stateInside.propulsionForceN > 0.1 && stateInside.propulsionForceN < 0.5, 'propulsion force is positive and of realistic size');
  assert.ok(stateInside.sparkIntensity > 0.3, 'sliding brush spark intensity is active inside coil');
  const [startTurn, endTurn] = stateInside.activeTurnRange;
  assert.ok(startTurn >= 0 && endTurn > startTurn && endTurn < TOTAL_COIL_TURNS);
  const activeSpan = endTurn - startTurn;
  assert.ok(activeSpan >= 14 && activeSpan <= 32, `expected ~18-24 active turns, got ${activeSpan}`);
});

test('solenoid field follows B = mu0 * n * I', () => {
  const state = computeElectromagneticState(0.5, 0);
  const expectedmT = 4e-7 * Math.PI * getCoilTurnsPerMeter() * state.currentAmps * 1000;
  assert.ok(Math.abs(state.magneticFieldmT - expectedmT) < 1e-9);
  assert.ok(state.magneticFieldmT > 0.5 && state.magneticFieldmT < 2, 'about a millitesla, not tens');
});

test('back-EMF lowers the current as the train speeds up along the force', () => {
  const atRest = computeElectromagneticState(0.5, 0);
  const moving = computeElectromagneticState(0.5, 0.6);
  assert.ok(moving.currentAmps < atRest.currentAmps);
  const expectedDrop = (getForceConstant() * 0.6) / PHYSICS.circuitResistanceOhm;
  assert.ok(Math.abs(atRest.currentAmps - moving.currentAmps - expectedDrop) < 1e-9);
});

test('an open circuit (one magnet over the insulated gap) produces no current and no force', () => {
  const trackLen = getTrackArcLength();
  const halfDeltaU = TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;
  // Front magnet just past the end of the coil, over the gap
  const state = computeElectromagneticState(0.99 - halfDeltaU + 0.01, 0.3);
  assert.equal(state.circuitClosed, false);
  assert.equal(state.currentAmps, 0);
  assert.equal(state.propulsionForceN, 0);
  assert.ok(state.acceleration < 0, 'the train only decelerates while coasting');
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

test('train starts from rest, reaches a terminal velocity and coasts across the gap', () => {
  let state = computeElectromagneticState(0.3, 0);
  for (let step = 0; step < 60 * 6; step++) {
    state = stepTrainPhysics(state, 1 / 60);
  }
  assert.ok(state.distanceMeters > 1, 'train advances along track');
  assert.ok(state.velocity > 0.4 && state.velocity < 1.0, `terminal velocity in a realistic range, got ${state.velocity}`);
  assert.ok(state.lapCount >= 1, 'momentum carries the train across the open gap');
});

test('a train dropped at rest with an open circuit stays put', () => {
  let state = computeElectromagneticState(0.0, 0);
  assert.equal(state.circuitClosed, false);
  for (let step = 0; step < 60; step++) {
    state = stepTrainPhysics(state, 1 / 60);
  }
  assert.equal(state.velocity, 0);
});

test('reversing polarity brakes the train and then drives it the other way', () => {
  let state = computeElectromagneticState(0.3, 0);
  for (let step = 0; step < 240; step++) state = stepTrainPhysics(state, 1 / 60);
  const before = state.velocity;
  assert.ok(before > 0.3);

  state = reverseTrainPolarity(state);
  assert.equal(state.direction, -1);
  assert.equal(state.velocity, before, 'momentum is kept at the instant of the switch');

  state = stepTrainPhysics(state, 1 / 60);
  assert.ok(state.velocity < before && state.velocity > 0, 'first it brakes');

  for (let step = 0; step < 240; step++) state = stepTrainPhysics(state, 1 / 60);
  assert.ok(state.velocity < -0.3, `then it runs backwards, got ${state.velocity}`);
});

test('crossing u = 0 increments the lap counter', () => {
  const nearEnd = computeElectromagneticState(0.9995, 0.6, 0.9, 0, 1);
  const wrapped = stepTrainPhysics(nearEnd, 0.05);
  assert.equal(wrapped.lapCount, 1);
  assert.ok(wrapped.u < 0.1);
});

test('first-person POV camera frame stays inside the helical coil tunnel and steers into curves', () => {
  for (let i = 0; i < 16; i++) {
    const u = i / 16;
    const state = computeElectromagneticState(u, 0.5, 0, 0, 1);
    const pov = computeFirstPersonPovFrame(state);
    assert.ok(pov.radialClearanceRatio > 0.45, 'POV camera maintains safe radial clearance from inner wire walls');
    const lookDir = pov.lookTarget.clone().sub(pov.eyePosition).normalize();
    const frame = evaluateTrackFrame(u);
    assert.ok(lookDir.dot(frame.tangent) > 0.65, 'POV camera looks forward along the coil tunnel');
  }
});

test('copper coil turns physically react with radial expansion and axial peristaltic wave as battery passes', () => {
  const state = computeElectromagneticState(0.5, 0.6, 1.2, 0, 1);
  const trackLen = getTrackArcLength();
  const frontMagnetU = 0.5 + TRAIN_MAGNET_CENTER_OFFSET_UNITS / trackLen;

  const atFrontBrush = computeCoilTurnPerturbation(frontMagnetU, state, 0.4);
  const atBridgeCenter = computeCoilTurnPerturbation(0.5, state, 0.4);
  const farAway = computeCoilTurnPerturbation(0.15, state, 0.4);

  assert.ok(atFrontBrush.radialDelta > 0.001, 'front magnet brush expands copper turn radially');
  assert.ok(atFrontBrush.radialDelta < 0.03 * COIL_RADIUS_UNITS * 10, 'deformation stays around a millimetre');
  assert.ok(atFrontBrush.intensity > 0.9, 'perturbation intensity peaks at magnet brush');
  assert.ok(atBridgeCenter.radialDelta > 0.0005, 'energized solenoid bridge expands slightly under Lorentz hoop stress');
  assert.ok(atBridgeCenter.axialDelta < 0, 'Newton 3rd law recoil shifts bridged turns opposite to forward propulsion');
  assert.equal(farAway.intensity, 0, 'distant coil turns remain unperturbed');
  assert.equal(farAway.radialDelta, 0, 'distant coil turns have zero radial displacement');
});
