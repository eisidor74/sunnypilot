"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from dataclasses import dataclass

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.lead_persistence.lead_persistence import LeadPersistence


_DREL_MIN = 40.0
_DREL_MAX = 180.0
_YREL_ABS_MAX = 1.6

_VREL_DEADBAND = -1.5
_VREL_DEADBAND_FAR = -0.5  # softer deadband for far tracks; catches slow closing at large dRel
_DREL_FAR_THRESH = 80.0    # tracks beyond this use the softer deadband
_VREL_FULL = -6.0
_LEADONE_PROB_MIN = 0.85

_V_EGO_MIN = 3.0
# Static clutter filter (parked cars, signs, curbs): only active at low ego speed
# where bins fill with non-threats. At higher speed, a stationary lead IS a threat.
_STATIC_FILTER_V_EGO_MAX = 5.0
_STATIC_VLEAD_ABS = 0.5

_ACTIVATE_FRAMES = 3
_DECAY_PER_MISS = 2
_DEACTIVATE_FRAMES = 10

_BIN_DREL = 8.0
_BIN_VREL = 2.0

_TTC_NONE = 10.0
_TTC_LIFT = 5.0
_TTC_BRAKE = 3.0

_A_CEIL_HIGH = 0.8
_A_CEIL_COAST = 0.0
_A_CEIL_BRAKE = -0.5
_A_CEIL_RELEASED = 2.5

# Emergency override: when a raw track shows extreme closing in our lane,
# bypass the TTC band stand-down so we keep some brake authority while the
# MPC catches up. Tight yRel + longer persistence guard against
# false-positive merges and sharp-curve roadside tracks.
_EMERGENCY_VREL          = -15.0  # m/s; only true closing
_EMERGENCY_TTC_MAX       = 5.0
_EMERGENCY_YREL_ABS      = 1.0    # tighter than _YREL_ABS_MAX
_EMERGENCY_ACTIVATE_MIN  = 4      # frames in bin before emergency override
_A_CEIL_EMERGENCY        = -1.0   # m/s^2 ceiling under emergency

_A_CEIL_RATE_DOWN = 0.6
_A_CEIL_RATE_UP = 0.8

_PARAM_REFRESH_FRAMES = max(1, int(1.0 / DT_MDL))


@dataclass
class FarLeadState:
  d_rel: float = 0.0
  v_rel: float = 0.0
  y_rel: float = 0.0
  ttc: float = float('inf')
  track_id: int = -1
  frames_seen: int = 0
  frames_lost: int = 0
  active: bool = False


class RadarDistanceController:
  def __init__(self):
    self.params = Params()
    self._frame = 0
    self._enabled = self.params.get_bool('RadarDistance')

    self._state = FarLeadState()
    self._track_persistence: dict[tuple[int, int], int] = {}

    self._ceiling = _A_CEIL_RELEASED
    self._first = True

    self._lead_persistence = LeadPersistence()

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool):
    self._enabled = bool(enabled)
    self.params.put_bool('RadarDistance', self._enabled)

  def toggle(self) -> bool:
    self.set_enabled(not self._enabled)
    return self._enabled

  def update(self, sm=None, sm_sp=None) -> None:
    self._frame += 1
    if self._frame % _PARAM_REFRESH_FRAMES == 0:
      self._enabled = self.params.get_bool('RadarDistance')

    radarstate = None
    if sm is not None:
      try:
        radarstate = sm['radarState']
      except (KeyError, AttributeError, TypeError):
        radarstate = None

    self._lead_persistence.update(radarstate, force_enabled=self._enabled)

    if sm is None or not self._enabled:
      self._release()
      return

    try:
      v_ego = float(sm['carState'].vEgo)
    except Exception:
      v_ego = 0.0

    if radarstate is not None and radarstate.leadOne.status \
        and float(radarstate.leadOne.vRel) <= _VREL_DEADBAND \
        and float(radarstate.leadOne.modelProb) >= _LEADONE_PROB_MIN:
      self._release()
      return

    tracks = self._extract_tracks(sm_sp)
    self._tick_radar(tracks, v_ego)
    self._step_ceiling()

  def smooth_radarstate(self, radarstate):
    if not self._enabled:
      return radarstate
    return self._lead_persistence.smooth(radarstate, force_enabled=True)

  def reset(self) -> None:
    self._state = FarLeadState()
    self._track_persistence.clear()
    self._ceiling = _A_CEIL_RELEASED
    self._first = True

  def get_accel_ceiling(self, v_ego: float) -> float | None:
    if not self._enabled or not self._state.active:
      return None
    return self._ceiling

  @property
  def ttc(self) -> float:
    return self._state.ttc

  @property
  def active(self) -> bool:
    return self._state.active

  @property
  def d_rel(self) -> float:
    return self._state.d_rel

  @property
  def track_id(self) -> int:
    return self._state.track_id

  @property
  def ceiling(self) -> float:
    return self._ceiling

  @staticmethod
  def _extract_tracks(sm_sp):
    if sm_sp is None:
      return []
    try:
      return list(sm_sp['liveTracks'].points)
    except (KeyError, AttributeError, TypeError):
      return []

  @staticmethod
  def _bin(d_rel: float, v_rel: float) -> tuple[int, int]:
    return int(d_rel // _BIN_DREL), int(v_rel // _BIN_VREL)

  def _tick_radar(self, tracks, v_ego: float) -> None:
    if v_ego < _V_EGO_MIN or not tracks:
      self._decay_unseen(seen_keys=set())
      self._lose_track()
      return

    seen_keys: set[tuple[int, int]] = set()
    best: tuple[float, float, float, float, int] | None = None

    for t in tracks:
      if not t.measured:
        continue
      d_rel = float(t.dRel)
      if not (_DREL_MIN < d_rel < _DREL_MAX):
        continue
      y_rel = float(t.yRel)
      if abs(y_rel) > _YREL_ABS_MAX:
        continue
      v_rel = float(t.vRel)
      vrel_gate = _VREL_DEADBAND_FAR if d_rel >= _DREL_FAR_THRESH else _VREL_DEADBAND
      if v_rel >= vrel_gate:
        continue
      # Reject stationary clutter at low ego speed only — bins otherwise saturate
      # with parked cars / signs. At highway speed, stationary lead = real threat.
      if v_ego < _STATIC_FILTER_V_EGO_MAX and abs(v_rel + v_ego) < _STATIC_VLEAD_ABS:
        continue

      key = self._bin(d_rel, v_rel)
      seen_keys.add(key)
      self._track_persistence[key] = self._track_persistence.get(key, 0) + 1

      if self._track_persistence[key] >= _ACTIVATE_FRAMES:
        ttc = d_rel / max(0.1, -v_rel)
        if best is None or ttc < best[0]:
          best = (ttc, d_rel, v_rel, y_rel, int(t.trackId))

    self._decay_unseen(seen_keys)

    if best is None:
      self._lose_track()
      return

    self._state.ttc = best[0]
    self._state.d_rel = best[1]
    self._state.v_rel = best[2]
    self._state.y_rel = best[3]
    self._state.track_id = best[4]
    self._state.frames_seen = self._track_persistence[self._bin(best[1], best[2])]
    self._state.frames_lost = 0
    self._state.active = True

  def _decay_unseen(self, seen_keys: set[tuple[int, int]]) -> None:
    for key in list(self._track_persistence.keys()):
      if key in seen_keys:
        continue
      self._track_persistence[key] -= _DECAY_PER_MISS
      if self._track_persistence[key] <= 0:
        del self._track_persistence[key]

  def _lose_track(self) -> None:
    self._state.frames_seen = 0
    self._state.frames_lost += 1
    if self._state.frames_lost >= _DEACTIVATE_FRAMES:
      self._state.active = False
      self._state.ttc = float('inf')
      self._state.track_id = -1

  def _target_ceiling(self) -> float:
    if not self._state.active:
      return _A_CEIL_RELEASED

    if (self._state.v_rel <= _EMERGENCY_VREL
        and self._state.ttc <= _EMERGENCY_TTC_MAX
        and abs(self._state.y_rel) <= _EMERGENCY_YREL_ABS
        and self._state.frames_seen >= _EMERGENCY_ACTIVATE_MIN):
      return _A_CEIL_EMERGENCY

    ttc = self._state.ttc
    if ttc >= _TTC_NONE:
      return _A_CEIL_RELEASED
    if ttc >= _TTC_LIFT:
      t = float(np.clip((_TTC_NONE - ttc) / (_TTC_NONE - _TTC_LIFT), 0.0, 1.0))
      return (1.0 - t) * _A_CEIL_HIGH + t * _A_CEIL_COAST
    if ttc >= _TTC_BRAKE:
      t = float(np.clip((_TTC_LIFT - ttc) / (_TTC_LIFT - _TTC_BRAKE), 0.0, 1.0))
      closing = float(np.clip(
        (self._state.v_rel - _VREL_DEADBAND) / (_VREL_FULL - _VREL_DEADBAND), 0.0, 1.0))
      return _A_CEIL_COAST + t * closing * (_A_CEIL_BRAKE - _A_CEIL_COAST)
    return _A_CEIL_RELEASED

  def _step_ceiling(self) -> None:
    target = self._target_ceiling()
    if self._first:
      self._ceiling = target
      self._first = False
      return
    rate = _A_CEIL_RATE_DOWN if target < self._ceiling else _A_CEIL_RATE_UP
    step = rate * DT_MDL
    self._ceiling = float(np.clip(target, self._ceiling - step, self._ceiling + step))

  def _release(self) -> None:
    self._track_persistence.clear()
    self._state.frames_seen = 0
    if self._state.frames_lost < _DEACTIVATE_FRAMES:
      self._state.frames_lost += 1
    if self._state.frames_lost >= _DEACTIVATE_FRAMES:
      self._state.active = False
      self._state.ttc = float('inf')
      self._state.track_id = -1
    target = _A_CEIL_RELEASED
    if self._first:
      self._ceiling = target
      self._first = False
      return
    step = _A_CEIL_RATE_UP * DT_MDL
    self._ceiling = float(np.clip(target, self._ceiling - step, self._ceiling + step))
