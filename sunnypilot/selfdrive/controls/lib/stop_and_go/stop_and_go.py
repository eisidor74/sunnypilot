"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL


LongPersonality = log.LongitudinalPersonality
PARAM = "StopAndGoComfort"

_PARAM_REFRESH_FRAMES = max(1, int(1.0 / DT_MDL))

_ACTIVE_SPEED = 4.5
_BRAKE_BYPASS_ACCEL = -0.85
_BRAKE_BYPASS_DREL = 3.0
_BRAKE_BYPASS_VREL = -1.5
_BRAKE_BYPASS_TTC = 2.0

_CRAWL_DREL_BP = [0.0, 2.0, 4.0, 8.0, 14.0, 20.0]
_CRAWL_A_MAX_V = [-0.10, 0.00, 0.12, 0.28, 0.55, 1.20]
_CRAWL_SPEED_SCALE_BP = [0.0, 2.0, _ACTIVE_SPEED]
_CRAWL_SPEED_SCALE_V = [0.55, 0.75, 1.00]
_TAKEOFF_SPEED = 1.5
_TAKEOFF_CLEAR_DREL = 6.0
_TAKEOFF_MIN_VREL = -0.2
_TAKEOFF_A_MIN = {
  LongPersonality.aggressive: 0.55,
  LongPersonality.standard: 0.42,
  LongPersonality.relaxed: 0.30,
}

_LOW_SPEED_POSITIVE_RISE = {
  LongPersonality.aggressive: 1.20,
  LongPersonality.standard: 0.85,
  LongPersonality.relaxed: 0.65,
}
_ZERO_DEADBAND = 0.08


class StopAndGoComfortController:
  def __init__(self):
    self.params = Params()
    self._frame = 0
    self._enabled = self.params.get_bool(PARAM)

    self._last_output = 0.0
    self._first = True

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool) -> None:
    self._enabled = bool(enabled)
    self.params.put_bool(PARAM, self._enabled)
    if not self._enabled:
      self.reset()

  def reset(self) -> None:
    self._last_output = 0.0
    self._first = True

  def update(self) -> None:
    self._frame += 1
    if self._frame % _PARAM_REFRESH_FRAMES == 0:
      enabled = self.params.get_bool(PARAM)
      if enabled != self._enabled:
        self._enabled = enabled
        if not self._enabled:
          self.reset()

  def apply(self, sm, a_target: float, should_stop: bool) -> float:
    self.update()
    if not self._enabled:
      return a_target

    state = self._read_state(sm)
    if not state["long_enabled"]:
      self.reset()
      return a_target

    v_ego = state["v_ego"]
    lead = state["lead"]
    personality = state["personality"]

    if self._should_bypass_for_braking(a_target, lead):
      self._last_output = min(self._last_output, a_target) if not self._first else a_target
      self._first = False
      return a_target

    holding_stop = should_stop or state["brake_pressed"] or state["cruise_standstill"]
    out = float(a_target)

    if not holding_stop:
      out = self._apply_takeoff_floor(out, lead, v_ego, personality)

    if not holding_stop and lead is not None and v_ego <= _ACTIVE_SPEED:
      out = self._cap_crawl_accel(out, lead, v_ego)

    if not holding_stop and self._in_low_speed_band(v_ego, lead):
      out = self._apply_zero_deadband(out)
      out = self._limit_positive_rise(out, personality)

    self._last_output = out
    self._first = False
    return out

  @staticmethod
  def _read_state(sm) -> dict:
    car_state = sm["carState"]
    car_control = sm["carControl"]
    radar_state = sm["radarState"]
    selfdrive_state = sm["selfdriveState"]

    lead = radar_state.leadOne if getattr(radar_state.leadOne, "status", False) else None
    return {
      "v_ego": float(car_state.vEgo),
      "standstill": bool(car_state.standstill),
      "brake_pressed": bool(car_state.brakePressed),
      "cruise_standstill": bool(car_state.cruiseState.standstill),
      "long_enabled": bool(car_control.enabled),
      "lead": lead,
      "personality": selfdrive_state.personality,
    }

  def _should_bypass_for_braking(self, a_target: float, lead) -> bool:
    if a_target <= _BRAKE_BYPASS_ACCEL:
      return True
    if lead is None or a_target > 0.0:
      return False

    d_rel = float(lead.dRel)
    v_rel = float(lead.vRel)
    ttc = d_rel / max(0.1, -v_rel) if v_rel < 0.0 else float("inf")
    return d_rel <= _BRAKE_BYPASS_DREL or v_rel <= _BRAKE_BYPASS_VREL or ttc <= _BRAKE_BYPASS_TTC

  @staticmethod
  def _apply_takeoff_floor(a_target: float, lead, v_ego: float, personality: LongPersonality) -> float:
    if v_ego > _TAKEOFF_SPEED or a_target <= 0.0:
      return a_target
    if lead is not None and (float(lead.dRel) < _TAKEOFF_CLEAR_DREL or float(lead.vRel) < _TAKEOFF_MIN_VREL):
      return a_target
    return max(a_target, _TAKEOFF_A_MIN[personality])

  @staticmethod
  def _cap_crawl_accel(a_target: float, lead, v_ego: float) -> float:
    if a_target <= 0.0:
      return a_target
    d_rel = float(lead.dRel)
    crawl_cap = float(np.interp(d_rel, _CRAWL_DREL_BP, _CRAWL_A_MAX_V))
    speed_scale = float(np.interp(v_ego, _CRAWL_SPEED_SCALE_BP, _CRAWL_SPEED_SCALE_V))
    return min(a_target, crawl_cap * speed_scale)

  @staticmethod
  def _apply_zero_deadband(a_target: float) -> float:
    if abs(a_target) <= _ZERO_DEADBAND:
      return 0.0
    return a_target

  def _limit_positive_rise(self, a_target: float, personality: LongPersonality) -> float:
    if self._first or a_target <= self._last_output:
      return a_target
    return min(a_target, self._last_output + _LOW_SPEED_POSITIVE_RISE[personality] * DT_MDL)

  @staticmethod
  def _in_low_speed_band(v_ego: float, lead) -> bool:
    return lead is not None and (v_ego <= _ACTIVE_SPEED or float(lead.dRel) <= _CRAWL_DREL_BP[-1])
