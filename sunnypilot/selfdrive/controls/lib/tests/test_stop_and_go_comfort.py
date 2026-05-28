"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log

from openpilot.sunnypilot.selfdrive.controls.lib.stop_and_go.stop_and_go import StopAndGoComfortController
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


LongPersonality = log.LongitudinalPersonality


class FakeCruiseState:
  def __init__(self, standstill=False):
    self.standstill = standstill


class FakeCarState:
  def __init__(self, v_ego=0.0, standstill=False, brake_pressed=False, cruise_standstill=False):
    self.vEgo = v_ego
    self.standstill = standstill
    self.brakePressed = brake_pressed
    self.cruiseState = FakeCruiseState(cruise_standstill)


class FakeCarControl:
  def __init__(self, enabled=True):
    self.enabled = enabled


class FakeLead:
  def __init__(self, status=True, d_rel=12.0, v_rel=0.0, v_lead=0.0):
    self.status = status
    self.dRel = d_rel
    self.vRel = v_rel
    self.vLead = v_lead


class FakeRadarState:
  def __init__(self, lead=None):
    self.leadOne = lead if lead is not None else FakeLead(status=False)


class FakeSelfdriveState:
  def __init__(self, personality=LongPersonality.standard):
    self.personality = personality


class FakeSM:
  def __init__(self, v_ego=0.0, standstill=False, lead=None, enabled=True, brake_pressed=False,
               cruise_standstill=False, personality=LongPersonality.standard):
    self._data = {
      "carState": FakeCarState(v_ego, standstill, brake_pressed, cruise_standstill),
      "carControl": FakeCarControl(enabled),
      "radarState": FakeRadarState(lead),
      "selfdriveState": FakeSelfdriveState(personality),
    }

  def __getitem__(self, key):
    return self._data[key]


def _make(enabled=True):
  controller = StopAndGoComfortController()
  controller.set_enabled(enabled)
  return controller


class TestStopAndGoComfortController:
  def test_disabled_passthrough(self):
    controller = _make(enabled=False)
    out = controller.apply(FakeSM(v_ego=1.0), 0.8, should_stop=False)
    assert out == 0.8

  def test_no_lead_traffic_light_start_passthrough(self):
    controller = _make()
    controller.apply(FakeSM(v_ego=0.0, standstill=True), 0.0, should_stop=True)

    out = controller.apply(FakeSM(v_ego=0.0, standstill=True), 1.0, should_stop=False)

    assert out == 1.0

  def test_no_lead_takeoff_gets_min_accel_once_go_allowed(self):
    controller = _make()

    out = controller.apply(FakeSM(v_ego=0.0, standstill=True), 0.05, should_stop=False)

    assert out > 0.05

  def test_no_lead_low_speed_accel_passthrough(self):
    controller = _make()
    sm = FakeSM(v_ego=2.0)

    controller.apply(sm, 0.0, should_stop=False)
    out = controller.apply(sm, 1.0, should_stop=False)

    assert out == 1.0

  def test_close_lead_caps_positive_crawl_accel(self):
    controller = _make()
    lead = FakeLead(d_rel=4.0, v_rel=0.0)

    out = controller.apply(FakeSM(v_ego=1.0, lead=lead), 1.0, should_stop=False)

    assert 0.0 <= out < 0.15

  def test_close_lead_does_not_bypass_positive_crawl_cap(self):
    controller = _make()
    lead = FakeLead(d_rel=2.0, v_rel=-0.1)

    out = controller.apply(FakeSM(v_ego=0.8, lead=lead), 0.8, should_stop=False)

    assert out <= 0.0

  def test_hard_brake_request_bypasses_comfort_layer(self):
    controller = _make()
    lead = FakeLead(d_rel=2.0, v_rel=-2.0)

    out = controller.apply(FakeSM(v_ego=2.0, lead=lead), -1.2, should_stop=False)

    assert out == -1.2

  def test_low_speed_positive_rise_is_limited(self):
    controller = _make()
    sm = FakeSM(v_ego=2.0, lead=FakeLead(d_rel=12.0))

    controller.apply(sm, 0.0, should_stop=False)
    out = controller.apply(sm, 1.0, should_stop=False)

    assert 0.0 < out < 0.1

  def test_inactive_longitudinal_resets_and_passes_through(self):
    controller = _make()

    out = controller.apply(FakeSM(v_ego=0.0, standstill=True, enabled=False), 0.7, should_stop=False)

    assert out == 0.7


class FakeStopAndGo:
  def __init__(self):
    self.calls = []

  def apply(self, sm, a_target, should_stop):
    self.calls.append((sm, a_target, should_stop))
    return 0.2


def test_planner_output_property_applies_stop_and_go_hook():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  fake_stop_and_go = FakeStopAndGo()
  fake_sm = FakeSM(v_ego=0.0)

  planner.stop_and_go = fake_stop_and_go
  planner._stop_and_go_sm = fake_sm
  planner._output_a_target = 0.0
  planner.output_should_stop = True

  planner.output_a_target = 1.0

  assert fake_stop_and_go.calls == [(fake_sm, 1.0, True)]
  assert planner.output_a_target == 0.2
