"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.selfdrive.controls.lib.dynamic_personality.dynamic_follow import (
  FollowDistanceController,
  ALEAD_RATE_DELTA,
  ALEAD_RATE_DEADBAND,
)


class FakeLead:
  def __init__(self, d_rel=30.0, v_lead=20.0, a_lead=0.0, a_tau=0.0, model_prob=1.0):
    self.status = True
    self.dRel = d_rel
    self.vLead = v_lead
    self.vRel = 0.0
    self.aLeadK = a_lead
    self.aLeadTau = a_tau
    self.modelProb = model_prob


class FakeRadarState:
  def __init__(self, lead=None):
    self.leadOne = lead if lead is not None else FakeLead()


def _make():
  df = FollowDistanceController()
  df.set_enabled(True)
  return df


class TestAleadRateAnticipator:
  def test_zero_when_no_change(self):
    df = _make()
    rs = FakeRadarState(FakeLead(a_lead=-0.5))
    for _ in range(20):
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
    assert df._dbg.alead_rate == 0.0

  def test_fires_on_rapid_brake_escalation(self):
    df = _make()
    df.update()
    peak = 0.0
    for i in range(15):
      a = max(0.0 - 0.30 * i, -3.0)
      rs = FakeRadarState(FakeLead(a_lead=a))
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
      peak = max(peak, df._dbg.alead_rate)
    assert peak > 0.0
    assert peak <= ALEAD_RATE_DELTA + 1e-6

  def test_saturates_at_delta_max(self):
    df = _make()
    df.update()
    peak = 0.0
    for i in range(15):
      a = max(0.0 - 0.6 * i, -8.0)
      rs = FakeRadarState(FakeLead(a_lead=a))
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
      peak = max(peak, df._dbg.alead_rate)
    assert peak > ALEAD_RATE_DELTA * 0.5

  def test_deadband_ignores_slow_changes(self):
    df = _make()
    df.update()
    for i in range(40):
      a = -0.01 * i
      rs = FakeRadarState(FakeLead(a_lead=a))
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
    assert df._dbg.alead_rate == 0.0

  def test_resets_when_lead_lost(self):
    df = _make()
    df.update()
    for i in range(15):
      rs = FakeRadarState(FakeLead(a_lead=-0.3 * i))
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
    for _ in range(20):
      df.update()
      df.get_follow_distance_multiplier(20.0, None)
    assert df._prev_alead_for_rate is None
    assert df._alead_rate_ema == 0.0

  def test_ema_smooths_single_frame_glitch(self):
    df = _make()
    df.update()
    for _ in range(10):
      rs = FakeRadarState(FakeLead(a_lead=-0.2))
      df.update()
      df.get_follow_distance_multiplier(20.0, rs)
    rs = FakeRadarState(FakeLead(a_lead=-5.0))
    df.update()
    df.get_follow_distance_multiplier(20.0, rs)
    rs = FakeRadarState(FakeLead(a_lead=-0.2))
    df.update()
    df.get_follow_distance_multiplier(20.0, rs)
    assert df._dbg.alead_rate < ALEAD_RATE_DELTA * 0.4

  def test_deadband_value_is_negative(self):
    assert ALEAD_RATE_DEADBAND < 0

  def test_boost_bypasses_rate_limit(self):
    df = _make()
    df.update()
    profile = [-0.5] * 5 + [-4.0] * 20
    tf_history = []
    for a in profile:
      rs = FakeRadarState(FakeLead(a_lead=a))
      df.update()
      tf = df.get_follow_distance_multiplier(15.0, rs)
      tf_history.append(tf)
    tf_just_after_step = tf_history[5]
    tf_baseline = tf_history[4]
    assert tf_just_after_step - tf_baseline > 0.05

  def test_alead_modifier_gated_on_vrel(self):
    df = _make()
    # mild aLK, vRel inside gate deadband → modifier should NOT fire
    for _ in range(10):
      lead = FakeLead(a_lead=-0.96)
      lead.vLead = 19.95  # v_rel = vLead - v_ego = 19.95 - 20 = -0.05
      df.update()
      df.get_follow_distance_multiplier(20.0, FakeRadarState(lead))
    assert df._dbg.alead == 0.0

    # same aLK but real closing → modifier fires
    df2 = _make()
    for _ in range(10):
      lead = FakeLead(a_lead=-0.96)
      lead.vLead = 18.0  # v_rel = -2.0, past gate
      df2.update()
      df2.get_follow_distance_multiplier(20.0, FakeRadarState(lead))
    assert df2._dbg.alead > 0.0

  def test_boost_resets_after_lead_lost(self):
    df = _make()
    df.update()
    for i in range(10):
      a = max(0 - 0.5 * i, -3.0)
      rs = FakeRadarState(FakeLead(a_lead=a))
      df.update()
      df.get_follow_distance_multiplier(15.0, rs)
    assert df._alead_rate_boost > 0.0
    for _ in range(30):
      df.update()
      df.get_follow_distance_multiplier(15.0, None)
    assert df._alead_rate_boost == 0.0
