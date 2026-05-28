"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections import deque
from dataclasses import dataclass


_HOLD_FRAMES = 12
_STATUS_WINDOW = 20
_STABILITY_FLIPS_FULL = 6.0

# Phantom-lead mask: when a newly-acquired close-range lead has low modelProb,
# it's almost always radar ghost (curb return, road edge, ground bounce).
# Override status=False in the proxy so MPC ignores it. process_lead in
# longitudinal_mpc_lib uses dRel/vLead/aLeadK/aLeadTau directly and does NOT
# weight by modelProb — masking status is the only way to suppress this class.
_NEW_LEAD_TRUST_FRAMES = 6
_NEW_LEAD_TRUST_DREL = 20.0
_PHANTOM_MODELPROB_MAX = 0.5
_PHANTOM_DREL_MAX = 5.0


@dataclass
class _LeadSnap:
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  vLead: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 0.0
  modelProb: float = 0.0
  aRel: float = 0.0
  fcw: bool = False


class _LeadProxy:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau',
               'modelProb', 'aRel', 'fcw')

  def __init__(self, snap: _LeadSnap):
    self.status = True
    self.dRel = snap.dRel
    self.yRel = snap.yRel
    self.vRel = snap.vRel
    self.vLead = snap.vLead
    self.aLeadK = snap.aLeadK
    self.aLeadTau = snap.aLeadTau
    self.modelProb = snap.modelProb
    self.aRel = snap.aRel
    self.fcw = snap.fcw


class _LeadProxyMasked:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau',
               'modelProb', 'aRel', 'fcw')

  def __init__(self):
    self.status = False
    self.dRel = 0.0
    self.yRel = 0.0
    self.vRel = 0.0
    self.vLead = 0.0
    self.aLeadK = 0.0
    self.aLeadTau = 0.0
    self.modelProb = 0.0
    self.aRel = 0.0
    self.fcw = False


class _RadarStateProxy:
  __slots__ = ('_raw', '_lead_one', '_lead_two')

  def __init__(self, raw, lead_one, lead_two):
    self._raw = raw
    self._lead_one = lead_one
    self._lead_two = lead_two

  @property
  def leadOne(self):
    return self._lead_one if self._lead_one is not None else self._raw.leadOne

  @property
  def leadTwo(self):
    return self._lead_two if self._lead_two is not None else self._raw.leadTwo

  def __getattr__(self, name):
    return getattr(self._raw, name)


class LeadPersistence:
  """Internal helper. Hold last-known leadOne/leadTwo alive for HOLD_FRAMES
  after a status drop, so the MPC view of radarState ignores brief flicker.
  Also masks freshly-acquired close-range low-confidence phantom leads so the
  planner doesn't demand emergency brake on radar ghosts.
  No own param — owner (RadarDistanceController) gates via force_enabled."""

  def __init__(self):
    self._last_one: _LeadSnap | None = None
    self._last_two: _LeadSnap | None = None
    self._alive_one = 0
    self._alive_two = 0

    self._status_hist: deque[bool] = deque(maxlen=_STATUS_WINDOW)
    self._stability = 1.0

    self._new_one_age = _NEW_LEAD_TRUST_FRAMES
    self._new_two_age = _NEW_LEAD_TRUST_FRAMES
    self._prev_one_status = False
    self._prev_two_status = False

  @property
  def stability(self) -> float:
    return self._stability

  def reset(self) -> None:
    self._last_one = None
    self._last_two = None
    self._alive_one = 0
    self._alive_two = 0
    self._status_hist.clear()
    self._stability = 1.0
    self._new_one_age = _NEW_LEAD_TRUST_FRAMES
    self._new_two_age = _NEW_LEAD_TRUST_FRAMES
    self._prev_one_status = False
    self._prev_two_status = False

  def update(self, radarstate, force_enabled: bool = True) -> None:
    if radarstate is None:
      return
    if not force_enabled:
      self.reset()
      return

    one = radarstate.leadOne
    two = radarstate.leadTwo

    # Treat phantom leads as if status=False: don't snap them, don't extend
    # alive counter. Otherwise the hold-alive path re-injects the ghost
    # for up to _HOLD_FRAMES (~600ms) after the raw status drops.
    one_valid = bool(one.status) and not self._is_phantom(one)
    two_valid = bool(two.status) and not self._is_phantom(two)

    if one_valid:
      if not self._prev_one_status and float(one.dRel) < _NEW_LEAD_TRUST_DREL:
        self._new_one_age = 0
      else:
        self._new_one_age = min(_NEW_LEAD_TRUST_FRAMES, self._new_one_age + 1)
      self._last_one = self._snap(one)
      self._alive_one = _HOLD_FRAMES
    elif self._alive_one > 0:
      self._alive_one -= 1
      self._new_one_age = _NEW_LEAD_TRUST_FRAMES
    else:
      self._new_one_age = _NEW_LEAD_TRUST_FRAMES

    if two_valid:
      if not self._prev_two_status and float(two.dRel) < _NEW_LEAD_TRUST_DREL:
        self._new_two_age = 0
      else:
        self._new_two_age = min(_NEW_LEAD_TRUST_FRAMES, self._new_two_age + 1)
      self._last_two = self._snap(two)
      self._alive_two = _HOLD_FRAMES
    elif self._alive_two > 0:
      self._alive_two -= 1
      self._new_two_age = _NEW_LEAD_TRUST_FRAMES
    else:
      self._new_two_age = _NEW_LEAD_TRUST_FRAMES

    self._prev_one_status = one_valid
    self._prev_two_status = two_valid

    self._status_hist.append(bool(one.status))
    if len(self._status_hist) >= 5:
      flips = sum(1 for i in range(1, len(self._status_hist))
                  if self._status_hist[i] != self._status_hist[i - 1])
      self._stability = max(0.0, 1.0 - min(1.0, flips / _STABILITY_FLIPS_FULL))
    else:
      self._stability = 1.0

  @staticmethod
  def _is_phantom(lead) -> bool:
    # Close-range lead with low modelProb = radar ghost (curb, road edge, etc).
    # Persistent low-confidence reads count: if model never agrees, don't trust
    # radar alone for emergency brake at near-zero range.
    return (bool(lead.status)
            and float(lead.modelProb) < _PHANTOM_MODELPROB_MAX
            and float(lead.dRel) < _PHANTOM_DREL_MAX)

  def smooth(self, radarstate, force_enabled: bool = True):
    if not force_enabled or radarstate is None:
      return radarstate

    l1 = None
    l2 = None

    if self._is_phantom(radarstate.leadOne):
      l1 = _LeadProxyMasked()
    elif not radarstate.leadOne.status and self._alive_one > 0 and self._last_one is not None:
      l1 = _LeadProxy(self._last_one)

    if self._is_phantom(radarstate.leadTwo):
      l2 = _LeadProxyMasked()
    elif not radarstate.leadTwo.status and self._alive_two > 0 and self._last_two is not None:
      l2 = _LeadProxy(self._last_two)

    if l1 is None and l2 is None:
      return radarstate

    return _RadarStateProxy(radarstate, l1, l2)

  @staticmethod
  def _snap(lead) -> _LeadSnap:
    return _LeadSnap(
      dRel=float(getattr(lead, 'dRel', 0.0)),
      yRel=float(getattr(lead, 'yRel', 0.0)),
      vRel=float(getattr(lead, 'vRel', 0.0)),
      vLead=float(getattr(lead, 'vLead', 0.0)),
      aLeadK=float(getattr(lead, 'aLeadK', 0.0)),
      aLeadTau=float(getattr(lead, 'aLeadTau', 0.0)),
      modelProb=float(getattr(lead, 'modelProb', 0.0)),
      aRel=float(getattr(lead, 'aRel', 0.0)),
      fcw=bool(getattr(lead, 'fcw', False)),
    )
