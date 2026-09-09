import Std

namespace DRCPT

theorem ratSquareNonnegative (x : Rat) : 0 ≤ x * x := by
  by_cases hx : 0 ≤ x
  · exact Rat.mul_nonneg hx hx
  · have hnx : 0 ≤ -x := by grind
    have hmul := Rat.mul_nonneg hnx hnx
    grind

/-- The two-path reference-point difference in Proposition 4.1 is algebraically correct. -/
theorem referencePathDifference
    (r0 high low terminal etaGain etaLoss : Rat) :
    ((1 - etaLoss) * ((1 - etaGain) * r0 + etaGain * high) + etaLoss * terminal)
      - ((1 - etaGain) * ((1 - etaLoss) * r0 + etaLoss * low) + etaGain * terminal)
    = etaGain * (1 - etaLoss) * high
      - etaLoss * (1 - etaGain) * low
      + (etaLoss - etaGain) * terminal := by
  grind

/-- Each reference update is a convex-combination formula. -/
theorem gainReferenceUpdate
    (reference wealth etaGain : Rat) :
    reference + etaGain * (wealth - reference)
      = (1 - etaGain) * reference + etaGain * wealth := by
  grind

theorem lossReferenceUpdate
    (reference wealth etaLoss : Rat) :
    reference - etaLoss * (reference - wealth)
      = (1 - etaLoss) * reference + etaLoss * wealth := by
  grind

/-- The error-bound assumption legitimately implies its squared form. -/
theorem squaredStationaryDistanceBound
    (distance gradientNorm kappa : Rat)
    (hDistance : 0 ≤ distance)
    (hGradient : 0 ≤ gradientNorm)
    (hKappa : 0 ≤ kappa)
    (hBound : distance ≤ kappa * gradientNorm) :
    distance * distance ≤ (kappa * kappa) * (gradientNorm * gradientNorm) := by
  have hProduct : 0 ≤ kappa * gradientNorm := Rat.mul_nonneg hKappa hGradient
  have hFirst := Rat.mul_le_mul_of_nonneg_right hBound hDistance
  have hSecond := Rat.mul_le_mul_of_nonneg_left hBound hProduct
  have hCombined := Rat.le_trans hFirst hSecond
  grind

def projectUnitInterval (x : Rat) : Rat := min 1 (max 0 x)

/-- Counterexample to the projected-ascent inequality in Lemma 4.6.

At the upper boundary, take J(x)=x, theta=1, d=1, gamma=1/2 and C_P=0.
Projection leaves theta unchanged, while the claimed right side increases by 1/2.
-/
theorem projectedAscentClaimCounterexample :
    projectUnitInterval (1 + (1 / 2 : Rat)) = 1 ∧
      ¬(projectUnitInterval (1 + (1 / 2 : Rat)) ≥ 1 + (1 / 2 : Rat)) := by
  native_decide

/-- A zero smoothed gradient does not force the constituent gradients to be small. -/
theorem smoothedGradientCancellationCounterexample :
    (((1 : Rat) + (-1 : Rat)) / 2) ^ 2 = 0 ∧
      (1 : Rat) ^ 2 + (-1 : Rat) ^ 2 = 2 := by
  native_decide

end DRCPT
