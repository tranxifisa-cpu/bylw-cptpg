import Std

namespace OptimizedAudit

-- Coefficient algebra only; this is not a formal differentiation theorem.
theorem cubicMultiplierSecondDerivativeCoefficients
    (c2 c3 p indicator : Rat) :
    2 * (3 * c3 * indicator - 2 * c2) - 18 * c3 * p =
      6 * c3 * (indicator - p) - 2 * (2 * c2 + 6 * c3 * p) := by
  grind

-- The antithetic difference vanishes for every affine multiplier.
theorem affineAntitheticCancellation (a b x y : Rat) :
    a * ((x + y) / 2) + b - ((a * x + b) + (a * y + b)) / 2 = 0 := by
  grind

theorem quadraticAntitheticDifference (a b c x y : Rat) :
    (a * ((x + y) / 2) ^ 2 + b * ((x + y) / 2) + c)
      - ((a * x ^ 2 + b * x + c) + (a * y ^ 2 + b * y + c)) / 2
      = -a * (x - y) ^ 2 / 4 := by
  grind

-- A small windowed vector can coexist with nonzero instantaneous residuals.
theorem cancellationDoesNotCertifyStationarity :
    (((1 : Rat) + (-1 : Rat)) / 2) ^ 2 = 0 ∧
      ((1 : Rat) ^ 2 + (-1 : Rat) ^ 2) / 2 = 1 := by
  native_decide

end OptimizedAudit
