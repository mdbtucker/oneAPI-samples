#ifndef __STATIC_MATH_HPP__
#define __STATIC_MATH_HPP__

// helper functions for calculating values at compile time
// input values must be compile time constants

// computes 2^n
template <typename T>
static constexpr T Pow2(T n) {
  return T(1) << n;
}
// base-2 logarithm
template <typename T>
static constexpr T Log2(T n) {
  return ((n < 2) ? T(0) : T(1) + Log2(n / 2));
}
// round up Log2
template <typename T>
static constexpr T CeilLog2(T n) {
  return ((n == 1) ? T(0) : Log2(n - 1) + T(1));
}


#endif  // __STATIC_MATH_HPP__
