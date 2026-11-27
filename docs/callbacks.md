# Objective and constraint callbacks in NLopt and D

The NLopt manual documents the C signatures and derivative conventions for [objective functions](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#objective-function) and [nonlinear constraints](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#nonlinear-constraints).

## The NLopt C interface

The NLopt C API represents a point as an array of `n` consecutive `double` values, where `n` is the optimizer dimension. The caller passes a pointer to the first element. The array holds the initial guess on entry and the final point on return.

To set the objective, the caller registers a function pointer with `nlopt_set_min_objective` or `nlopt_set_max_objective`. During a solve, NLopt calls that function for each evaluation, supplying the dimension and a read-only pointer to the trial point. The function returns the objective value. When NLopt also requests derivatives, it supplies a writable gradient array of the same length for the function to fill.

A scalar nonlinear constraint uses the same function signature. The return value is the constraint residual: an inequality requires it to be nonpositive, and an equality requires it to be zero. When requested, the function writes the gradient of that residual.

## How nloptd exposes the C interface in D

In D, pass your objective function or delegate to `setMinObjective` or `setMaxObjective`. Register each scalar constraint with `addInequalityConstraint` or `addEqualityConstraint`. In both cases, nloptd registers a C callback that invokes your D code with two built-in slices over NLopt’s arrays:

- The trial point is a `const(double)[]` of length `n` (read-only so the callback cannot alter the point while evaluating it).
- The gradient is a writable `double[]` of length `n` when derivatives are requested. A null gradient pointer becomes an empty slice.

A vector-valued constraint writes several residuals into an output slice. When derivatives are requested, its gradient slice holds one row per residual. All slices are valid for the duration of the call. See the [vector constraint example and Jacobian layout](optimizer.md#vector-valued-nonlinear-constraints).
