# Configuring an optimizer

## Callbacks and ownership

An objective callback receives the current coordinates and a writable gradient. Both slices are valid for that call. Return the objective value each time, and fill the gradient whenever it is nonempty. The optimizer retains the delegate and its captured context between solves.

If a callback throws, the optimizer stops and rethrows the original D throwable to the caller. To stop deliberately and receive a partial result, call `forceStop()` from the callback and check `result.termination` after the solve.

A local optimizer releases its native handle at scope exit. If the owner is destroyed during an evaluation, the active solve retains the handle until it finishes handling the result or callback exception. For a GC-managed class with an `Optimizer` field, prefer explicit destruction when the resource is no longer needed. If relying on GC finalization, declare a class destructor (an empty `~this() {}` suffices). With DMD 2.114.0, the field destructor alone did not mark the enclosing class allocation for finalization.

## Maximization

Use `setMaxObjective` with the value and derivatives of the function you want to maximize. This quadratic peaks at `x = 3`, where the returned objective value is 7.

```d
import nloptd;

auto opt = Optimizer(Algorithm.ldSLSQP, 1);
opt.setMaxObjective((x, gradient) {
    const delta = x[0] - 3;
    if (gradient.length) gradient[0] = -2 * delta;
    return 7 - delta * delta;
});
opt.setXTolRel(1e-9);
opt.setMaxEvaluations(100);
double[1] point = [0];
auto result = opt.optimizeInPlace(point[]);
```

## Stopping criteria

Choose criteria in the units of the problem. With `setStopValue(target)`, minimization seeks an objective value below the target and maximization seeks one above it. Whether exact equality triggers `Termination.stopValueReached` depends on the algorithm. A finite target stays in effect when you switch objective direction.

`setFTolAbs` and `setFTolRel` set tolerances on changes in the objective. `setXTolAbs` accepts either a uniform absolute tolerance or one per coordinate. For example, `opt.setXTolAbs([1e-4, 1e-8])` gives a two-dimensional problem different absolute tolerances for its two coordinates. Use `setXTolRel` for a relative coordinate tolerance.

Use `setMaxEvaluations` to limit objective evaluations or `setMaxTime` for an approximate time limit in seconds. An evaluation in progress can run past the time limit. Check `result.termination` to see which criterion ended the run.

## Scalar nonlinear constraints

Use `addInequalityConstraint` for `c(x) ≤ 0` and `addEqualityConstraint` for `h(x) = 0`. Each call registers another constraint. The optional second argument is the absolute tolerance NLopt uses to judge feasibility when stopping. It defaults to zero. Select an algorithm that supports the constraints you need, such as SLSQP for equalities or MMA for inequalities.

For example, this SLSQP constraint requires the point to lie on the unit circle in two dimensions.

```d
import nloptd;

auto opt = Optimizer(Algorithm.ldSLSQP, 2);
opt.addEqualityConstraint((x, gradient) {
    if (gradient.length)
    {
        gradient[0] = 2 * x[0];
        gradient[1] = 2 * x[1];
    }
    return x[0] * x[0] + x[1] * x[1] - 1;
}, 1e-8);
```

Constraints follow the objective's slice lifetime and exception rules. Each registration retains its own callback across solves. The [consumer test](../tests/consumer/source/app.d) includes the NLopt MMA tutorial with two separately captured cubic inequalities.

## Vector-valued nonlinear constraints

Use `addEqualityMConstraint` or `addInequalityMConstraint` to evaluate several residuals at the same point. Supply one nonnegative absolute feasibility tolerance per residual. NLopt copies these tolerances at registration. An empty tolerance slice registers no constraints.

For MMA and CCSAQ, NLopt 2.11.0 calculates its workspace size using unsigned integer arithmetic. The wrapper checks the dimension at construction and the total number of residuals across scalar and vector registrations before adding a constraint.

The callback borrows all three slices for one invocation. Write every residual. When NLopt requests derivatives, write one row per residual in the flat gradient slice. For `n` coordinates, `gradient[i * n + j]` is the derivative of residual `i` with respect to coordinate `j`.

```d
import nloptd;

auto opt = Optimizer(Algorithm.ldSLSQP, 3);
opt.addEqualityMConstraint((residuals, x, gradient) {
    residuals[0] = x[0] + 2 * x[1] - 3;
    residuals[1] = -x[1] + 3 * x[2] - 8;
    if (gradient.length)
        gradient[] = [1.0, 2, 0, 0, -1, 3];
}, [1e-9, 1e-8]);
```

### Changing constraints between solves

To reuse an optimizer with a different constraint set, call `removeEqualityConstraints()` or `removeInequalityConstraints()` between solves. Each method clears both scalar and vector callbacks of that kind and releases the wrapper's references to them. You can then add fresh constraints. Repeated removal is allowed.

## Initial steps for derivative-free optimization

For Nelder–Mead, use `opt.setInitialStep(0.5)` to set the same initial step on every coordinate. When coordinates have different scales, use one signed, nonzero step per coordinate, for example `opt.setInitialStep([0.1, 20.0])`. NLopt copies the steps and keeps them across solves. Call `opt.resetInitialStep()` to let NLopt choose steps from the starting point and bounds again.

## Population size and vector storage

For a population-based algorithm such as CRS2, `opt.setPopulation(40)` requests a population of 40 candidate points. For a limited-memory quasi-Newton algorithm such as L-BFGS, `opt.setVectorStorage(5)` requests storage for five previous updates. A larger history uses more memory. Pass zero to either setter to restore NLopt's default for that setting.

CRS2 needs at least one more candidate than the problem dimension. Both setters check explicit settings against the algorithm's native integer and workspace limits before replacing the current value. Available memory can still limit the solve.

## Subsidiary optimizers

Use `setLocalOptimizer` to select the local algorithm and its stopping criteria. Configure the objective, bounds, and constraints on the parent. The local optimizer must have the same dimension. NLopt copies its configuration at registration, so you can reuse it or let it leave scope before the parent runs.

```d
import nloptd;

auto opt = Optimizer(Algorithm.auglag, 1);
{
    auto local = Optimizer(Algorithm.ldSLSQP, 1);
    local.setXTolRel(1e-10);
    opt.setLocalOptimizer(local);
}
```

Calling the setter again replaces the local solver. A dimension mismatch preserves the existing configuration. If NLopt reports an out-of-memory error while copying a replacement, install a local solver again before retrying the solve.

All AUGLAG variants used through `Optimizer`, including legacy algorithm IDs, require an explicit local solver before a solve or before installation as another optimizer's subsidiary. You can register constraints before choosing the local solver. Callers using NLopt's legacy process-wide defaults can still access them through `nloptd.raw`.

MMA and CCSAQ create their dual solver from the selected algorithm and stopping criteria. This discards the local solver AUGLAG needs, so `setLocalOptimizer` rejects all AUGLAG variants in that position and preserves the previously configured solver.

The wrapper checks AUGLAG's workspace size using the total number of residuals it penalizes and the width of the largest constraint callback. AUGLAG_EQ forwards inequalities to its local solver, whose limits also apply.
