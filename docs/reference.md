# NLopt D reference

Use `import nloptd;` for the D interface. An `Optimizer` owns the native optimizer, accepts D functions and delegates, and runs directly on writable slices of doubles. The D module needs the NLopt header at build time and the matching native library at link time. See [Build and run](../README.md#build-and-run) for platform-specific setup and the [upstream C reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/) for the underlying algorithms and settings.

- [The optimizer](#the-optimizer) and [objective function](#objective-function)
- [Bound constraints](#bound-constraints), [nonlinear constraints](#nonlinear-constraints), and [vector-valued constraints](#vector-valued-constraints)
- [Stopping criteria](#stopping-criteria), [forced termination](#forced-termination), and [named algorithm parameters](#named-algorithm-parameters)
- [Performing the optimization](#performing-the-optimization), [return values](#return-values), and [exceptions](#exceptions)
- [Subsidiary optimizers](#subsidiary-optimizers), [initial step size](#initial-step-size), [population size](#population-size), [random seeds](#random-seeds), and [vector storage](#vector-storage)
- [Library version](#library-version)

## The optimizer

Choose an algorithm and a positive dimension (the number of variables) when constructing an optimizer. Both are fixed for its lifetime. The D enum names five algorithms: `Algorithm.lnNelderMead`, `Algorithm.ldLBFGS`, `Algorithm.ldMMA`, `Algorithm.ldSLSQP`, and `Algorithm.auglag`. The same constructor accepts a native algorithm identifier from the installed header. For example, these declarations each create a two-variable derivative-free optimizer:

```d
import nloptd;
import c = nloptd.raw;

auto simplex = Optimizer(Algorithm.lnNelderMead, 2);
auto cobyla = Optimizer(c.NLOPT_LN_COBYLA, 2);
assert(cobyla.dimension == 2);
assert(cobyla.algorithm == c.NLOPT_LN_COBYLA);
```

Read `opt.dimension` for the number of variables, `opt.algorithm` for the native identifier, and `opt.algorithmName` for NLopt's human-readable description. All three properties are available on a const optimizer.

Choose an algorithm that supports the derivatives and constraints of your problem. The upstream algorithm guide lists those capabilities for each solver. [NLopt algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)

`Optimizer` is a move-only struct. It releases its native handle at scope exit. Pass it by `ref` to helpers that configure it. You can reuse an optimizer between runs, including replacing the objective or changing constraints. Confine an active optimizer and its callbacks to one thread. [Ownership and callback lifetime](optimizer.md#callbacks-and-ownership)

## Objective function

Register the function to minimize with `opt.setMinObjective(f)`, or the function to maximize with `opt.setMaxObjective(f)`. Either call replaces the previous objective. Both accept a function pointer or a delegate, so an objective can capture problem data. The optimizer retains the registered callable. The delegate type is:

```d
alias Objective = double delegate(scope const(double)[] x, scope double[] gradient);
```

Each time NLopt evaluates the objective, `x` contains the current trial point. Return the function value there. For maximization, supply the value and derivatives with their original signs.

### Writing derivatives in place

When NLopt requests derivatives, `gradient` has the same length as `x`. Write the partial derivative with respect to `x[j]` into `gradient[j]`. A derivative-free algorithm supplies an empty gradient. Even a gradient-based algorithm can request the value alone, so check `gradient.length` on each call.

The slices borrow NLopt's storage for this invocation. Assign to elements or to `gradient[]` to write through the slice. Reassigning the local variable `gradient` only changes which storage it refers to. Copy any values you need after the callback returns. [Callback conventions](callbacks.md) · [Maximization example](optimizer.md#maximization)

## Bound constraints

Use `setLowerBounds` and `setUpperBounds` to specify the interval for each variable. A scalar applies to every variable. A slice supplies one bound per variable in input order. For a two-variable problem:

```d
opt.setLowerBounds(0.0);
opt.setUpperBounds([2.0, double.infinity]);
```

Here both variables must be nonnegative and only the first has a finite upper bound. Bounds default to negative and positive infinity. The initial point must satisfy every bound, and NLopt keeps subsequent evaluations within them. Equal lower and upper bounds fix a variable at its starting value.

## Nonlinear constraints

A scalar constraint uses the same callback signature as the objective. Return a residual `c(x)`, and fill in its gradient when requested. Register an inequality for `c(x) ≤ 0` or an equality for `c(x) = 0`. For example, to impose `x[0] + x[1] ≤ 3`:

```d
opt.addInequalityConstraint((x, gradient) {
    if (gradient.length) gradient[] = 1.0;
    return x[0] + x[1] - 3;
}, 1e-8);
```

`addEqualityConstraint` takes the same arguments. Each registration adds a constraint. The optional second argument is a nonnegative absolute tolerance in units of the residual, defaulting to zero. For stopping purposes, an inequality is feasible when `c(x) <= tolerance` and an equality when `abs(c(x)) <= tolerance`. Trial points can violate nonlinear constraints, so the objective must still be evaluable there. [Scalar constraint examples](optimizer.md#scalar-nonlinear-constraints)

### Vector-valued constraints

When several residuals share a calculation, evaluate them in one call with `addInequalityMConstraint` or `addEqualityMConstraint`. Pass a delegate (`VectorConstraint`) or function pointer (`VectorConstraintFunction`) followed by a slice of nonnegative tolerances. Its length determines the number of residuals, `m`. For example, group two inequalities in two variables:

```d
opt.addInequalityMConstraint((residuals, x, gradient) {
    residuals[0] = x[0] + x[1] - 3;
    residuals[1] = x[0] - x[1];
    if (gradient.length)
        gradient[] = [1.0, 1.0, 1.0, -1.0];
}, [1e-8, 1e-8]);
```

Write all residuals on every call. The writable derivative buffer, when present, contains one row per residual. For a problem of dimension `n`, write the derivative of residual `i` with respect to variable `j` at `gradient[i * n + j]`. The callback returns `void`. Its slices have the same lifetime as those of the objective.

Between runs, `removeInequalityConstraints()` or `removeEqualityConstraints()` clears all scalar and vector registrations of that kind. [Changing constraints between solves](optimizer.md#changing-constraints-between-solves)

## Stopping criteria

Enable the tests relevant to the problem. NLopt can return as soon as one enabled criterion is met. A tolerance measures progress between estimates and does not guarantee a given distance from the true optimum.

| Method | When the run can stop |
| --- | --- |
| `setStopValue(double value)` | The objective falls below the target for minimization or rises above it for maximization. Whether equality suffices depends on the algorithm. |
| `setFTolRel(double tolerance)` | The fractional change in objective value is sufficiently small. |
| `setFTolAbs(double tolerance)` | The change in objective value is sufficiently small in objective units. Useful when the optimum value is near zero. |
| `setXTolRel(double tolerance)` | The change in the point is small relative to its size. |
| `setXTolAbs(tolerance)` | Changes in the individual variables are small in their own units. Supply a scalar for all variables or a slice with one tolerance per variable. |
| `setMaxEvaluations(int limit)` | The objective evaluation budget is reached. Some algorithms can exceed it slightly. |
| `setMaxTime(double seconds)` | The time budget is reached. An evaluation already in progress can run beyond it. |

A nonpositive evaluation limit or time budget disables that test. For example, combine `opt.setXTolRel(1e-8)` with `opt.setMaxEvaluations(1000)` to bound the work if progress is slow. The result identifies which test ended the run. [Stopping criteria examples](optimizer.md#stopping-criteria)

### Variable weights

Algorithms that use NLopt's weighted relative-change test give each variable a default weight of one. For variables with typical magnitudes of 1 and 100, `opt.setXWeights([1.0, 0.01])` gives changes of those respective magnitudes equal weight. Supply one finite, nonnegative weight per variable. A zero weight omits that variable from the norm. Use `opt.setXWeights(1.0)` to restore unit weights. Some algorithms interpret `setXTolRel` differently. In NLopt 2.11.0, COBYLA and BOBYQA use it to set a final trust-region radius and ignore variable weights.

In NLopt 2.11.0, algorithms such as Nelder–Mead remove variables whose lower and upper bounds are equal. The weights stay in their original order, so fixing the first of two variables makes the solver use the first weight for the remaining variable. When you need nonuniform weights with one of these algorithms, construct the optimizer with only the free variables, supply their weights in the same order, and use the fixed values as constants in your objective and constraints. [NLopt 2.11.0 dimension reduction](https://github.com/stevengj/nlopt/blob/v2.11.0/src/api/optimize.c#L350-L354)

### Forced termination

An objective or constraint callback can call `opt.forceStop()` on the active optimizer's thread. Finish the callback normally. The solve returns with `Termination.forcedStop`, and the point may represent partial progress. Calling `forceStop` outside an active solve throws an exception. To report a failure originating inside the callback, you can also throw a D exception. [Exceptions](#exceptions)

## Named algorithm parameters

Use `setParam` to set or replace an algorithm-specific numeric setting between solves. For example, MMA accepts `"inner_maxeval"` to limit its inner iterations:

```d
auto opt = Optimizer(Algorithm.ldMMA, 2);
opt.setParam("inner_maxeval", 20);
assert(opt.hasParam("inner_maxeval"));
assert(opt.getParam("inner_maxeval", 0) == 20);
```

`getParam(name, fallback)` returns your fallback when the name is absent. `hasParam(name)` reports whether a value was explicitly set. NLopt can store names the selected algorithm ignores. Names are case-sensitive and may contain up to 1023 bytes with no embedded NUL. These methods are available when the installed header provides the named-parameter API.

For MMA and CCSAQ, select the dual algorithm with `setLocalOptimizer`. The wrapper rejects `setParam("dual_algorithm", ...)` on those optimizers because a native named override would take precedence over the validated subsidiary solver regardless of setter order. Other dual settings, such as `"dual_ftol_rel"`, can be set by name. [Subsidiary optimizers](#subsidiary-optimizers) · [Upstream algorithm-specific parameters](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters)

## Performing the optimization

Pass a writable `double[]` of the configured dimension to `optimizeInPlace`. Its contents are the initial guess on entry and the final point on return. To keep the initial guess, pass a copy made with `.dup`. The following complete program minimizes the squared distance to 3 with an upper bound of 2. Its solution is 2, where the objective value is 1.

```d
import nloptd;
import std.stdio : writeln;

void main()
{
    double target = 3;
    auto opt = Optimizer(Algorithm.ldSLSQP, 1);
    opt.setMinObjective((x, gradient) {
        const delta = x[0] - target;
        if (gradient.length) gradient[0] = 2 * delta;
        return delta * delta;
    });
    opt.setUpperBounds(2.0);
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(100);

    double[1] point = [0.0];
    auto result = opt.optimizeInPlace(point[]);
    writeln(point[], " ", result.value, " ", result.termination);
}
```

For another run, set the desired initial values in the point before calling `optimizeInPlace` again. A contiguous Mir ndslice backed by a writable `double[]` can also be passed directly on this branch. With nloptd 0.3.0, pass its `.field`. [Mir interoperability](mir.md)

### Return values

An `OptimizationResult` describes the individual run. Read `value` (`double`) for the objective at the returned point, `evaluations` (`int`) for the objective evaluation count, and `termination` (`Termination`) to interpret the result.

| Termination | Meaning |
| --- | --- |
| `success` | The algorithm reported success without a more specific reason. |
| `stopValueReached` | The objective reached the requested target. |
| `fToleranceReached` | The relative or absolute objective tolerance was met. |
| `xToleranceReached` | The relative or absolute tolerance on the variables was met. |
| `maxEvaluationsReached`, `maxTimeReached` | The run used its evaluation or time budget. Inspect the available point and value before deciding whether to continue. |
| `roundoffLimited`, `forcedStop` | Numerical roundoff limited progress, or a callback explicitly stopped the run. The result can be partial. |

### Exceptions

Wrapper validation throws a D `Exception`, for example if the point has the wrong length or the optimizer has no objective. A native failure reported as `NloptException` carries the original `nlopt_result` in its `.code` field. Roundoff and an explicit forced stop are handled as return values above.

If a callback throws, nloptd catches the throwable at the C boundary, stops NLopt, and rethrows the original object to the caller of `optimizeInPlace`. Code that relies on a completed solve should consume the result only after the call returns normally.

## Subsidiary optimizers

For algorithms that use a local solver, configure a second optimizer of the same dimension. Set its local algorithm and stopping criteria, then install it on the parent:

```d
auto parent = Optimizer(Algorithm.auglag, 2);
auto local = Optimizer(Algorithm.ldSLSQP, 2);
local.setXTolRel(1e-10);
parent.setLocalOptimizer(local);
```

Set the objective, bounds, and constraints on the parent. NLopt copies the subsidiary settings at registration, so the original local optimizer can leave scope or be reconfigured. Call the setter again to replace the installed configuration. All AUGLAG variants used through the D wrapper require an explicit local solver. [Subsidiary optimizer details](optimizer.md#subsidiary-optimizers)

## Initial step size

Derivative-free local algorithms such as Nelder–Mead use initial steps to explore around the starting point. With variables on different scales, use `opt.setInitialStep([0.1, 20.0])` for a two-variable problem. A scalar applies the same step to each variable. Each step is signed and must be nonzero. NLopt copies the values. Call `opt.resetInitialStep()` to resume automatic selection from the starting point and bounds. [Initial step example](optimizer.md#initial-steps-for-derivative-free-optimization)

## Population size

For population-based algorithms, `opt.setPopulation(40)` requests 40 candidate points. The size is algorithm-dependent. For example, CRS2 requires at least one more candidate than the dimension. Pass zero to restore the native default. [Population size and storage examples](optimizer.md#population-size-and-vector-storage)

## Random seeds

To reproduce a stochastic run, import `nloptd.raw` as `c` and call `c.nlopt_srand(1234)` on the solving thread before running the optimizer. Reseed before a second run to restart the random sequence. This setting applies to the native generator on that thread and takes no optimizer handle. Use `c.nlopt_srand_time()` for a time-based seed.

## Vector storage

Limited-memory algorithms such as L-BFGS keep a history of previous updates. Use `opt.setVectorStorage(5)` to retain five updates. A larger history uses more memory and can change convergence. Zero restores NLopt's choice. [Vector storage example](optimizer.md#population-size-and-vector-storage)

## Library version

Query the linked native library with `nlopt_version`. The three output arguments receive the major, minor, and patch numbers:

```d
import c = nloptd.raw;

int major, minor, patch;
c.nlopt_version(&major, &minor, &patch);
```
