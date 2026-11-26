# Migrating from nloptd 0.2.1

Keep the `nloptd` DUB dependency and `import nloptd;`. The modern wrapper imports your installed `nlopt.h` and links to the matching native library. It no longer depends on the `libnlopt` D package. For include paths, linking, and platform-specific prerequisites, see the [build instructions](README.md#build-and-run).

## Port a solve

Both programs minimize the squared distance from a target of 3. The minimum value is zero. Each solve writes the final coordinates back into `point`.

### Version 0.2.1

```d
import nloptd;
import std.math : abs;

extern(C) double objective(uint n, const(double)* x, double* gradient, void* data) nothrow @nogc
{
    const delta = x[0] - *cast(double*) data;
    if (gradient) gradient[0] = 2 * delta;
    return delta * delta;
}

void main()
{
    double target = 3;
    auto opt = Opt(Algorithm.ldSLSQP, 1);
    opt.setMinObjective(&objective, target);
    opt.setXTolRel(1e-8);
    opt.setMaxeval(100);
    double[] point = [0.0];
    double minimum;
    opt.optimize(point, minimum);
    assert(opt.getResult() > 0);
    assert(abs(point[0] - target) < 1e-6 && abs(minimum) < 1e-9);
}
```

### Modern API

```d
import nloptd;
import std.math : abs;

void main()
{
    double target = 3;
    auto opt = Optimizer(Algorithm.ldSLSQP, 1);
    opt.setMinObjective((x, gradient) {
        const delta = x[0] - target;
        if (gradient.length) gradient[0] = 2 * delta;
        return delta * delta;
    });
    opt.setXTolRel(1e-8);
    opt.setMaxEvaluations(100);
    double[] point = [0.0];
    auto result = opt.optimizeInPlace(point);
    assert(abs(point[0] - target) < 1e-6 && abs(result.value) < 1e-9);
}
```

The optimizer retains the delegate and its captured context. The coordinate and gradient slices borrow native memory for one callback invocation. Fill every gradient entry when the slice is nonempty. A D exception thrown by a callback stops the solve and propagates back to the caller.

Use `result.termination` to distinguish convergence, a limit, roundoff, and a forced stop. A limit or interruption can leave a partial result. `result.evaluations` reports the objective evaluation count for that run. Native failures such as invalid arguments during a solve throw `NloptException`.

## Common calls

| In 0.2.1 | In the modern wrapper |
| --- | --- |
| `Opt(algorithm, n)` | Construct `Optimizer(algorithm, n)`. For algorithms outside the convenience enum, pass the identifier from `nloptd.raw`. |
| `optimize(point, minf)`, then `getResult()` | Call `optimizeInPlace(point)` and read `value` and `termination` from the returned result. Supply a writable, contiguous `double[]` of length `n`. |
| `setMaxeval`, `setMaxtime`, `setStopval` | Use `setMaxEvaluations`, `setMaxTime` (seconds), and `setStopValue`. |
| `setXTolRel()` or the constructor's implicit tolerance | Set stopping criteria explicitly. Use `setXTolRel(1e-4)` to reproduce the relative coordinate tolerance installed by the 0.2.1 constructor. |
| `setLowerBounds`, `setUpperBounds`, `setInitialStep` | The names are retained. Pass one scalar or a slice with one value per coordinate. Use `resetInitialStep()` to restore automatic step selection. |
| `addEqualityConstraint`, `addInequalityConstraint` | The default feasibility tolerance changed from `1e-8` to zero. To preserve the old default, pass it explicitly, as in `opt.addInequalityConstraint(callback, 1e-8)`. |
| `addEqualityMConstraint`, `addInequalityMConstraint` | Pass the callback and a slice with one tolerance per residual. The slice length supplies the residual count. |
| `removeEqualityConstraints()`, `removeInequalityConstraints()` | The names and grouping are retained. Each clears scalar and vector constraints of its kind between solves. |
| `setLocalOptimizer`, `setPopulation`, `setVectorStorage` | The names are retained. Pass a same-dimension `Optimizer` to `setLocalOptimizer`. NLopt copies its settings. |

## Ownership and lower-level features

`Optimizer` is move-only and releases its native handle at destruction. Pass it by `ref` to helpers that configure it. Use `std.algorithm.mutation.move` to transfer ownership. If you previously used `Opt.copy()` to create two independently configurable solvers, construct two optimizers and apply the shared configuration to each.

If your code used a getter to read back a setting, keep that value alongside the optimizer. For example, save the evaluation limit you pass to `opt.setMaxEvaluations(limit)` and use `limit` when reporting the configuration.

If you already manage a native `nlopt_opt` or need to register a preconditioned callback, use `nloptd.raw` to call the C API on your own handle. An `Optimizer` owns its handle privately. When you create a handle with `nlopt_create`, keep its callback data alive while registered and release the handle with `nlopt_destroy` when finished.
