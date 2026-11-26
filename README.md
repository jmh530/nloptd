# nloptd

A D interface to the native NLopt optimization library. The development API provides a move-only optimizer, scalar minimization and maximization with D functions and captured delegates, and in-place results. Import `nloptd.raw` for the full C API from your installed `nlopt.h`.

## Build and run

Install a D compiler with ImportC support, DUB, a working host C preprocessor and the NLopt development headers and shared library for your target architecture. Primary targets are DMD 2.114.0 and LDC 1.43.0 on Windows x64 and Ubuntu x86-64. The development tests use NLopt 2.11.0.

With DMD or LDC on Ubuntu, install `libnlopt-dev` and `pkg-config`, then run:

```sh
export NLOPT_INCLUDE_DIR="$(pkg-config --variable=includedir nlopt)"
dub run --root=example
dub run --root=tests/consumer
```

For custom Unix prefixes, also make the library directory available to the linker and runtime loader. On Linux these can be set with `LIBRARY_PATH` and `LD_LIBRARY_PATH`. On macOS with Homebrew, set `NLOPT_INCLUDE_DIR="$(brew --prefix nlopt)/include"` and `LIBRARY_PATH="$(brew --prefix nlopt)/lib"`.

On Windows, use an x64 MSVC-compatible build of NLopt. For example, after installing `nlopt:x64-windows` with vcpkg, use PowerShell:

```powershell
$prefix = Join-Path $env:VCPKG_ROOT installed/x64-windows
$env:NLOPT_INCLUDE_DIR = Join-Path $prefix include
$env:NLOPT_LIB_DIR = Join-Path $prefix lib
$env:PATH = (Join-Path $prefix bin) + ';' + $env:PATH
dub run --root=example
dub run --root=tests/consumer
```

These settings apply to downstream DUB builds as well. Use the header and native library from the same installation. With DMD and LDC, the module name `nlopt` must resolve to that header. [NLopt installation guide](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/)

### Experimental GDC 16 support on Debian

GDC 16.2 finds `nlopt.h` but rejects directives such as `#ifndef` when importing it directly. Use GDC's C preprocessor to produce `nlopt.i` from the installed header. Select C11 explicitly because GCC 16's default C mode exposes a C23 `nullptr_t` declaration in `<stddef.h>` that this version of ImportC cannot parse. [GDC ImportC manual](https://gcc.gnu.org/onlinedocs/gdc/ImportC.html)

Install `gdc-16`, `dub`, `libnlopt-dev`, and `pkg-config`. From the repository root on Debian sid x86-64, run:

```sh
dc=/usr/bin/gdc-16
header="$(pkg-config --variable=includedir nlopt)/nlopt.h"
export NLOPT_INCLUDE_DIR="$(mktemp -d)"
"$dc" -std=c11 -E -x c "$header" -o "$NLOPT_INCLUDE_DIR/nlopt.i"
dub run --root=tests/consumer --compiler="$dc"
```

For your own DUB project, run the same preprocessing commands, then use `dub build --compiler="$dc"` from your project root. Keep the generated directory available to DUB: `nloptd.raw` imports `nlopt.i` from it and links against the installed NLopt library. Regenerate the file and rebuild with `--force` after changing the headers, compiler, or target. The temporary directory can be removed when those builds are finished.

## Bound-constrained minimization

```d
import nloptd;

auto opt = Optimizer(Algorithm.ldSLSQP, 1);
double target = 2;
opt.setMinObjective((x, grad) {
    const delta = x[0] - target;
    if (grad.length) grad[0] = 2 * delta;
    return delta * delta;
});
opt.setUpperBounds(1.0);
opt.setXTolRel(1e-9);
opt.setMaxEvaluations(200);

double[1] point = [0.25];
auto result = opt.optimizeInPlace(point[]);
// The minimum occurs at the upper bound. Both point[0] and result.value are approximately 1.
```

The callback borrows both slices for one invocation. An empty gradient means NLopt requests only the value. The registration retains the delegate between runs. Exceptions stop the native solve and propagate back to the caller as the original D throwable. Call `forceStop()` from a callback to request an ordinary interrupted result. Interpret partial results using the termination reason.

A local optimizer releases its native handle at scope exit. If the owner is destroyed during an evaluation, the active solve retains the handle until it finishes handling the result or callback exception. For a GC-managed class with an `Optimizer` field, prefer explicit destruction when the resource is no longer needed. If relying on GC finalization, declare a class destructor (an empty `~this() {}` suffices). With DMD 2.114.0, the field destructor alone did not mark the enclosing class allocation for finalization.

## Initial steps for derivative-free optimization

For Nelder–Mead, use `opt.setInitialStep(0.5)` to set the same initial step on every coordinate. When coordinates have different scales, use one signed, nonzero step per coordinate, for example `opt.setInitialStep([0.1, 20.0])`. NLopt copies the steps and keeps them across solves. Call `opt.resetInitialStep()` to let NLopt choose steps from the starting point and bounds again.

## Stopping criteria

Choose criteria in the units of the problem. With `setStopValue(target)`, minimization seeks an objective value below the target and maximization seeks one above it. Whether exact equality triggers `Termination.stopValueReached` depends on the algorithm. A finite target stays in effect when you switch objective direction.

`setFTolAbs` and `setFTolRel` set tolerances on changes in the objective. `setXTolAbs` accepts either a uniform absolute tolerance or one per coordinate. For example, `opt.setXTolAbs([1e-4, 1e-8])` gives a two-dimensional problem different absolute tolerances for its two coordinates. Use `setXTolRel` for a relative coordinate tolerance.

Use `setMaxEvaluations` to limit objective evaluations or `setMaxTime` for an approximate time limit in seconds. An evaluation in progress can run past the time limit. Check `result.termination` to see which criterion ended the run.

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

Constraints follow the objective's slice lifetime and exception rules. Each registration retains its own callback across solves. The [consumer test](tests/consumer/source/app.d) includes the NLopt MMA tutorial with two separately captured cubic inequalities.

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

## Migrating from 0.2.1

Use `Optimizer` in place of `Opt`. Pass it by `ref` for configuration or transfer ownership with `std.algorithm.mutation.move`. Ordinary copies are disabled. Set stopping criteria explicitly. Replace `optimize(x, minf)` and `getResult()` with `optimizeInPlace(x)` and its returned result. Pass a scalar or one value per coordinate to `setLowerBounds` and `setUpperBounds`. Use `setLocalOptimizer` for subsidiary optimizer configuration. The DUB package no longer depends on libnlopt.

## License

The modernized D wrapper is distributed under the [MIT License](LICENSE). The [v0.2.1 release](https://github.com/jmh530/nloptd/tree/v0.2.1) was distributed under LGPL-2.1-or-later.

The separately installed native NLopt library has its own license terms. Upstream NLopt 2.11.0 documents LGPL-2.1-or-later for its default build and MIT for builds that omit the Luksan solvers. [NLopt licensing](https://github.com/stevengj/nlopt/blob/v2.11.0/COPYING)
