# nloptd

A D interface to the native NLopt optimization library. The development API provides a move-only optimizer, scalar minimization with D functions and captured delegates, and in-place results. Import `nloptd.raw` for the full C API from your installed `nlopt.h`.

## Build and run

Install a D compiler with direct ImportC header support, DUB, a working host C preprocessor and the NLopt development headers and shared library for your target architecture. Primary targets are DMD 2.114.0 and LDC 1.43.0 on Windows x64 and Ubuntu x86-64. The development tests use NLopt 2.11.0.

On Ubuntu with `libnlopt-dev` and `pkg-config` installed:

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

These settings apply to downstream DUB builds as well. Use headers, import library and DLL from the same installation. The module name `nlopt` must resolve to that header. [NLopt installation guide](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/)

## Scalar minimization

```d
import nloptd;

auto opt = Optimizer(Algorithm.ldSLSQP, 1);
double target = 2;
opt.setMinObjective((x, grad) {
    const delta = x[0] - target;
    if (grad.length) grad[0] = 2 * delta;
    return delta * delta;
});
opt.setXTolRel(1e-9);
opt.setMaxEvaluations(200);

double[1] point = [0.25];
auto result = opt.optimizeInPlace(point[]);
// point[0] is approximately 2. result contains value, termination and evaluations.
```

The callback borrows both slices for one invocation. An empty gradient means NLopt requests only the value. The registration retains the delegate between runs. Exceptions stop the native solve and propagate back to the caller as the original D throwable. Call `forceStop()` from a callback to request an ordinary interrupted result. Interpret partial results using the termination reason.

## Migrating from 0.2.1

Use `Optimizer` in place of `Opt`. Pass it by `ref` for configuration or transfer ownership with `std.algorithm.mutation.move`. Ordinary copies are disabled. Set stopping criteria explicitly. Replace `optimize(x, minf)` and `getResult()` with `optimizeInPlace(x)` and its returned result. Bounds, nonlinear constraints and subsidiary optimizer configuration are available through the raw C API while the high-level port is in progress. The DUB package no longer depends on libnlopt.

The wrapper is licensed under LGPL-2.1-or-later. See COPYING.LESSER.txt and upstream's [native library licensing](https://github.com/stevengj/nlopt/blob/v2.11.0/COPYING).
