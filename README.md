# nloptd

A D interface to the native NLopt optimization library. The development API provides a move-only optimizer, scalar minimization with D functions and captured delegates, and in-place results. Import `nloptd.raw` for the full C API from your installed `nlopt.h`.

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

A local optimizer releases its native handle at scope exit. If the owner is destroyed during an evaluation, the active solve retains the handle until it finishes handling the result or callback exception. For a GC-managed class with an `Optimizer` field, prefer explicit destruction when the resource is no longer needed. If relying on GC finalization, declare a class destructor (an empty `~this() {}` suffices). With DMD 2.114.0, the field destructor alone did not mark the enclosing class allocation for finalization.

## Migrating from 0.2.1

Use `Optimizer` in place of `Opt`. Pass it by `ref` for configuration or transfer ownership with `std.algorithm.mutation.move`. Ordinary copies are disabled. Set stopping criteria explicitly. Replace `optimize(x, minf)` and `getResult()` with `optimizeInPlace(x)` and its returned result. Bounds, nonlinear constraints and subsidiary optimizer configuration are available through the raw C API while the high-level port is in progress. The DUB package no longer depends on libnlopt.

The wrapper is licensed under LGPL-2.1-or-later. See COPYING.LESSER.txt and upstream's [native library licensing](https://github.com/stevengj/nlopt/blob/v2.11.0/COPYING).
