# nloptd

nloptd lets you minimize or maximize a function written in D using algorithms from NLopt. Import `nloptd` for the D interface or `nloptd.raw` for the C API from your installed `nlopt.h`.

Upgrading from 0.2.1? The [migration guide](MIGRATION.md) shows the same problem with the old and new APIs.

## A first solve

Find the value of `x` closest to 2, subject to `x ≤ 1`. The objective is the squared distance `(x − 2)²`. SLSQP uses its derivative, which the callback writes whenever NLopt requests a gradient.

```d
import nloptd;
import std.stdio : writeln;

void main()
{
    double target = 2;
    auto opt = Optimizer(Algorithm.ldSLSQP, 1);
    opt.setMinObjective((x, gradient) {
        const delta = x[0] - target;
        if (gradient.length) gradient[0] = 2 * delta;
        return delta * delta;
    });
    opt.setUpperBounds(1.0);
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(200);
    double[1] point = [0.25];
    auto result = opt.optimizeInPlace(point[]);
    writeln(point[], " ", result);
}
```

The best point is `x = 1`, where the objective value is also 1. After the solve, `point` holds the final coordinates. Read `result.value` for the objective, `result.termination` for the stopping reason, and `result.evaluations` for the number of objective evaluations.

## Build and run

Use DUB, a D compiler with ImportC support, and a working C preprocessor. Install the NLopt headers and shared library for your target architecture from the same installation. With DMD and LDC, the import name `nlopt` must resolve to that installation’s `nlopt.h`. Primary targets are DMD 2.114.0 and LDC 1.43.0 on Ubuntu x86-64 and Windows x64. CI also tests LDC on macOS.

### Ubuntu

```sh
sudo apt install libnlopt-dev pkg-config
export NLOPT_INCLUDE_DIR="$(pkg-config --variable=includedir nlopt)"
```

For a custom NLopt installation, add its library directory to `LIBRARY_PATH` for linking and `LD_LIBRARY_PATH` for running the program. [Upstream NLopt installation guide](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/)

### macOS

```sh
brew install nlopt
export NLOPT_INCLUDE_DIR="$(brew --prefix nlopt)/include"
export LIBRARY_PATH="$(brew --prefix nlopt)/lib"
```

### Windows x64

Use an MSVC-compatible build of NLopt. With vcpkg, run the following in PowerShell with `VCPKG_ROOT` set to your vcpkg installation.

```powershell
& (Join-Path $env:VCPKG_ROOT 'vcpkg.exe') install nlopt:x64-windows
$prefix = Join-Path $env:VCPKG_ROOT installed/x64-windows
$env:NLOPT_INCLUDE_DIR = Join-Path $prefix include
$env:NLOPT_LIB_DIR = Join-Path $prefix lib
$env:PATH = (Join-Path $prefix bin) + ';' + $env:PATH
```

### Run the example

From the nloptd repository root, run `dub run --root=example`. To run the downstream integration tests, use `dub run --root=tests/consumer`.

### Use nloptd in your application

From your DUB project directory, add nloptd from the registry:

```sh
dub add nloptd@0.3.0
```

To try the [first solve](#a-first-solve) in a new DUB application, save the example as `source/app.d` and run `dub run` with the NLopt environment settings for your platform.

<details>
<summary>Experimental GDC 16 setup on Debian</summary>

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

</details>

## Working with an optimizer

- **Objectives and callbacks.** Use `setMinObjective` or `setMaxObjective` with a D function or captured delegate. The callback can use the coordinate and gradient slices for the duration of the call. [Callbacks and ownership](docs/optimizer.md#callbacks-and-ownership) · [Maximization example](docs/optimizer.md#maximization)
- **Stopping criteria.** Set a tolerance, a target objective value, or a time or evaluation limit. Check the stopping reason before treating the result as converged. [Stopping criteria](docs/optimizer.md#stopping-criteria)
- **Constraints.** Use bounds for coordinate limits and callbacks for nonlinear equalities and inequalities. A vector callback can evaluate several residuals together. [Scalar constraints](docs/optimizer.md#scalar-nonlinear-constraints) · [Vector constraints and gradient layout](docs/optimizer.md#vector-valued-nonlinear-constraints)
- **Algorithm settings.** Choose initial steps for derivative-free methods, tune population size or vector storage, or supply a local solver for AUGLAG. [Initial steps](docs/optimizer.md#initial-steps-for-derivative-free-optimization) · [Population and storage](docs/optimizer.md#population-size-and-vector-storage) · [Subsidiary optimizers](docs/optimizer.md#subsidiary-optimizers)

## License

The modernized D wrapper is distributed under the [MIT License](LICENSE). The [v0.2.1 release](https://github.com/jmh530/nloptd/tree/v0.2.1) was distributed under LGPL-2.1-or-later.

The separately installed native NLopt library has its own license terms. Upstream NLopt 2.11.0 documents LGPL-2.1-or-later for its default build and MIT for builds that omit the Luksan solvers. [NLopt licensing](https://github.com/stevengj/nlopt/blob/v2.11.0/COPYING)
