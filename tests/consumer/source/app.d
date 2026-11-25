module app;

import core.memory : GC;
import nloptd;
import nloptd.raw : nlopt_version, NLOPT_LD_SLSQP;
import std.algorithm.mutation : move;
import std.exception : assertThrown;
import std.math : abs;

private Optimizer quadratic(double center)
{
    auto opt = Optimizer(NLOPT_LD_SLSQP, 1);
    opt.setMinObjective((x, gradient) {
        const delta = x[0] - center;
        if (gradient.length) gradient[0] = 2 * delta;
        return delta * delta;
    });
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(200);
    return opt;
}

private double sphere(scope const(double)[] x, scope double[] gradient)
{
    assert(gradient.length == 0);
    return x[0] * x[0];
}

void main()
{
    int major, minor, patch;
    nlopt_version(&major, &minor, &patch);
    assert(major >= 2);
    auto initial = quadratic(2.5);
    static assert(!__traits(compiles, { auto copy = initial; }));
    auto opt = move(initial);
    assertThrown(initial.setMaxEvaluations(1));
    GC.collect();
    double[1] x = [-5];
    auto result = opt.optimizeInPlace(x[]);
    assert(result.evaluations > 0 && abs(result.value) < 1e-10);
    assert(abs(x[0] - 2.5) < 1e-6);
    assertThrown(opt.optimizeInPlace(null));

    auto marker = new Exception("objective failure");
    bool fail = true;
    opt.setMinObjective((x, grad) {
        if (fail) throw marker;
        if (grad.length) grad[0] = 2 * x[0];
        return x[0] * x[0];
    });
    try { opt.optimizeInPlace(x[]); assert(0); }
    catch (Exception e) { assert(e is marker); }
    fail = false;
    result = opt.optimizeInPlace(x[]);
    assert(abs(x[0]) < 1e-6 && result.evaluations > 0);

    opt = Optimizer(Algorithm.lnNelderMead, 1);
    x[0] = 2.5;
    opt.setMinObjective((x, grad) { opt.forceStop(); return 0.0; });
    assert(opt.optimizeInPlace(x[]).termination == Termination.forcedStop);
    opt.setMinObjective(&sphere);
    opt.setMaxEvaluations(1);
    assert(opt.optimizeInPlace(x[]).termination == Termination.maxEvaluationsReached);
    opt.setMaxEvaluations(200);
    opt.setXTolRel(1e-9);
    result = opt.optimizeInPlace(x[]);
    assert(result.evaluations > 1 && abs(x[0]) < 1e-6);
    assertThrown(opt.forceStop());
}
