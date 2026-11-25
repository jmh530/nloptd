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

private void testBounds()
{
    auto opt = Optimizer(Algorithm.ldSLSQP, 2);
    opt.setMinObjective((x, gradient) {
        const dx = x[0] + 2;
        const dy = x[1] - 3;
        if (gradient.length)
        {
            gradient[0] = 2 * dx;
            gradient[1] = 2 * dy;
        }
        return dx * dx + dy * dy;
    });
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(200);
    opt.setLowerBounds(0.0);
    opt.setUpperBounds(1.0);
    double[2] point = [0.5, 0.5];
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0]) < 1e-6 && abs(point[1] - 1) < 1e-6);
    assert(abs(result.value - 8) < 1e-6);

    double[2] lower = [-1, -5], upper = [5, 2];
    opt.setLowerBounds(lower[]);
    opt.setUpperBounds(upper[]);
    lower[] = 0;
    upper[] = 1;
    assertThrown(opt.setLowerBounds(lower[0 .. 1]));
    assertThrown(opt.setUpperBounds([0.0, 0.0, 0.0]));
    result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] + 1) < 1e-6 && abs(point[1] - 2) < 1e-6);
    assert(abs(result.value - 2) < 1e-6);
}

void main()
{
    testBounds();
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
