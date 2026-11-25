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

private Objective cubicConstraint(double a, double b)
{
    return (x, gradient) {
        const t = a * x[0] + b;
        if (gradient.length)
        {
            gradient[0] = 3 * a * t * t;
            gradient[1] = -1;
        }
        return t * t * t - x[1];
    };
}

private void testMmaTutorial()
{
    import std.math : sqrt;

    auto initial = Optimizer(Algorithm.ldMMA, 2);
    initial.setMinObjective((x, gradient) {
        const value = sqrt(x[1]);
        if (gradient.length)
        {
            gradient[0] = 0;
            gradient[1] = 0.5 / value;
        }
        return value;
    });
    initial.setLowerBounds([-double.infinity, 1e-6]);
    initial.addInequalityConstraint(cubicConstraint(2, 0), 1e-8);
    initial.addInequalityConstraint(cubicConstraint(-1, 1), 1e-8);
    initial.setXTolRel(1e-6);
    initial.setMaxEvaluations(500);
    auto opt = move(initial);
    GC.collect();
    double[2] point = [1.234, 5.678];
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] - 1.0 / 3) < 1e-4);
    assert(abs(point[1] - 8.0 / 27) < 1e-4);
    assert(abs(result.value - sqrt(8.0 / 27)) < 1e-4);
}

private double unitEquality(scope const(double)[] x, scope double[] gradient)
{
    if (gradient.length) gradient[0] = 2 * x[0];
    return x[0] * x[0] - 1;
}

private void testEquality()
{
    auto opt = quadratic(0.5);
    assertThrown(opt.addEqualityConstraint(&unitEquality, -1));
    opt.addEqualityConstraint(&unitEquality, 1e-8);
    double[1] point = [1.5];
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] - 1) < 1e-6 && abs(result.value - 0.25) < 1e-6);
}

private void testConstraintException()
{
    import nloptd.raw : NLOPT_LN_COBYLA;

    auto opt = Optimizer(NLOPT_LN_COBYLA, 1);
    opt.setMinObjective(&sphere);
    auto marker = new Exception("constraint failure");
    bool fail = true;
    opt.addInequalityConstraint((x, gradient) {
        assert(gradient.length == 0);
        if (fail) throw marker;
        return 1 - x[0];
    });
    opt.setXTolRel(1e-8);
    opt.setMaxEvaluations(200);
    double[1] point = [2];
    try { opt.optimizeInPlace(point[]); assert(0); }
    catch (Exception error) { assert(error is marker); }
    fail = false;
    point[0] = 2;
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] - 1) < 1e-6 && abs(result.value - 1) < 1e-6);
}

private void planes(scope double[] residuals, scope const(double)[] x, scope double[] gradient)
{
    assert(residuals.length == 2 && x.length == 3);
    residuals[0] = x[0] + 2 * x[1] - 3;
    residuals[1] = -x[1] + 3 * x[2] - 8;
    if (gradient.length)
        gradient[] = [1.0, 2, 0, 0, -1, 3];
}

private void testVectorEquality()
{
    auto opt = Optimizer(Algorithm.ldSLSQP, 3);
    opt.setMinObjective((x, gradient) {
        if (gradient.length) gradient[] = 2 * x[];
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    });
    opt.addEqualityMConstraint(&planes, null);
    assertThrown(opt.addEqualityMConstraint(&planes, [0.0, double.nan]));
    opt.addEqualityMConstraint(&planes, [1e-9, 1e-8]);
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(200);
    double[3] point = [0, 0, 0];
    auto result = opt.optimizeInPlace(point[]);
    // The closest feasible point to the origin is the sum of the two plane normals.
    assert(abs(point[0] - 1) < 1e-6 && abs(point[1] - 1) < 1e-6 && abs(point[2] - 3) < 1e-6);
    assert(abs(result.value - 11) < 1e-6);
}

private void testVectorInequality()
{
    import nloptd.raw : NLOPT_LN_COBYLA;

    auto initial = Optimizer(NLOPT_LN_COBYLA, 1);
    initial.setMinObjective(&sphere);
    auto marker = new Exception("vector constraint failure");
    bool fail = true;
    initial.addInequalityMConstraint((residuals, x, gradient) {
        assert(gradient.length == 0 && residuals.length == 2);
        residuals[0] = 1 - x[0];
        if (fail) throw marker;
        residuals[1] = x[0] - 3;
    }, [1e-8, 1e-8]);
    initial.setMaxEvaluations(200);
    initial.setXTolRel(1e-8);
    auto opt = move(initial);
    GC.collect();
    double[1] point = [2];
    try { opt.optimizeInPlace(point[]); assert(0); }
    catch (Exception error) { assert(error is marker); }
    fail = false;
    point[0] = 2;
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] - 1) < 1e-6 && abs(result.value - 1) < 1e-6);
}

private void testStoppingCriteria()
{
    auto opt = Optimizer(Algorithm.lnNelderMead, 1);
    opt.setMinObjective((x, gradient) => 1 + x[0] * x[0]);
    opt.setMaxEvaluations(200);
    double[1] point = [3];
    opt.setStopValue(1.25);
    auto result = opt.optimizeInPlace(point[]);
    assert(result.termination == Termination.stopValueReached && result.value <= 1.25);

    opt.setStopValue(-double.infinity);
    opt.setFTolRel(1e-6);
    point[0] = 3;
    assert(opt.optimizeInPlace(point[]).termination == Termination.fToleranceReached);

    opt.setFTolRel(0);
    opt.setFTolAbs(1e-6);
    point[0] = 3;
    assert(opt.optimizeInPlace(point[]).termination == Termination.fToleranceReached);

    opt.setFTolAbs(0);
    opt.setXTolAbs(1e-6);
    point[0] = 3;
    assert(opt.optimizeInPlace(point[]).termination == Termination.xToleranceReached);
}

void main()
{
    testStoppingCriteria();
    testMmaWorkspaceLimit();
    testVectorEquality();
    testVectorInequality();
    testBounds();
    testMmaTutorial();
    testEquality();
    testConstraintException();
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

private void testMmaWorkspaceLimit()
{
    import nloptd.raw : NLOPT_LD_CCSAQ, NLOPT_LD_MMA;

    enum uint n = 32_768;
    enum ulong limit = size_t.sizeof == 4 ? size_t.max / double.sizeof : uint.max;
    enum capacity = (limit - 6UL * n) / (2UL * n + 7);
    auto tolerances = new double[capacity - 1];
    tolerances[] = 0;
    VectorConstraint vector = (values, x, grad) {};
    Objective scalar = (x, grad) => 0.0;
    foreach (algorithm; [NLOPT_LD_MMA, NLOPT_LD_CCSAQ])
    {
        auto opt = Optimizer(algorithm, n);
        opt.addInequalityMConstraint(vector, tolerances);
        assertThrown!NloptException(opt.addEqualityConstraint(scalar));
        assertThrown(opt.addInequalityMConstraint(vector, [0.0, 0.0]));
        opt.addInequalityConstraint(scalar);
        assertThrown(opt.addInequalityConstraint(scalar));
        assertThrown(opt.addInequalityMConstraint(vector, [0.0]));
    }
}
