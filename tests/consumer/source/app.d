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

private void testLocalOptimizer()
{
    auto opt = Optimizer(Algorithm.auglag, 1);
    opt.setMinObjective((x, grad) {
        if (grad.length) grad[0] = 2 * x[0];
        return x[0] * x[0];
    });
    opt.addEqualityConstraint((x, grad) {
        if (grad.length) grad[0] = 1;
        return x[0] - 1;
    }, 1e-8);
    opt.setXTolRel(1e-8);
    opt.setMaxEvaluations(1000);
    {
        auto local = Optimizer(Algorithm.ldSLSQP, 1);
        local.setLowerBounds(5);
        local.setMinObjective((x, grad) { assert(0, "Unexpected local objective"); return 0.0; });
        local.setXTolRel(1e-10);
        opt.setLocalOptimizer(local);
        local.setMaxEvaluations(1);
    }
    auto wrongDimension = Optimizer(Algorithm.ldSLSQP, 2);
    assertThrown!NloptException(opt.setLocalOptimizer(wrongDimension));
    import nloptd.raw : NLOPT_LD_AUGLAG_EQ;
    auto incomplete = Optimizer(NLOPT_LD_AUGLAG_EQ, 1);
    assertThrown(opt.setLocalOptimizer(incomplete));
    assertThrown(opt.setLocalOptimizer(opt));
    Optimizer inert;
    assertThrown(opt.setLocalOptimizer(inert));
    double[1] point = [2];
    auto result = opt.optimizeInPlace(point[]);
    assert(abs(point[0] - 1) < 1e-6 && abs(result.value - 1) < 1e-6);

    incomplete.setMinObjective((x, grad) {
        if (grad.length) grad[0] = 2 * x[0];
        return x[0] * x[0];
    });
    incomplete.setStopValue(0.01);
    incomplete.setMaxEvaluations(200);
    assertThrown(incomplete.optimizeInPlace(point[]));
    auto slsqp = Optimizer(Algorithm.ldSLSQP, 1);
    slsqp.setXTolRel(1e-9);
    incomplete.setLocalOptimizer(slsqp);
    assert(incomplete.optimizeInPlace(point[]).value <= 0.01);
}

void main()
{
    testLocalOptimizer();
    testDualSolverReplacement();
    testNestedAuglagWorkspace();
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

private void testDualSolverReplacement()
{
    import c = nloptd.raw;

    auto slsqp = Optimizer(Algorithm.ldSLSQP, 1);
    slsqp.setXTolRel(1e-8);
    slsqp.setMaxEvaluations(50);
    foreach (algorithm; [c.NLOPT_LD_MMA, c.NLOPT_LD_CCSAQ])
    {
        auto opt = Optimizer(algorithm, 1);
        opt.setLocalOptimizer(slsqp);
        opt.setXTolRel(1e-8);
        opt.setMaxEvaluations(200);
        opt.setMinObjective((x, grad) {
            if (grad.length) grad[0] = 2 * x[0];
            return x[0] * x[0];
        });
        opt.addInequalityConstraint((x, grad) {
            if (grad.length) grad[0] = -1;
            return 1 - x[0];
        });
        foreach (variant; [c.NLOPT_AUGLAG, c.NLOPT_LD_AUGLAG, c.NLOPT_LN_AUGLAG,
                c.NLOPT_AUGLAG_EQ, c.NLOPT_LD_AUGLAG_EQ, c.NLOPT_LN_AUGLAG_EQ])
        {
            auto auglag = Optimizer(variant, 1);
            auglag.setLocalOptimizer(slsqp);
            assertThrown(opt.setLocalOptimizer(auglag));
        }
        double[1] point = [2];
        auto result = opt.optimizeInPlace(point[]);
        assert(abs(point[0] - 1) < 1e-5 && abs(result.value - 1) < 1e-5);
    }
}

private void testMmaWorkspaceLimit()
{
    import nloptd.raw : NLOPT_AUGLAG_EQ, NLOPT_LD_CCSAQ, NLOPT_LD_MMA;

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

        auto parent = Optimizer(NLOPT_AUGLAG_EQ, n);
        parent.addInequalityMConstraint(vector, tolerances);
        parent.setLocalOptimizer(opt);
        parent.addInequalityConstraint(scalar);
        assertThrown(parent.addInequalityConstraint(scalar));
        auto alternative = Optimizer(Algorithm.ldSLSQP, n);
        parent.setLocalOptimizer(alternative);
        parent.addInequalityConstraint(scalar);
        assertThrown(parent.setLocalOptimizer(opt));
    }
}

private void testNestedAuglagWorkspace()
{
    import nloptd.raw : NLOPT_AUGLAG_EQ, NLOPT_LD_AUGLAG_EQ;

    enum uint n = 32_768;
    enum ulong mmaLimit = size_t.sizeof == 4 ? size_t.max / double.sizeof : uint.max;
    enum capacity = (mmaLimit - 6UL * n) / (2UL * n + 7);
    auto tolerances = new double[capacity + 1];
    tolerances[] = 0;
    VectorConstraint vector = (values, x, grad) {};

    auto outer = Optimizer(NLOPT_AUGLAG_EQ, n);
    auto inner = Optimizer(NLOPT_LD_AUGLAG_EQ, n);
    auto mma = Optimizer(Algorithm.ldMMA, n);
    inner.setLocalOptimizer(mma);
    outer.setLocalOptimizer(inner);
    auto slsqp = Optimizer(Algorithm.ldSLSQP, n);
    inner.setLocalOptimizer(slsqp);
    // The copy still uses the explicitly installed MMA until we replace it.
    assertThrown(outer.addInequalityMConstraint(vector, tolerances));
    outer.setLocalOptimizer(inner);
    outer.addInequalityMConstraint(vector, tolerances);

    enum ulong auglagLimit = size_t.sizeof == 4 ? size_t.max / double.sizeof : int.max;
    enum width = (auglagLimit - n) / (n + 2);
    auto wide = new double[width + 1];
    wide[] = 0;
    auto penalized = Optimizer(Algorithm.auglag, n);
    penalized.setLocalOptimizer(slsqp);
    assertThrown(penalized.addEqualityMConstraint(vector, wide));
    // The parent's inequalities would become the nested AUGLAG's penalties.
    outer.addInequalityMConstraint(vector, wide);
    assertThrown(outer.setLocalOptimizer(penalized));
}
