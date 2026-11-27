/// Optimizer ownership, scalar objectives, nonlinear constraints and termination results.
module nloptd.optimizer;

private import c = nloptd.raw;
private import std.exception : enforce;
private import std.string : fromStringz;

/// Common algorithms. Use the raw constructor for additional upstream identifiers.
enum Algorithm
{
    lnNelderMead = c.NLOPT_LN_NELDERMEAD,
    ldLBFGS = c.NLOPT_LD_LBFGS,
    ldMMA = c.NLOPT_LD_MMA,
    ldSLSQP = c.NLOPT_LD_SLSQP,
    auglag = c.NLOPT_AUGLAG,
}

/// Reason optimization ended. A limit or interruption can leave a partial result.
enum Termination
{
    success = c.NLOPT_SUCCESS,
    stopValueReached = c.NLOPT_STOPVAL_REACHED,
    fToleranceReached = c.NLOPT_FTOL_REACHED,
    xToleranceReached = c.NLOPT_XTOL_REACHED,
    maxEvaluationsReached = c.NLOPT_MAXEVAL_REACHED,
    maxTimeReached = c.NLOPT_MAXTIME_REACHED,
    roundoffLimited = c.NLOPT_ROUNDOFF_LIMITED,
    forcedStop = c.NLOPT_FORCED_STOP,
}

/// Outcome of one call. NLopt writes the corresponding point into the input slice.
struct OptimizationResult
{
    /// Objective value reported by NLopt.
    double value;
    /// Native stopping reason.
    Termination termination;
    /// Objective evaluations in this run.
    int evaluations;
}

/// Receives the borrowed point and optional writable gradient for one evaluation.
/// Fill the whole gradient when nonempty and return the objective or constraint value.
alias Objective = double delegate(scope const(double)[] x, scope double[] gradient);

/** Evaluate multiple constraint residuals at one point. All slices borrow native
    storage for this call.
    Params:
        residuals = Write all m constraint values, in tolerance order.
        x = The n coordinates of the evaluation point.
        gradient = Empty when derivatives are unnecessary. Otherwise write all m*n
            entries, with the derivative of residual i in coordinate j at i*n+j.
*/
alias VectorConstraint = void delegate(scope double[] residuals, scope const(double)[] x, scope double[] gradient);

/// Function pointer with the same contract as $(D VectorConstraint).
alias VectorConstraintFunction = void function(scope double[] residuals, scope const(double)[] x, scope double[] gradient);

/// Failure reported by a native operation.
class NloptException : Exception
{
    /// Native failure code.
    c.nlopt_result code;
    private this(c.nlopt_result code, string message)
    {
        super(message);
        this.code = code;
    }
}

// The optimizer and each active solve own one reference apiece. A GC
// finalizer can release the owner's reference on another thread.
private struct NativeOwner
{
    c.nlopt_opt handle;
    shared uint references;

    static NativeOwner* create(c.nlopt_algorithm algorithm, uint dimension)
    {
        import core.exception : onOutOfMemoryError;
        import core.stdc.stdlib : free, malloc;

        auto owner = cast(NativeOwner*) malloc(NativeOwner.sizeof);
        if (!owner) onOutOfMemoryError();
        scope(failure) free(owner);
        owner.handle = c.nlopt_create(algorithm, dimension);
        enforce(owner.handle, "nlopt_create failed");
        owner.references = 1;
        return owner;
    }

    void retain()
    {
        import core.atomic : atomicOp;
        atomicOp!"+="(references, 1u);
    }

    void release()
    {
        import core.atomic : atomicOp;
        import core.stdc.stdlib : free;
        if (atomicOp!"-="(references, 1u) == 0)
        {
            c.nlopt_destroy(handle);
            free(&this);
        }
    }
}

// Each native callback borrows a stable record retained by the solve state.
private struct Callback(F)
{
    State* state;
    F callable;
    Callback* nextConstraint;
}

private alias ScalarCallback = Callback!Objective;
private alias VectorCallback = Callback!VectorConstraint;

// NLopt 2.11.0's MMA and CCSAQ workspaces contain 6*n + m*(2*n + 7)
// doubles. The element count is computed in unsigned C arithmetic.
private uint mmaResidualCapacity(uint dimension)
{
    import std.algorithm.comparison : min;
    enum maxElements = min(ulong(uint.max), ulong(size_t.max / double.sizeof));
    const base = 6UL * dimension;
    enforce(base <= maxElements, "MMA/CCSAQ workspace dimension is too large");
    return cast(uint)((maxElements - base) / (2UL * dimension + 7));
}

unittest
{
    import std.exception : assertThrown;
    enum ulong limit = size_t.sizeof == 4 ? size_t.max / double.sizeof : uint.max;
    enum uint lastDimension = cast(uint)(limit / 6);
    assert(mmaResidualCapacity(lastDimension) == 0);
    assertThrown(mmaResidualCapacity(lastDimension + 1));

    enum uint n = 32_768;
    const m = mmaResidualCapacity(n);
    assert(6UL * n + 2UL * m * n + 7UL * m <= limit);
    assert(6UL * n + 2UL * (m + 1) * n + 7UL * (m + 1) > limit);
}

// NLopt 2.11.0 stores these settings as unsigned but passes them to solvers
// as int. Check explicit settings before NLopt copies them into the handle.
private void checkPopulation(c.nlopt_algorithm algorithm, uint n, uint size)
{
    enforce(size <= int.max, "Population size exceeds the native integer range");
    if (size == 0) return;
    enum maxDoubles = size_t.max / double.sizeof;
    switch (algorithm)
    {
        case c.NLOPT_GN_CRS2_LM:
            // crs.c indexes the scratch point at size*(n+1) and allocates
            // one more row for it. Both additions are signed in C.
            const width = 1UL + n;
            enforce(size >= width, "CRS2 needs at least dimension + 1 points");
            enforce(size < int.max && width * size <= int.max &&
                    width * (1UL + size) <= maxDoubles, "CRS2 population workspace is too large");
            break;
        case c.NLOPT_GN_ISRES:
            // isres.c packs two population-by-n arrays, two population
            // vectors and a scratch point into one signed element count.
            const elements = 2UL * size * (1UL + n) + n;
            enforce(elements <= int.max && elements <= maxDoubles, "ISRES population workspace is too large");
            break;
        case c.NLOPT_GN_ESCH:
            // esch.c combines parents and offspring, indexes offspring as
            // int, and multiplies offspring by dimension in unsigned int.
            struct Individual { double* parameters; double fitness; }
            const offspring = 3UL * size / 2;
            const total = size + offspring;
            enforce(offspring <= int.max && total <= uint.max &&
                    total <= size_t.max / Individual.sizeof && offspring * n <= uint.max &&
                    n <= maxDoubles, "ESCH population workspace is too large");
            break;
        case c.NLOPT_G_MLSL, c.NLOPT_G_MLSL_LDS, c.NLOPT_GN_MLSL,
             c.NLOPT_GN_MLSL_LDS, c.NLOPT_GD_MLSL, c.NLOPT_GD_MLSL_LDS:
            enforce(10UL * n + size <= int.max, "MLSL sample count is too large");
            break;
        case c.NLOPT_GD_STOGO_RAND:
            enforce(2UL * n + 1 <= int.max && size < int.max, "StoGO sample count is too large");
            break;
        default: break;
    }
}

private void checkVectorStorage(c.nlopt_algorithm algorithm, uint n, uint count)
{
    import std.algorithm.comparison : max, min;
    enforce(count <= int.max, "Vector storage exceeds the native integer range");
    if (count == 0) return;
    // The Luksan solvers each allocate one double workspace. Their linear
    // terms and number of n-by-count blocks differ (plis.c, plip.c, pnet.c).
    uint vectors, blocks;
    switch (algorithm)
    {
        case c.NLOPT_LD_LBFGS:
            vectors = 4; blocks = 2; break;
        case c.NLOPT_LD_VAR1, c.NLOPT_LD_VAR2:
            vectors = 7; blocks = 1; break;
        case c.NLOPT_LD_TNEWTON, c.NLOPT_LD_TNEWTON_RESTART,
             c.NLOPT_LD_TNEWTON_PRECOND, c.NLOPT_LD_TNEWTON_PRECOND_RESTART:
            vectors = 9; blocks = 2; break;
        default: return;
    }
    enum limit = min(ulong(int.max), ulong(size_t.max / double.sizeof));
    const fixed = ulong(vectors) * n + 2UL * max(n, count);
    enforce(fixed <= limit && ulong(n) * count <= (limit - fixed) / blocks,
            "Limited-memory solver workspace is too large");
}

private struct ResidualGroups
{
    uint total;
    uint largest;
}

private struct ConstraintSizes
{
    ResidualGroups inequalities, equalities;

    ConstraintSizes adding(bool inequality, uint count) const
    {
        import std.algorithm.comparison : max;
        ConstraintSizes result = this;
        auto group = inequality ? &result.inequalities : &result.equalities;
        enforce(count <= uint.max - group.total, "Too many constraint residuals");
        group.total += count;
        group.largest = max(group.largest, count);
        return result;
    }
}

// Mirrors only the native algorithm choices. A subsidiary copy carries no
// registered residuals or D callbacks from its original optimizer.
private struct SolverConfig
{
    c.nlopt_algorithm algorithm;
    const(SolverConfig)* local;

    private bool forwardsInequalities() const
    {
        return algorithm == c.NLOPT_AUGLAG_EQ || algorithm == c.NLOPT_LD_AUGLAG_EQ ||
                algorithm == c.NLOPT_LN_AUGLAG_EQ;
    }

    private bool isAuglag() const
    {
        return forwardsInequalities || algorithm == c.NLOPT_AUGLAG ||
                algorithm == c.NLOPT_LD_AUGLAG || algorithm == c.NLOPT_LN_AUGLAG;
    }

    void checkWorkspace(uint n, ConstraintSizes sizes, bool requireLocal = false) const
    {
        import std.algorithm.comparison : max, min;
        if (algorithm == c.NLOPT_LD_MMA || algorithm == c.NLOPT_LD_CCSAQ)
        {
            enforce(local is null || !local.isAuglag,
                    "MMA and CCSAQ cannot use AUGLAG as their dual solver");
            enforce(sizes.inequalities.total <= mmaResidualCapacity(n),
                    "Too many residuals for the MMA/CCSAQ workspace");
            return;
        }
        if (!isAuglag) return;
        const forward = forwardsInequalities;

        // auglag.c uses signed int for n, residual totals, and the workspace
        // expression n + largest * (1 + n) + equalityTotal + inequalityTotal.
        enforce(n < int.max, "AUGLAG workspace dimension is too large");
        const handled = forward ? ResidualGroups.init : sizes.inequalities;
        const largest = max(handled.largest, sizes.equalities.largest);
        const elements = n + (1UL + n) * largest + sizes.equalities.total + handled.total;
        enum limit = min(ulong(int.max), ulong(size_t.max / double.sizeof));
        enforce(elements <= limit, "AUGLAG workspace is too large");

        ConstraintSizes subsidiary;
        if (forward) subsidiary.inequalities = sizes.inequalities;
        if (local)
            local.checkWorkspace(n, subsidiary, true);
        else
            enforce(!requireLocal, "Configure the AUGLAG local solver with setLocalOptimizer");
    }
}

unittest
{
    import std.exception : assertThrown;
    enum ulong limit = size_t.sizeof == 4 ? size_t.max / double.sizeof : int.max;
    enum uint n = 32_768;
    enum uint largest = cast(uint)((limit - n) / (n + 2));
    enum uint total = cast(uint)(limit - n - largest * (1UL + n));
    auto all = SolverConfig(c.NLOPT_AUGLAG);
    auto eqOnly = SolverConfig(c.NLOPT_AUGLAG_EQ);
    ConstraintSizes sizes;
    sizes.equalities = ResidualGroups(total, largest);
    all.checkWorkspace(n, sizes);
    eqOnly.checkWorkspace(n, sizes);
    assertThrown(all.checkWorkspace(n, sizes.adding(false, 1)));
    assertThrown(eqOnly.checkWorkspace(n, sizes.adding(false, 1)));
    sizes = sizes.adding(true, 1);
    assertThrown(all.checkWorkspace(n, sizes));
    eqOnly.checkWorkspace(n, sizes);
    assertThrown(all.checkWorkspace(int.max, ConstraintSizes.init));
}

private struct ConstraintCallbacks
{
    ScalarCallback* scalar;
    VectorCallback* vector;
}

private struct State
{
    // Borrowed from NativeOwner for configuration and callback dispatch.
    c.nlopt_opt handle;
    ScalarCallback objective;
    ConstraintCallbacks inequalities, equalities;
    Throwable pending;
    bool running;
    SolverConfig solver;
    ConstraintSizes sizes;

    void finishLocalReplacement(c.nlopt_result code, SolverConfig replacement)
    {
        // NLopt destroys the old subsidiary before allocating its replacement.
        if (code == c.NLOPT_OUT_OF_MEMORY) solver.local = null;
        check(code);
        solver = replacement;
    }

    void check(c.nlopt_result code)
    {
        if (code >= 0) return;
        auto message = c.nlopt_get_errmsg(handle);
        if (!message) message = c.nlopt_result_to_string(code);
        throw new NloptException(code, message ? message.fromStringz.idup : "NLopt failed");
    }
}

private extern(C) double evaluate(uint n, const(double)* x, double* gradient, void* data)
{
    auto callback = cast(ScalarCallback*) data;
    auto state = callback.state;
    if (!state.pending)
    {
        try
            return callback.callable(x[0 .. n], gradient ? gradient[0 .. n] : null);
        catch (Throwable error)
        {
            state.pending = error;
            c.nlopt_force_stop(state.handle);
        }
    }
    return double.infinity;
}

private extern(C) void evaluateVector(uint m, double* residuals, uint n, const(double)* x,
                                      double* gradient, void* data)
{
    auto callback = cast(VectorCallback*) data;
    auto state = callback.state;
    auto values = residuals[0 .. m];
    auto derivatives = gradient ? gradient[0 .. cast(size_t) m * n] : null;
    if (!state.pending)
    {
        try
        {
            callback.callable(values, x[0 .. n], derivatives);
            return;
        }
        catch (Throwable error)
        {
            state.pending = error;
            c.nlopt_force_stop(state.handle);
        }
    }
    values[] = double.infinity;
    derivatives[] = 0;
}

private template isMirSlice(S)
{
    import std.traits : moduleName;

    // Only resolve the optional dependency when a Mir slice is passed.
    static if (is(S == T!Args, alias T, Args...) && moduleName!T == "mir.ndslice.slice")
    {
        import mir.ndslice.slice : Slice;
        import std.traits : isInstanceOf;

        enum isMirSlice = isInstanceOf!(Slice, S);
    }
    else
        enum isMirSlice = false;
}

/// Move-only owner. Confine an active optimizer and its callbacks to one thread.
/// A default-initialized or moved-from optimizer is inert.
struct Optimizer
{
    private NativeOwner* owner;
    private State* state;
    @disable this(this);

    /** Create an optimizer with upstream stopping defaults.
        Params:
            algorithm = Optimization algorithm.
            dimension = Positive number of coordinates.
        Throws: Exception for an invalid dimension or failed native creation.
    */
    this(Algorithm algorithm, uint dimension)
    {
        this(cast(c.nlopt_algorithm) algorithm, dimension);
    }

    /** Create an optimizer from an algorithm identifier in nloptd.raw.
        Params:
            algorithm = Upstream algorithm identifier.
            dimension = Positive number of coordinates.
        Throws: Exception for an invalid dimension or failed native creation.
    */
    this(c.nlopt_algorithm algorithm, uint dimension)
    {
        enforce(dimension > 0, "NLopt requires a positive dimension");
        state = new State;
        state.solver = SolverConfig(algorithm);
        state.solver.checkWorkspace(dimension, state.sizes);
        owner = NativeOwner.create(algorithm, dimension);
        state.handle = owner.handle;
    }

    ~this()
    {
        if (owner) owner.release();
    }

    unittest
    {
        import core.memory : GC;
        static class Holder
        {
            Optimizer opt;
            static bool finalized;
            ~this() { finalized = GC.inFinalizer; }
        }

        auto holder = new Holder;
        holder.opt = Optimizer(Algorithm.lnNelderMead, 1);
        c.nlopt_set_munge(holder.opt.owner.handle, &countDestruction, null);
        auto before = destructions;
        // Run the enclosing class's GC finalizer without relying on stack
        // scanning to decide when the test's last reference disappears.
        GC.runFinalizers((cast(const void*) typeid(Holder).destructor)[0 .. 1]);
        assert(Holder.finalized);
        assert(destructions == before + 1);
    }

    private State* idle()
    {
        enforce(state && state.handle, "Optimizer has no native handle");
        enforce(!state.running, "Optimizer is already running");
        return state;
    }

    private void setObjective(alias register)(Objective objective)
    {
        auto s = idle();
        enforce(objective !is null, "Missing objective");
        s.check(register(s.handle, &evaluate, &s.objective));
        s.objective = ScalarCallback(s, objective);
    }

    /** Register or replace the scalar objective to minimize.
        Params:
            objective = Retained delegate. Both slices are valid during the call.
        Throws: Exception for a null objective, inactive owner, active solve or native failure.
    */
    void setMinObjective(Objective objective)
    {
        setObjective!(c.nlopt_set_min_objective)(objective);
    }

    /** Register a D function to minimize.
        Params:
            objective = Function with the same borrowed slice contract as Objective.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void setMinObjective(double function(scope const(double)[], scope double[]) objective)
    {
        import std.functional : toDelegate;
        enforce(objective !is null, "Missing objective");
        setMinObjective(toDelegate(objective));
    }

    /** Register or replace the scalar objective to maximize.
        Params:
            objective = Retained delegate returning the value to maximize. Both
                slices are borrowed for the call. When the gradient is nonempty,
                fill it with the derivatives of the returned value.
        Throws: Exception for a null objective, inactive owner, active solve or native failure.
    */
    void setMaxObjective(Objective objective)
    {
        setObjective!(c.nlopt_set_max_objective)(objective);
    }

    /** Register a D function to maximize.
        Params:
            objective = Function with the same borrowed slice and gradient contract
                as the delegate overload.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void setMaxObjective(double function(scope const(double)[], scope double[]) objective)
    {
        import std.functional : toDelegate;
        enforce(objective !is null, "Missing objective");
        setMaxObjective(toDelegate(objective));
    }

    /** Set the same lower bound on every coordinate.
        Params:
            bound = Uniform bound. Use negative infinity to remove the bound.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setLowerBounds(double bound)
    {
        auto s = idle();
        s.check(c.nlopt_set_lower_bounds1(s.handle, bound));
    }

    /** Set a lower bound for each coordinate.
        Params:
            bounds = One value per dimension, copied by NLopt. Use negative infinity to remove an individual bound.
        Throws: Exception for a dimension mismatch, an inactive owner, active solve or native failure.
    */
    void setLowerBounds(scope const(double)[] bounds)
    {
        auto s = idle();
        enforce(bounds.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.check(c.nlopt_set_lower_bounds(s.handle, bounds.ptr));
    }

    /** Set the same upper bound on every coordinate.
        Params:
            bound = Uniform bound. Use positive infinity to remove the bound.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setUpperBounds(double bound)
    {
        auto s = idle();
        s.check(c.nlopt_set_upper_bounds1(s.handle, bound));
    }

    /** Set an upper bound for each coordinate.
        Params:
            bounds = One value per dimension, copied by NLopt. Use positive infinity to remove an individual bound.
        Throws: Exception for a dimension mismatch, an inactive owner, active solve or native failure.
    */
    void setUpperBounds(scope const(double)[] bounds)
    {
        auto s = idle();
        enforce(bounds.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.check(c.nlopt_set_upper_bounds(s.handle, bounds.ptr));
    }

    /** Set a uniform initial step for algorithms such as Nelder-Mead.
        Params:
            step = Nonzero signed step in coordinate units, used for every dimension.
        Throws: Exception for an inactive owner, active solve, zero step or native failure.
    */
    void setInitialStep(double step)
    {
        auto s = idle();
        s.check(c.nlopt_set_initial_step1(s.handle, step));
    }

    /** Set an initial step for each coordinate.
        Params:
            steps = One nonzero signed step per dimension, in coordinate order.
                NLopt copies these values during registration.
        Throws: Exception for a dimension mismatch, inactive owner, active solve,
            zero step or native failure.
    */
    void setInitialStep(scope const(double)[] steps)
    {
        auto s = idle();
        enforce(steps.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.check(c.nlopt_set_initial_step(s.handle, steps.ptr));
    }

    /** Restore automatic initial steps based on the starting point and bounds.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void resetInitialStep()
    {
        auto s = idle();
        s.check(c.nlopt_set_initial_step(s.handle, null));
    }

    unittest
    {
        import std.exception : assertThrown;
        auto opt = Optimizer(Algorithm.lnNelderMead, 2);
        double[2] point = [3, 5];
        double[2] readSteps()
        {
            double[2] steps;
            opt.state.check(c.nlopt_get_initial_step(opt.state.handle, point.ptr, steps.ptr));
            return steps;
        }

        const automatic = readSteps();
        opt.setInitialStep(-0.5);
        assert(readSteps() == [-0.5, -0.5]);
        double[2] steps = [0.25, -2];
        opt.setInitialStep(steps[]);
        steps[] = 0;
        assertThrown(opt.setInitialStep(null));
        assertThrown(opt.setInitialStep(steps[0 .. 1]));
        assertThrown!NloptException(opt.setInitialStep(0.0));
        assertThrown!NloptException(opt.setInitialStep([1.0, 0.0]));
        assert(readSteps() == [0.25, -2]);
        opt.resetInitialStep();
        assert(readSteps() == automatic);
    }

    /** Set the population size for algorithms such as CRS2 and ISRES.
        Params:
            size = Number of candidate points. Zero selects NLopt's default.
                An explicit size must fit the chosen algorithm's population
                and workspace limits.
        Throws: Exception for an unsupported population size or workspace, an inactive owner,
            an active solve or native failure.
    */
    void setPopulation(uint size)
    {
        auto s = idle();
        checkPopulation(s.solver.algorithm, c.nlopt_get_dimension(s.handle), size);
        s.check(c.nlopt_set_population(s.handle, size));
    }

    unittest
    {
        import std.exception : assertThrown;
        auto opt = Optimizer(c.NLOPT_GN_CRS2_LM, 2);
        opt.setPopulation(40);
        assertThrown(opt.setPopulation(2));
        assertThrown(opt.setPopulation(uint.max));
        assert(c.nlopt_get_population(opt.state.handle) == 40);
        opt.setPopulation(0);
        assert(c.nlopt_get_population(opt.state.handle) == 0);
    }

    unittest
    {
        import std.algorithm.comparison : min;
        import std.exception : assertThrown;
        enum uint n = 8;
        enum ulong maxDoubles = size_t.max / double.sizeof;
        enum limit = min(ulong(int.max), maxDoubles);
        void checkBoundary(c.nlopt_algorithm algorithm, uint last)
        {
            auto opt = Optimizer(algorithm, n);
            opt.setPopulation(last);
            assertThrown(opt.setPopulation(last + 1));
            assert(c.nlopt_get_population(opt.state.handle) == last);
        }
        checkBoundary(c.NLOPT_GN_CRS2_LM,
                cast(uint)min(int.max / (n + 1), maxDoubles / (n + 1) - 1));
        checkBoundary(c.NLOPT_GN_ISRES, cast(uint)((limit - n) / (2 * (n + 1))));
        checkBoundary(c.NLOPT_G_MLSL_LDS, int.max - 10 * n);
        checkBoundary(c.NLOPT_GD_STOGO_RAND, int.max - 1);
        // At this dimension the unsigned offspring-by-dimension product
        // is the first ESCH limit on both 32-bit and 64-bit targets.
        enum uint eschDimension = 128;
        enum uint maxOffspring = uint.max / eschDimension;
        enum uint lastParents = (2 * maxOffspring + 1) / 3;
        auto esch = Optimizer(c.NLOPT_GN_ESCH, eschDimension);
        esch.setPopulation(lastParents);
        assertThrown(esch.setPopulation(lastParents + 1));
        assert(c.nlopt_get_population(esch.state.handle) == lastParents);
    }

    /** Set the history length for limited-memory quasi-Newton algorithms.
        Params:
            count = Number of previous updates to retain, for example in L-BFGS.
                Zero selects NLopt's default. An explicit count must fit the
                chosen algorithm's workspace for the problem dimension.
        Throws: Exception for an unsupported count or workspace, an inactive owner,
            an active solve or native failure.
    */
    void setVectorStorage(uint count)
    {
        auto s = idle();
        checkVectorStorage(s.solver.algorithm, c.nlopt_get_dimension(s.handle), count);
        s.check(c.nlopt_set_vector_storage(s.handle, count));
    }

    unittest
    {
        import std.exception : assertThrown;
        auto opt = Optimizer(Algorithm.ldLBFGS, 2);
        opt.setVectorStorage(5);
        assertThrown(opt.setVectorStorage(uint.max));
        assert(c.nlopt_get_vector_storage(opt.state.handle) == 5);
        opt.setVectorStorage(0);
        assert(c.nlopt_get_vector_storage(opt.state.handle) == 0);
    }

    unittest
    {
        import std.algorithm.comparison : min;
        import std.exception : assertThrown;
        enum uint n = 4;
        enum limit = min(ulong(int.max), ulong(size_t.max / double.sizeof));
        void checkBoundary(c.nlopt_algorithm algorithm, uint last)
        {
            auto opt = Optimizer(algorithm, n);
            opt.setVectorStorage(last);
            assertThrown(opt.setVectorStorage(last + 1));
            assert(c.nlopt_get_vector_storage(opt.state.handle) == last);
        }
        checkBoundary(c.NLOPT_LD_LBFGS, cast(uint)((limit - 4 * n) / (2 * n + 2)));
        foreach (algorithm; [c.NLOPT_LD_VAR1, c.NLOPT_LD_VAR2])
            checkBoundary(algorithm, cast(uint)((limit - 7 * n) / (n + 2)));
        foreach (algorithm; [c.NLOPT_LD_TNEWTON, c.NLOPT_LD_TNEWTON_RESTART,
                c.NLOPT_LD_TNEWTON_PRECOND, c.NLOPT_LD_TNEWTON_PRECOND_RESTART])
            checkBoundary(algorithm, cast(uint)((limit - 9 * n) / (2 * n + 2)));
    }

    private void addConstraint(alias register)(Objective constraint, double tolerance)
    {
        auto s = idle();
        enforce(constraint !is null, "Missing constraint");
        enforce(tolerance >= 0, "Constraint tolerance must be nonnegative");
        enum inequality = __traits(isSame, register, c.nlopt_add_inequality_constraint);
        auto callbacks = inequality ? &s.inequalities : &s.equalities;
        auto sizes = s.sizes.adding(inequality, 1);
        s.solver.checkWorkspace(c.nlopt_get_dimension(s.handle), sizes);
        auto callback = new ScalarCallback(s, constraint, callbacks.scalar);
        s.check(register(s.handle, &evaluate, callback, tolerance));
        s.sizes = sizes;
        callbacks.scalar = callback;
    }

    /** Add a scalar inequality constraint $(I c(x) ≤ 0).
        Params:
            constraint = Retained delegate returning c(x). Fill the gradient when requested.
                Both slices are borrowed for one evaluation.
            tolerance = Nonnegative absolute tolerance for judging feasibility at termination.
        Throws: Exception for a missing callback, invalid tolerance, inactive owner,
            active solve or native failure (including an unsupported algorithm).
    */
    void addInequalityConstraint(Objective constraint, double tolerance = 0)
    {
        addConstraint!(c.nlopt_add_inequality_constraint)(constraint, tolerance);
    }

    /** Add a scalar inequality constraint using a D function.
        Params:
            constraint = Function with the same contract as the delegate overload.
            tolerance = Nonnegative feasibility tolerance, zero by default.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void addInequalityConstraint(double function(scope const(double)[], scope double[]) constraint,
                                 double tolerance = 0)
    {
        import std.functional : toDelegate;
        enforce(constraint !is null, "Missing constraint");
        addInequalityConstraint(toDelegate(constraint), tolerance);
    }

    /** Add a scalar equality constraint $(I h(x) = 0).
        Params:
            constraint = Retained delegate returning h(x). Fill the gradient when requested.
                Both slices are borrowed for one evaluation.
            tolerance = Nonnegative absolute tolerance for judging feasibility at termination.
        Throws: Exception for a missing callback, invalid tolerance, inactive owner,
            active solve or native failure (including an unsupported algorithm).
    */
    void addEqualityConstraint(Objective constraint, double tolerance = 0)
    {
        addConstraint!(c.nlopt_add_equality_constraint)(constraint, tolerance);
    }

    /** Add a scalar equality constraint using a D function.
        Params:
            constraint = Function with the same contract as the delegate overload.
            tolerance = Nonnegative feasibility tolerance, zero by default.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void addEqualityConstraint(double function(scope const(double)[], scope double[]) constraint,
                               double tolerance = 0)
    {
        import std.functional : toDelegate;
        enforce(constraint !is null, "Missing constraint");
        addEqualityConstraint(toDelegate(constraint), tolerance);
    }

    private void addVectorConstraint(alias register)(VectorConstraint constraint, scope const(double)[] tolerances)
    {
        auto s = idle();
        enforce(constraint !is null, "Missing constraint");
        if (!tolerances.length) return;
        enforce(tolerances.length <= uint.max, "Too many constraint residuals");
        const count = cast(uint) tolerances.length;
        enum inequality = __traits(isSame, register, c.nlopt_add_inequality_mconstraint);
        auto callbacks = inequality ? &s.inequalities : &s.equalities;
        auto sizes = s.sizes.adding(inequality, count);
        s.solver.checkWorkspace(c.nlopt_get_dimension(s.handle), sizes);
        enforce(tolerances.length <= size_t.max / double.sizeof / c.nlopt_get_dimension(s.handle),
                "Constraint gradient is too large");
        foreach (tolerance; tolerances)
            enforce(tolerance >= 0, "Constraint tolerance must be nonnegative");
        auto callback = new VectorCallback(s, constraint, callbacks.vector);
        s.check(register(s.handle, count, &evaluateVector, callback, tolerances.ptr));
        s.sizes = sizes;
        callbacks.vector = callback;
    }

    /** Register a vector of inequality constraints, one per tolerance.
        Params:
            constraint = Retained callback. Write each residual (nonpositive when feasible)
                and its derivatives when requested, as described by $(D VectorConstraint).
            tolerances = Nonnegative absolute feasibility tolerances copied by NLopt.
                An empty slice registers no constraints.
        Throws: Exception for a missing callback, invalid size or tolerance, inactive
            owner, active solve or native registration failure.
    */
    void addInequalityMConstraint(VectorConstraint constraint, scope const(double)[] tolerances)
    {
        addVectorConstraint!(c.nlopt_add_inequality_mconstraint)(constraint, tolerances);
    }

    /** Register vector inequality constraints using a D function.
        Params:
            constraint = Function with the same borrowed slices as $(D VectorConstraint).
            tolerances = One nonnegative feasibility tolerance per residual.
                An empty slice registers no constraints.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void addInequalityMConstraint(VectorConstraintFunction constraint, scope const(double)[] tolerances)
    {
        import std.functional : toDelegate;
        enforce(constraint !is null, "Missing constraint");
        addInequalityMConstraint(toDelegate(constraint), tolerances);
    }

    /** Register a vector of equality constraints, one per tolerance.
        Params:
            constraint = Retained callback. Write each residual (zero when feasible)
                and its derivatives when requested, as described by $(D VectorConstraint).
            tolerances = Nonnegative absolute feasibility tolerances copied by NLopt.
                An empty slice registers no constraints.
        Throws: Exception for a missing callback, invalid size or tolerance, inactive
            owner, active solve or native registration failure.
    */
    void addEqualityMConstraint(VectorConstraint constraint, scope const(double)[] tolerances)
    {
        addVectorConstraint!(c.nlopt_add_equality_mconstraint)(constraint, tolerances);
    }

    /** Register vector equality constraints using a D function.
        Params:
            constraint = Function with the same borrowed slices as $(D VectorConstraint).
            tolerances = One nonnegative feasibility tolerance per residual.
                An empty slice registers no constraints.
        Throws: Exception under the same conditions as the delegate overload.
    */
    void addEqualityMConstraint(VectorConstraintFunction constraint, scope const(double)[] tolerances)
    {
        import std.functional : toDelegate;
        enforce(constraint !is null, "Missing constraint");
        addEqualityMConstraint(toDelegate(constraint), tolerances);
    }

    /** Remove all scalar and vector inequality constraints from this optimizer.
        Calling this with no registered inequalities succeeds.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void removeInequalityConstraints()
    {
        auto s = idle();
        s.check(c.nlopt_remove_inequality_constraints(s.handle));
        s.inequalities = ConstraintCallbacks.init;
        s.sizes.inequalities = ResidualGroups.init;
    }

    /** Remove all scalar and vector equality constraints from this optimizer.
        Calling this with no registered equalities succeeds.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void removeEqualityConstraints()
    {
        auto s = idle();
        s.check(c.nlopt_remove_equality_constraints(s.handle));
        s.equalities = ConstraintCallbacks.init;
        s.sizes.equalities = ResidualGroups.init;
    }

    unittest
    {
        auto opt = Optimizer(Algorithm.ldSLSQP, 3);
        Objective scalar = (x, gradient) => x[0];
        VectorConstraint vector = (values, x, gradient) {};
        opt.addEqualityConstraint(scalar);
        opt.addInequalityMConstraint(vector, [0.0, 0.0]);
        opt.addEqualityMConstraint(vector, [0.0, 0.0]);
        opt.addInequalityConstraint(scalar);
        const retained = opt.state.equalities;
        assert(retained.scalar && retained.vector);
        opt.removeInequalityConstraints();
        assert(opt.state.inequalities == ConstraintCallbacks.init);
        assert(opt.state.sizes.inequalities == ResidualGroups.init);
        assert(opt.state.equalities == retained);
        assert(opt.state.sizes.equalities == ResidualGroups(3, 2));
        opt.removeEqualityConstraints();
        assert(opt.state.equalities == ConstraintCallbacks.init);
        assert(opt.state.sizes.equalities == ResidualGroups.init);
    }

    /** Configure the subsidiary solver for an algorithm that uses local optimization.
        NLopt copies the local algorithm and its settings. The parent supplies the
        objective, bounds and constraints when it runs.
        Params:
            local = A distinct, idle optimizer of the same dimension. It can be
                reconfigured or destroyed after this call. An AUGLAG subsidiary
                must already have its own local solver. MMA and CCSAQ reject
                AUGLAG subsidiaries because NLopt discards their nested solver.
        Throws: Exception for an inactive or running optimizer, self-registration,
            missing AUGLAG local solver, AUGLAG dual solver for MMA or CCSAQ,
            incompatible dimensions or workspace size, or native failure.
    */
    void setLocalOptimizer(ref Optimizer local)
    {
        auto s = idle();
        auto source = local.idle();
        enforce(s !is source, "Optimizer cannot be its own local solver");
        source.solver.checkWorkspace(c.nlopt_get_dimension(source.handle), ConstraintSizes.init, true);
        auto replacement = SolverConfig(s.solver.algorithm,
                new SolverConfig(source.solver.algorithm, source.solver.local));
        const dimension = c.nlopt_get_dimension(s.handle);
        if (dimension == c.nlopt_get_dimension(source.handle))
            replacement.checkWorkspace(dimension, s.sizes);
        s.finishLocalReplacement(c.nlopt_set_local_optimizer(s.handle, source.handle), replacement);
    }

    ///
    unittest
    {
        auto opt = Optimizer(Algorithm.auglag, 1);
        auto local = Optimizer(Algorithm.ldSLSQP, 1);
        local.setXTolRel(1e-9);
        opt.setLocalOptimizer(local);
    }

    unittest
    {
        import std.exception : assertThrown;
        auto parent = Optimizer(c.NLOPT_LD_AUGLAG_EQ, 1);
        auto local = Optimizer(Algorithm.ldSLSQP, 1);
        parent.setLocalOptimizer(local);
        // Model the native postcondition of a failed copy without exhausting memory.
        parent.state.check(c.nlopt_set_local_optimizer(parent.state.handle, null));
        assertThrown!NloptException(parent.state.finishLocalReplacement(c.NLOPT_OUT_OF_MEMORY, local.state.solver));
        assert(parent.state.solver.local is null);
        assert(parent.state.solver.algorithm == c.NLOPT_LD_AUGLAG_EQ);
        assertThrown(parent.state.solver.checkWorkspace(1, ConstraintSizes.init, true));
        parent.setLocalOptimizer(local);
        parent.state.solver.checkWorkspace(1, ConstraintSizes.init, true);
    }

    /** Set the target objective value.
        Params:
            value = Objective threshold. Minimization seeks a value below it and
                maximization seeks a value above it. Whether equality triggers
                $(D Termination.stopValueReached) depends on the algorithm.
                Disable the target with negative infinity for minimization or
                positive infinity for maximization.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setStopValue(double value)
    {
        auto s = idle();
        s.check(c.nlopt_set_stopval(s.handle, value));
    }

    /** Set the relative tolerance on changes in the objective value.
        Params:
            tolerance = Fractional change in the objective value. Nonpositive values disable this criterion.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setFTolRel(double tolerance)
    {
        auto s = idle();
        s.check(c.nlopt_set_ftol_rel(s.handle, tolerance));
    }

    /** Set the absolute tolerance on changes in the objective value.
        Params:
            tolerance = Tolerance in objective units. Nonpositive values disable this criterion.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setFTolAbs(double tolerance)
    {
        auto s = idle();
        s.check(c.nlopt_set_ftol_abs(s.handle, tolerance));
    }

    unittest
    {
        // Both criteria report fToleranceReached, but use separate native settings.
        auto opt = Optimizer(Algorithm.lnNelderMead, 1);
        opt.setFTolAbs(0.125);
        opt.setFTolRel(0.25);
        assert(c.nlopt_get_ftol_abs(opt.state.handle) == 0.125);
        assert(c.nlopt_get_ftol_rel(opt.state.handle) == 0.25);
    }

    /** Set the relative tolerance on changes in coordinates.
        Params:
            tolerance = Relative tolerance. Zero disables this criterion.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setXTolRel(double tolerance)
    {
        auto s = idle();
        s.check(c.nlopt_set_xtol_rel(s.handle, tolerance));
    }

    /** Set the same absolute tolerance on every coordinate.
        Params:
            tolerance = Tolerance in coordinate units. Nonpositive values disable this criterion.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setXTolAbs(double tolerance)
    {
        auto s = idle();
        s.check(c.nlopt_set_xtol_abs1(s.handle, tolerance));
    }

    /** Set an absolute tolerance for each coordinate.
        Params:
            tolerances = One tolerance per dimension in coordinate order, copied by NLopt.
                Use positive tolerances for all coordinates to enable this criterion.
        Throws: Exception for a dimension mismatch, inactive owner, active solve or native failure.
    */
    void setXTolAbs(scope const(double)[] tolerances)
    {
        auto s = idle();
        enforce(tolerances.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.check(c.nlopt_set_xtol_abs(s.handle, tolerances.ptr));
    }

    unittest
    {
        import std.exception : assertThrown;
        auto opt = Optimizer(Algorithm.lnNelderMead, 2);
        double[2] actual;
        opt.setXTolAbs(0.5);
        opt.state.check(c.nlopt_get_xtol_abs(opt.state.handle, actual.ptr));
        assert(actual == [0.5, 0.5]);

        double[2] tolerances = [0.25, 0.125];
        opt.setXTolAbs(tolerances[]);
        tolerances[] = 0;
        assertThrown(opt.setXTolAbs(tolerances[0 .. 1]));
        opt.state.check(c.nlopt_get_xtol_abs(opt.state.handle, actual.ptr));
        assert(actual == [0.25, 0.125]);
    }

    /** Set the evaluation limit.
        Params:
            limit = Maximum evaluations requested. Nonpositive values disable the limit.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setMaxEvaluations(int limit)
    {
        auto s = idle();
        s.check(c.nlopt_set_maxeval(s.handle, limit));
    }

    /** Set the approximate time limit for each optimization run.
        Params:
            seconds = Time limit in seconds. Nonpositive values disable the limit.
        Throws: Exception for an inactive owner, active solve or native failure.
    */
    void setMaxTime(double seconds)
    {
        auto s = idle();
        s.check(c.nlopt_set_maxtime(s.handle, seconds));
    }

    unittest
    {
        auto opt = Optimizer(Algorithm.lnNelderMead, 1);
        opt.setMaxTime(0.125);
        assert(c.nlopt_get_maxtime(opt.state.handle) == 0.125);
        opt.setMaxTime(0);
        assert(c.nlopt_get_maxtime(opt.state.handle) == 0);
    }

    /** Request termination from a callback on this optimizer's thread.
        Throws: Exception outside an active solve or on native failure.
    */
    void forceStop()
    {
        enforce(state && state.running, "No active optimization");
        state.check(c.nlopt_force_stop(state.handle));
    }

    /** Optimize the registered objective in place.
        Params:
            point = Initial coordinates, overwritten with the native final point.
                Its length must equal the optimizer dimension.
        Returns: Value, termination reason and evaluation count from this run.
            Interrupted runs may return a partial point.
        Throws: The original callback throwable, NloptException on native failure,
            or Exception for invalid wrapper state or dimension, including an AUGLAG
            optimizer without an explicit local solver.
    */
    OptimizationResult optimizeInPlace(scope double[] point)
    {
        auto s = idle();
        enforce(s.objective.callable !is null, "Missing objective");
        enforce(point.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.solver.checkWorkspace(c.nlopt_get_dimension(s.handle), s.sizes, true);
        auto activeOwner = owner;
        activeOwner.retain();
        scope(exit) activeOwner.release();
        s.running = true;
        scope(exit)
        {
            s.running = false;
            s.pending = null;
        }
        double value;
        auto code = c.nlopt_optimize(s.handle, point.ptr, &value);
        if (s.pending) throw s.pending;
        if (code != c.NLOPT_ROUNDOFF_LIMITED && code != c.NLOPT_FORCED_STOP)
            s.check(code);
        return OptimizationResult(value, cast(Termination)code, c.nlopt_get_numevals(s.handle));
    }

    ///
    unittest
    {
        import std.math : abs;
        auto opt = Optimizer(Algorithm.ldSLSQP, 1);
        double target = 3;
        opt.setMinObjective((x, gradient) {
            const delta = x[0] - target;
            if (gradient.length) gradient[0] = 2 * delta;
            return delta * delta;
        });
        opt.setMaxEvaluations(100);
        double[1] point = [1];
        auto result = opt.optimizeInPlace(point[]);
        assert(abs(point[0] - target) < 1e-6);
        assert(result.value < 1e-10);
    }

    /**
        Optimize the coordinates in a contiguous Mir ndslice in place.
        Params:
            S = Contiguous Mir ndslice type whose field exposes a mutable double[].
            point = Initial coordinates in storage order. The total element count
                must equal the optimizer dimension. The solve writes the final
                coordinates directly into the same storage.
        Returns: An OptimizationResult containing the objective value, termination
            reason and evaluation count. Interrupted runs may leave a partial point.
        Throws: The original callback throwable, NloptException on native failure,
            or Exception for invalid optimizer state or dimension.
    */
    auto optimizeInPlace(S)(scope S point)
        if (isMirSlice!S)
    {
        import mir.ndslice.slice : Contiguous;

        static if (S.kind != Contiguous)
            static assert(false, "optimizeInPlace requires a contiguous Mir slice. "
                ~ "For a strided view, copy the coordinates into a double[], optimize it, and copy the result back.");
        else static if (!is(typeof(point.field) == double[]))
            static assert(false, "optimizeInPlace requires writable double storage. "
                ~ "Copy the coordinates into a mutable double[] and optimize that array.");
        else
            return optimizeInPlace(point.field);
    }

    unittest
    {
        auto opt = Optimizer(Algorithm.lnNelderMead, 1);
        auto error = new Exception("owner destroyed during evaluation");
        auto before = destructions;
        opt.setMinObjective((x, gradient) {
            destroy(opt);
            assert(destructions == before);
            throw error;
            return 0.0;
        });
        c.nlopt_set_munge(opt.owner.handle, &countDestruction, null);
        double[1] point = [1];
        try { opt.optimizeInPlace(point[]); assert(0); }
        catch (Exception caught) { assert(caught is error); }
        assert(destructions == before + 1);
    }
}

version (unittest)
{
    private uint destructions;
    // NLopt invokes this hook on the objective data when freeing the handle.
    private extern(C) void* countDestruction(void*)
    {
        ++destructions;
        return null;
    }
}
