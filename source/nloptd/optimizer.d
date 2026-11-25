/// Optimizer ownership, scalar objectives and termination results.
module nloptd.optimizer;

private import c = nloptd.raw;
private import std.exception : enforce;
private import std.string : fromStringz;

/// Common algorithms. Use the raw constructor for additional upstream identifiers.
enum Algorithm
{
    lnNelderMead = c.NLOPT_LN_NELDERMEAD,
    ldLBFGS = c.NLOPT_LD_LBFGS,
    ldSLSQP = c.NLOPT_LD_SLSQP,
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
/// Fill the whole gradient when nonempty and return the objective value.
alias Objective = double delegate(scope const(double)[] x, scope double[] gradient);

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

// C borrows this record. The owner and an active solve retain it across GC
// collections. Moving the owner preserves the address registered with C.
private struct State
{
    c.nlopt_opt handle;
    Objective objective;
    Throwable pending;
    bool running;
    bool destroyAfterRun;

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
    auto state = cast(State*) data;
    if (!state.pending)
    {
        try
            return state.objective(x[0 .. n], gradient ? gradient[0 .. n] : null);
        catch (Throwable error)
        {
            state.pending = error;
            c.nlopt_force_stop(state.handle);
        }
    }
    return double.infinity;
}

/// Move-only owner. Confine an active optimizer and its callbacks to one thread.
/// A default-initialized or moved-from optimizer is inert.
struct Optimizer
{
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
        state.handle = c.nlopt_create(algorithm, dimension);
        enforce(state.handle, "nlopt_create failed");
    }

    ~this()
    {
        if (!state) return;
        if (state.running)
            state.destroyAfterRun = true;
        else
            c.nlopt_destroy(state.handle);
    }

    private State* idle()
    {
        enforce(state && state.handle, "Optimizer has no native handle");
        enforce(!state.running, "Optimizer is already running");
        return state;
    }

    /** Register or replace the scalar objective to minimize.
        Params:
            objective = Retained delegate. Both slices are valid during the call.
        Throws: Exception for a null objective, inactive owner, active solve or native failure.
    */
    void setMinObjective(Objective objective)
    {
        auto s = idle();
        enforce(objective !is null, "Missing objective");
        s.check(c.nlopt_set_min_objective(s.handle, &evaluate, s));
        s.objective = objective;
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

    /** Request termination from a callback on this optimizer's thread.
        Throws: Exception outside an active solve or on native failure.
    */
    void forceStop()
    {
        enforce(state && state.running, "No active optimization");
        state.check(c.nlopt_force_stop(state.handle));
    }

    /** Minimize the registered objective in place.
        Params:
            point = Initial coordinates, overwritten with the native final point.
                Its length must equal the optimizer dimension.
        Returns: Value, termination reason and evaluation count from this run.
            Interrupted runs may return a partial point.
        Throws: The original callback throwable, NloptException on native failure,
            or Exception for invalid wrapper state or dimension.
    */
    OptimizationResult optimizeInPlace(scope double[] point)
    {
        auto s = idle();
        enforce(s.objective !is null, "Missing objective");
        enforce(point.length == c.nlopt_get_dimension(s.handle), "Dimension mismatch");
        s.running = true;
        scope(exit)
        {
            s.running = false;
            s.pending = null;
            if (s.destroyAfterRun) c.nlopt_destroy(s.handle);
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
}
