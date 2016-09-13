// Written in the D programming language.

/**
 * Contains a higher-level D friendly binding to the C nlopt library. 
 * 
 *
 *
 *
 * Copyright: Copyright © 2016, John Michael Hall
 * License:   LGPL-2.1 or later
 * Authors:   John Michael Hall
 * Date:      2/21/2016
 * See_Also:  $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt)
 */

module nloptd;

import nlopt;


private
{
	import std.traits : Parameters, isFunctionPointer, isDelegate;

	alias nloptFunc = extern(C) double function(
		uint n, const(double)* x, double* grad, void* func_data);
									  						  
	alias nloptMFunc = extern(C) void function(
		uint m, double* result, uint n, const(double)* x, double* gradient, 
			void* func_data);
	alias nloptPrecond = extern(C) void function(
		uint n, double* x, const(double)* v, double* vpre, void* data);
		
	alias nloptFunc_ = double function(
		uint n, const(double)* x, double* grad, void* func_data);
									  						  
	alias nloptMFunc_ = void function(
		uint m, double* result, uint n, const(double)* x, double* gradient, 
			void* func_data);
	alias nloptPrecond_ = void function(
		uint n, double* x, const(double)* v, double* vpre, void* data);

	/**************************************
	 * Process function pointer based on nlopt_func for optimization
	 *
	 * ----
	 * Params:
	 *
	 * t = function pointer to process for optimization
	 * ----
	 */
	auto processFP(T)(T t)
		if (isFunctionPointer!T || isDelegate!T)
	{
		import nlopt : nlopt_func;
		

		static if (is(T == nlopt_func))
		{
			return t;
		}
		else static if (is(T == nloptFunc))
		{
			import manipulateFP : setNoThrowNoGC;
			return setNoThrowNoGC(t);
		}
		else static if (is(T == nloptFunc_))
		{
			//assert(0, "extern(C) is missing");
			return t;
			//import manipulateFP : addExternCsetNoThrowNoGC;
			//return addExternCsetNoThrowNoGC(t);
		}
		else
		{
			//assert(0, "unacceptable input");
			return t;
		}
	}

	/**************************************
	 * Process function pointer based on nlopt_precond for optimization
	 *
	 * ----
	 * Params:
	 *
	 * t = function pointer to process for optimization
	 * ----
	 */
	auto processFPPre(T)(T t)
		if (isFunctionPointer!T || isDelegate!T)
	{
		import nlopt : nlopt_precond;
		
		static if (is(T == nlopt_precond))
		{
			return t;
		}
		else static if (is(T == nloptMFunc))
		{
			import manipulateFP : setNoThrowNoGC;
			return setNoThrowNoGC(t);
		}
		else static if (is(T == nloptMFunc_))
		{
			//assert(0, "extern(C) is missing");
			return t;
			//import manipulateFP : addExternCsetNoThrowNoGC;
			//return addExternCsetNoThrowNoGC(t);
		}
		else
		{
			//assert(0, "unacceptable input");
			return t;
		}
	}

	/**************************************
	 * Process function pointer based on nlopt_mfunc for optimization
	 *
	 * ----
	 * Params:
	 *
	 * t = function pointer to process for optimization
	 * ----
	 */
	auto processFPM(T)(T t)
		if (isFunctionPointer!T || isDelegate!T)
	{
		import nlopt : nlopt_mfunc;

		static if (is(T == nlopt_mfunc))
		{
			return t;
		}
		else static if (is(T == nloptMFunc))
		{
			import manipulateFP : setNoThrowNoGC;
			return setNoThrowNoGC(t);
		}
		else static if (is(T == nloptMFunc_))
		{
			//assert(0, "extern(C) is missing");
			return t;
			//import manipulateFP : addExternCsetNoThrowNoGC;
			//return addExternCsetNoThrowNoGC(t);
		}
		else
		{
			//assert(0, "unacceptable input");
			return t;
		}
	}
}

private
{
	import std.traits : Unqual, ForeachType;
	
	/**************************************
	 * ForeachType adjusted to handle const.
	 * 
	 */
	template unqualForeachType(T)
	{
		alias unqualForeachType = Unqual!(ForeachType!(Unqual!T));
	}
	
	/**************************************
	 * Returns true if T has a unqualForeachType equal to double.
	 * 
	 */
	template isDoubleForeach(T)
	{
		static if ( is(unqualForeachType!T == double) )
			enum bool isDoubleForeach = true;
		else
			enum bool isDoubleForeach = false;
	}

	/**************************************
	 * Looks up the equivalent member of T x in E.
	 * 
	 */
	auto lookupEnum(E, T)(T x)
		if (is(T == enum))
	{
		import std.traits : EnumMembers;
		
		foreach (i, member; EnumMembers!E)
		{
			if (x == member)
				return member;
		}
		assert(0, "Not an enum member");
	}
}

/**************************************
 * List of possible algorithms.
 * 
 * Prefix meaning:
 * ----
 * 		gn = non-derivative global optimization algorithm,
 * 		gd = derivative global optimization algorithm,
 * 		ln = non-derivative local optimization algorithm,
 * 		ld = derivative local optimization algorithm,
 * ----
 * 
 * Equivalent in C API: nlopt_algorithm
 */
enum Algorithm
{
    gnDirect					= nlopt_algorithm.NLOPT_GN_DIRECT,
    gnDirectL					= nlopt_algorithm.NLOPT_GN_DIRECT_L,
    gnDirectLRand				= nlopt_algorithm.NLOPT_GN_DIRECT_L_RAND,
    gnDirectNoScal				= nlopt_algorithm.NLOPT_GN_DIRECT_NOSCAL,
    gnDirectLNoScal				= nlopt_algorithm.NLOPT_GN_DIRECT_L_NOSCAL,
    gnDirectLRandNoScal			= nlopt_algorithm.NLOPT_GN_DIRECT_L_RAND_NOSCAL,
    gnOrigDirect				= nlopt_algorithm.NLOPT_GN_ORIG_DIRECT,
    gnOrigDirectL				= nlopt_algorithm.NLOPT_GN_ORIG_DIRECT_L,

	gdStoGo						= nlopt_algorithm.NLOPT_GD_STOGO,
	gdStoGoRand					= nlopt_algorithm.NLOPT_GD_STOGO_RAND,

    ldLBFGSNocedal				= nlopt_algorithm.NLOPT_LD_LBFGS_NOCEDAL,
    ldLBFGS						= nlopt_algorithm.NLOPT_LD_LBFGS,

	lnPraxis					= nlopt_algorithm.NLOPT_LN_PRAXIS,

    ldVar1						= nlopt_algorithm.NLOPT_LD_VAR1,
    ldVar2						= nlopt_algorithm.NLOPT_LD_VAR2,

    ldTNewton					= nlopt_algorithm.NLOPT_LD_TNEWTON,
    ldTNewtonRestart			= nlopt_algorithm.NLOPT_LD_TNEWTON_RESTART,
    ldTNewtonPrecond			= nlopt_algorithm.NLOPT_LD_TNEWTON_PRECOND,
    ldTNewtonPrecondRestart		= nlopt_algorithm.NLOPT_LD_TNEWTON_PRECOND_RESTART,

	gnCRS2LM					= nlopt_algorithm.NLOPT_GN_CRS2_LM,
	gnMlsl						= nlopt_algorithm.NLOPT_GN_MLSL,
	gdMlsl						= nlopt_algorithm.NLOPT_GD_MLSL,
	gnMlslLDS					= nlopt_algorithm.NLOPT_GN_MLSL_LDS,
	gdMlslLDS					= nlopt_algorithm.NLOPT_GD_MLSL_LDS,

    ldMMA						= nlopt_algorithm.NLOPT_LD_MMA,

	lnCOBYLA					= nlopt_algorithm.NLOPT_LN_COBYLA,
	lnNewuoa					= nlopt_algorithm.NLOPT_LN_NEWUOA,
	lnNewuoaBound				= nlopt_algorithm.NLOPT_LN_NEWUOA_BOUND,
	lnNelderMead				= nlopt_algorithm.NLOPT_LN_NELDERMEAD,
	lnSBPLX						= nlopt_algorithm.NLOPT_LN_SBPLX,

 /* new variants that require local_optimizer to be set,
not with older constants for backwards compatibility */

	lnAuglag					= nlopt_algorithm.NLOPT_LN_AUGLAG,
	lnAuglagEQ					= nlopt_algorithm.NLOPT_LN_AUGLAG_EQ,
    ldAuglag					= nlopt_algorithm.NLOPT_LD_AUGLAG,
	ldAuglagEQ					= nlopt_algorithm.NLOPT_LD_AUGLAG_EQ,
	
	lnBOBYQA					= nlopt_algorithm.NLOPT_LN_BOBYQA,
	gnISRES						= nlopt_algorithm.NLOPT_GN_ISRES,

	auglag						= nlopt_algorithm.NLOPT_AUGLAG,
    auglag_eq					= nlopt_algorithm.NLOPT_AUGLAG_EQ,
	
	gMlsl						= nlopt_algorithm.NLOPT_G_MLSL,
    gMlslLQS					= nlopt_algorithm.NLOPT_G_MLSL_LDS,

	ldSLSQP						= nlopt_algorithm.NLOPT_LD_SLSQP,

	ldCCSAQ						= nlopt_algorithm.NLOPT_LD_CCSAQ,

	gnESCH						= nlopt_algorithm.NLOPT_GN_ESCH,
	
    numAlgorithms				= nlopt_algorithm.NLOPT_NUM_ALGORITHMS,
    	//not an algorithm, just the number of them
}

/**************************************
 * List of possible result codes.
 * 
 * Equivalent in C API: nlopt_result
 */
enum Result
{
    failure				= nlopt_result.NLOPT_FAILURE,
    invalidArgs			= nlopt_result.NLOPT_INVALID_ARGS,
    outOfMemory			= nlopt_result.NLOPT_OUT_OF_MEMORY,
    roundoffLimited		= nlopt_result.NLOPT_ROUNDOFF_LIMITED,
    forcedStop			= nlopt_result.NLOPT_FORCED_STOP,
	
    success				= nlopt_result.NLOPT_SUCCESS,
    stopvalReached		= nlopt_result.NLOPT_STOPVAL_REACHED,
    ftolReached			= nlopt_result.NLOPT_FTOL_REACHED,
    xtolReached			= nlopt_result.NLOPT_XTOL_REACHED,
    maxevalReached		= nlopt_result.NLOPT_MAXEVAL_REACHED,
    maxtimeReached		= nlopt_result.NLOPT_MAXTIME_REACHED,
}

/**************************************
 * Default tolerances for some algorithms.
 * 
 */
enum defaultTol : double
{
	inequality	= 1e-8,
	equality	= 1e-8,
	FTolRel		= 1e-4,
	FTolAbs		= 1e-4,
	XTolRel		= 1e-4,
	XTolAbs		= 1e-4,
}


/**************************************
 * Defines an object to interact with nlopt C API.
 * 
 * Equivalent in C API: nlopt_opt
 *
 * Example: TODO
 */
struct Opt
{
	import std.traits : ForeachType;
	
	private
	{
		import nlopt : nlopt_opt, nlopt_result;
		
		nlopt_opt _opt;
		nlopt_result _result;
		
		this(nlopt_opt opt)
		{
			this._opt = opt;
		}
	}
	
	/**************************************
	 * Initializes Opt. 
	 *
	 * ----
	 * Params:
	 *
	 * algorithm = Algorithm to use to optimize.
	 *
	 * n = The dimension of the optimization problem.
	 * ----
	 */
	this(const Algorithm algorithm, const uint n)
	{
		create(algorithm, n);
		this.setXTolRel(defaultTol.XTolRel);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
	}
	
	/**************************************
	 * Calls nlopt_create and sets its result to _opt.
	 *
	 * Equivalent in C API: nlopt_create
	 *
	 * ----
	 * Params:
	 *
	 * algorithm = Algorithm to use to optimize.
	 *
	 * n = The dimension of the optimization problem.
	 * ----
	 */
	void create(const Algorithm algorithm, const uint n)
	{
		import nlopt : nlopt_create;
		
		_opt = nlopt_create(algorithm, n);
	}
	
	///
	unittest
	{
		Opt opt;
		
		opt.create(Algorithm.ldMMA, 2);
	}
	
	/**************************************
	 * Standard destructor.
	 *
	 * Equivalent in C API: nlopt_destroy
	 */
	~this()
	{
		import nlopt : nlopt_destroy;

		nlopt_destroy(_opt);
	}
	
	///
	unittest
	{
		Opt opt;
		opt.create(Algorithm.ldMMA, 2);
		
		destroy(opt);
	}
	
	/**************************************
	 * Sets the seed for stochastic optimization.
	 *
	 * Equivalent in C API: nlopt_srand
	 */
	void srand(uint seed)
	{
		import nlopt : nlopt_srand;
		
		nlopt_srand(seed);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.srand(3215);
	}
	
	/**************************************
	 * Re-sets the seed for stochastic optimization back to system time.
	 *
	 * Equivalent in C API: nlopt_srand_time
	 */
	void srandTime()
	{
		import nlopt : nlopt_srand_time;
		
		nlopt_srand_time();
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.srand(3215);
		
		opt.srandTime();
	}
	
	/**************************************
	 * Copies an instance of Opt to new instance.
	 *
	 * Equivalent in C API: nlopt_copy
	 */
	auto copy()
	{
		import nlopt : nlopt_copy;
		
		auto opt = Opt(nlopt_copy(_opt));
		opt._result = this._result;
		return opt;
	}
	
	///
	unittest
	{
		auto opt1 = Opt(Algorithm.ldMMA, 10);
		
		auto opt2 = opt1.copy();
	}
	
	unittest
	{
		auto opt1 = Opt(Algorithm.ldMMA, 10);
		auto opt2 = opt1.copy();
		
		assert(opt1.getAlgorithm() == opt2.getAlgorithm());
		assert(opt1.getDimension() == opt2.getDimension());
		assert(opt1._result == opt2._result);
		assert(is(typeof(opt1) == typeof(opt2)));
	}
	
	/**************************************
	 * Optimize
	 * 
	 * Equivalent in C API: nlopt_optimize
	 * 
	 * ----
	 * Params:
	 * 
	 * x = the initial guess of the optimization
	 *
	 * min_f = the value of the objective function when the optimization 
	 *         finishes
	 * ----
	 */
	void optimize(T)(ref T x, ref double minf)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == x.length, "the length of x must equal n");

		import nlopt : nlopt_optimize;
		
		processResult(
			nlopt_optimize(_opt, &x[0], &minf));
	}
	
	///
	unittest
	{
		import core.stdc.math : HUGE_VAL;
		import std.math : approxEqual;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] lb = [-HUGE_VAL, 0];
		double[] x = [1.234, 5.678];
		double minf;
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];

		opt.setLowerBounds(lb);
		
		opt.setMinObjective(&myFuncC);

		opt.addInequalityConstraint(&myConstraintC, c_data[0]);
		opt.addInequalityConstraint(&myConstraintC, c_data[1]);
		
		opt.optimize(x, minf);
		
		assert(opt.getResult() > 0);
		assert(approxEqual(x[0], 0.333333));
		assert(approxEqual(x[1], 0.296296));
		assert(approxEqual(minf, 0.5443310476));
	}
	
	unittest
	{
		import core.stdc.math : HUGE_VAL;
		
		double[] lb = [-HUGE_VAL, 0];
		
		double[] x = [1.234, 5.678];
		
		nloptOptimizeTest(x, lb);
	}

	unittest
	{
		import core.stdc.math : HUGE_VAL;
		
		double[2] lb = [-HUGE_VAL, 0];
		
		double[2] x = [1.234, 5.678];
		
		nloptOptimizeTest(x, lb);
	}

	unittest
	{
		import std.experimental.ndslice : sliced;
		import core.stdc.math : HUGE_VAL;

		double[] lbPre = [-HUGE_VAL, 0];
		auto lb = lbPre.sliced(2);
		
		double[] xPre = [1.234, 5.678];
		auto x = xPre.sliced(2);

		nloptOptimizeTest(x, lb);
	}
	
	unittest
	{
		import core.stdc.math : HUGE_VAL;
		import std.container : make, Array;
		
		auto lb = make!(Array!double)(-HUGE_VAL, 0);
		
		auto x = make!(Array!double)(1.234, 5.678);
		
		nloptOptimizeTest(x, lb);
	}
	
	unittest
	{
		import core.stdc.math : HUGE_VAL;
		import std.experimental.allocator : theAllocator, makeArray, dispose;
		
		auto lb = theAllocator.makeArray!double(2, 0);
		lb[0] = -HUGE_VAL;
		
		auto x = theAllocator.makeArray!double(2, 0);
		x[0] = 1.234;
		x[1] = 5.678;
		
		nloptOptimizeTest(x, lb);
		
		theAllocator.dispose(lb);
		theAllocator.dispose(x);
	}
	
	/**************************************
	 * Set Minimum Objective
	 * 
	 * Equivalent in C API: nlopt_set_min_objective
	 * 
	 * ----
	 * Params:
	 * 
	 * f = function pointer representing objective, must be extern(C)
	 *
	 * f_data = additional data to pass to function (optional)
	 * ----
	 */
	void setMinObjective(T, U...)(T f, ref U f_data)
	{
		import nlopt : nlopt_set_min_objective;
		
		static if (f_data.length == 1)
			auto f_data_ = &f_data[0];
		else static if (f_data.length == 0)
			auto f_data_ = null;
		
		processResult(
			nlopt_set_min_objective(
				_opt, processFP(f), f_data_));
	}
	
	///
	unittest
	{
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMinObjective(&myFuncC, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMinObjective(&myFuncC);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMinObjective(&myFuncC_, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMinObjective(&myFuncC_);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Set Maximum Objective
	 *
	 * Equivalent in C API: nlopt_set_max_objective
	 * 
	 * ----
	 * Params:
	 * 
	 * f = function pointer representing objective, must be extern(C)
	 *
	 * f_data = additional data to pass to function (optional)
	 * ----
	 */
	void setMaxObjective(T, U...)(T f, ref U f_data)
	{
		import nlopt : nlopt_set_max_objective;
		
		static if (f_data.length == 1)
			auto f_data_ = &f_data[0];
		else static if (f_data.length == 0)
			auto f_data_ = null;
		
		processResult(
			nlopt_set_max_objective(
				_opt, processFP(f), f_data_));
	}

	///
	unittest
	{
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];

		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMaxObjective(&myFuncC, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMaxObjective(&myFuncC);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMaxObjective(&myFuncC_, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setMaxObjective(&myFuncC_);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Set Minimum Objective with Pre-conditioned Hessian
	 * 
	 * Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Preconditioning_with_approximate_Hessians)
	 * for information on preconditioning. 
	 *
	 * Equivalent in C API: nlopt_set_precond_min_objective
	 * 
	 * ----
	 * Params:
	 * 
	 * f = function pointer representing objective, must be extern(C)
	 * 
	 * pre = function pointer representing preconditioner function, must be 
			 extern(C)
	 *
	 * f_data = additional data to pass to function (optional)
	 * ----
	 */
	void setPrecondMinObjective(T, U, V...)(
		T f, U pre, ref V f_data)
	{
		import nlopt : nlopt_set_precond_min_objective;
		
		static if (f_data.length == 1)
			auto f_data_ = &f_data[0];
		else static if (f_data.length == 0)
			auto f_data_ = null;
		
		processResult(
			nlopt_set_precond_min_objective(
				_opt, processFP(f), processFPPre(pre), f_data_));
	}
	
	//add unittest
	
	/**************************************
	 * Set Maximum Objective with Pre-conditioned Hessian
	 * 
	 * Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Preconditioning_with_approximate_Hessians)
	 * for information on preconditioning. 
	 *
	 * Equivalent in C API: nlopt_set_precond_max_objective
	 * 
	 * ----
	 * Params:
	 * 
	 * f = function pointer representing objective, must be extern(C)
	 * 
	 * pre = function pointer representing preconditioner function, must be 
			 extern(C)
	 *
	 * f_data = additional data to pass to function (optional)
	 * ----
	 */
	void setPrecondMaxObjective(T, U, V...)(
		T f, U pre, ref V f_data)
	{
		import nlopt : nlopt_set_precond_max_objective;
		
		static if (f_data.length == 1)
			auto f_data_ = &f_data[0];
		else static if (f_data.length == 0)
			auto f_data_ = null;
		
		processResult(
			nlopt_set_precond_max_objective(
				_opt, processFP(f), processFPPre(pre), f_data_));
	}
	
	//add unittest
	
	/**************************************
	 * Get Raw Algorithm
	 *
	 * Equivalent in C API: nlopt_get_algorithm
	 *
	 * Returns:
	 *		Raw algorithm from private nlopt_opt object
	 */
	auto getAlgorithmRaw()
	{
		import nlopt : nlopt_get_algorithm;
		
		return nlopt_get_algorithm(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.gnDirect, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnDirectL, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT_L);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnDirectLRand, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT_L_RAND);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnDirectNoScal, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT_NOSCAL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnDirectLNoScal, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT_L_NOSCAL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnDirectLRandNoScal, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_DIRECT_L_RAND_NOSCAL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnOrigDirect, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_ORIG_DIRECT);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnOrigDirectL, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_ORIG_DIRECT_L);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gdStoGo, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GD_STOGO);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gdStoGoRand, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GD_STOGO_RAND);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldLBFGSNocedal, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_LBFGS_NOCEDAL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldLBFGS, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_LBFGS);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_PRAXIS);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldVar1, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_VAR1);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldVar2, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_VAR2);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldTNewton, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_TNEWTON);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldTNewtonRestart, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_TNEWTON_RESTART);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldTNewtonPrecond, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_TNEWTON_PRECOND);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldTNewtonPrecondRestart, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_TNEWTON_PRECOND_RESTART);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnCRS2LM, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_CRS2_LM);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnMlsl, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_MLSL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gdMlsl, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GD_MLSL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnMlslLDS, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_MLSL_LDS);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gdMlslLDS, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GD_MLSL_LDS);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_MMA);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_COBYLA);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnNewuoa, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_NEWUOA);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnNewuoaBound, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_NEWUOA_BOUND);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnNelderMead, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_NELDERMEAD);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnSBPLX, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_SBPLX);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnAuglag, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_AUGLAG);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnAuglagEQ, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_AUGLAG_EQ);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldAuglag, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_AUGLAG);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldAuglagEQ, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_AUGLAG_EQ);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnBOBYQA, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LN_BOBYQA);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnISRES, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_ISRES);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		assert(opt.getAlgorithmRaw == NLOPT_AUGLAG);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.auglag_eq, 2);
		assert(opt.getAlgorithmRaw == NLOPT_AUGLAG_EQ);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gMlsl, 2);
		assert(opt.getAlgorithmRaw == NLOPT_G_MLSL);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gMlslLQS, 2);
		assert(opt.getAlgorithmRaw == NLOPT_G_MLSL_LDS);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_SLSQP);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldCCSAQ, 2);
		assert(opt.getAlgorithmRaw == NLOPT_LD_CCSAQ);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.gnESCH, 2);
		assert(opt.getAlgorithmRaw == NLOPT_GN_ESCH);
	}
	
	/**************************************
	 * Get Algorithm
	 *
	 * Equivalent in C API: nlopt_get_dimension
	 *
	 * Returns:
	 *		The member of Algorithm matching the raw nlopt_algorithm
	 */
	auto getAlgorithm()
	{
		return getAlgorithmRaw().lookupEnum!(Algorithm, nlopt_algorithm)();
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.gnESCH, 2);
		
		assert(opt.getAlgorithm == Algorithm.gnESCH);
	}
	
	/**************************************
	 * Get dimension
	 *
	 * Equivalent in C API: nlopt_get_dimension
	 *
	 * Returns:
	 *		Dimension from Opt object
	 */
	auto getDimension()
	{
		import nlopt : nlopt_get_dimension;
		
		return nlopt_get_dimension(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.gnESCH, 2);
		
		assert(opt.getDimension == 2);
	}
	
/* constraints: */

	/**************************************
	 * Set lower bounds
	 *
	 * Equivalent in C API: nlopt_get_lower_bounds and nlopt_set_lower_bounds1
	 * ----
	 * Params:
	 *
	 * lb = lower bounds
	 * ----
	 */
	void setLowerBounds(T)(ref T lb)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == lb.length, "the length of lb must equal n");

		import nlopt : nlopt_set_lower_bounds;
		
		processResult(
			nlopt_set_lower_bounds(_opt, &lb[0]));
	}
	
	/// ditto	
	void setLowerBounds(double lb)
	{
		import nlopt : nlopt_set_lower_bounds1;
		
		processResult(
			nlopt_set_lower_bounds1(_opt, lb));
	}

	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ub = [-1, 0];
		opt.setLowerBounds(ub);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] ub = [-1, 0];
		opt.setLowerBounds(ub);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ubPre = [-1, 0];
		auto ub = ubPre.sliced(2);
		opt.setLowerBounds(ub);
		
		assert(opt.getResult() > 0);
	}
	
	///
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		opt.setLowerBounds(1);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Get lower bounds
	 *
	 * Equivalent in C API: nlopt_get_lower_bounds
	 * ----
	 * Params:
	 *
	 * lb = storage for output
	 * ----
	 */
	void getLowerBounds(T)(ref T lb)
		if (is(ForeachType!T == double))
	{
		assert(lb.length == getDimension(), 
			"length must match dimension of optimization");

		import nlopt : nlopt_get_lower_bounds;
		
		nlopt_get_lower_bounds(_opt, &lb[0]);
	}
	
	/// ditto
	void getLowerBounds()(ref double lb)
	{
		import nlopt : nlopt_get_lower_bounds;
		
		nlopt_get_lower_bounds(_opt, &lb);
	}
	
	///
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] lb = [1, 0];
		opt.setLowerBounds(lb);
		
		double[] lbNew;
		lbNew.length = 2;
		opt.getLowerBounds(lbNew);
		auto result = cmp(lb[], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] lb = [1, 0];
		opt.setLowerBounds(lb);
		
		double[] lbNew;
		lbNew.length = 2;
		opt.getLowerBounds(lbNew);
		auto result = cmp(lb[], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] lb = [1, 0];
		opt.setLowerBounds(lb);
		
		double[2] lbNew;
		opt.getLowerBounds(lbNew);
		auto result = cmp(lb[], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] lbPre = [1, 0];
		auto lb = lbPre.sliced(2);
		opt.setLowerBounds(lb);
		
		double[] lbNew;
		lbNew.length = 2;
		opt.getLowerBounds(lbNew);
		auto result = cmp(lb[], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] lbPre = [1, 0];
		auto lb = lbPre.sliced(2);
		opt.setLowerBounds(lb);
		
		double[] lbNewPre;
		lbNewPre.length = 2;
		auto lbNew = lbNewPre.sliced(2);
		
		opt.getLowerBounds(lbNew);
		auto result = cmp(lb[], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		opt.setLowerBounds(1);
		
		double[] lbNew;
		lbNew.length = 2;
		opt.getLowerBounds(lbNew);
		auto result = cmp([1, 1], lbNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 1);
		
		opt.setLowerBounds(1);
		
		double lbNew;
		opt.getLowerBounds(lbNew);
		
		assert(lbNew == 1);
	}
	
	/**************************************
	 * Set upper bounds
	 *
	 * Equivalent in C API: nlopt_get_upper_bounds and nlopt_set_upper_bounds1
	 * ----
	 * Params:
	 *
	 * ub = upper bounds
	 * ----
	 */
	void setUpperBounds(T)(ref T ub)
		if (isDoubleForeach!T)
	{
		assert(this.getDimension() == ub.length, 
			"the length of ub must equal n");

		import nlopt : nlopt_set_upper_bounds;
	
		processResult(
			nlopt_set_upper_bounds(_opt, &ub[0]));
	}
	
	/// ditto
	void setUpperBounds(double ub)
	{
		import nlopt : nlopt_set_upper_bounds1;
	
		processResult(
			nlopt_set_upper_bounds1(_opt, ub));
	}
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ub = [1, 0];
		opt.setUpperBounds(ub);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] ub = [1, 0];
		opt.setUpperBounds(ub);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ubPre = [1, 0];
		auto ub = ubPre.sliced(2);
		opt.setUpperBounds(ub);
		
		assert(opt.getResult() > 0);
	}

	///
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);

		opt.setUpperBounds(1);
		
		assert(opt.getResult() > 0);
	}

	/**************************************
	 * Get upper bounds
	 *
	 * Equivalent in C API: nlopt_get_upper_bounds
	 * ----
	 * Params:
	 *
	 * ub = storage for output
	 * ----
	 */
	void getUpperBounds(T)(ref T ub)
		if (is(ForeachType!T == double))
	{
		assert(ub.length == getDimension(), 
			"length must match dimension of optimization");

		import nlopt : nlopt_get_upper_bounds;

		nlopt_get_upper_bounds(_opt, &ub[0]);
	}
	
	/// ditto
	void getUpperBounds(ref double ub)
	{
		import nlopt : nlopt_get_upper_bounds;

		nlopt_get_upper_bounds(_opt, &ub);
	}

	
	///
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ub = [1, 0];
		opt.setUpperBounds(ub);
		
		double[] ubNew;
		ubNew.length = 2;
		opt.getUpperBounds(ubNew);
		auto result = cmp(ub[], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] ub = [1, 0];
		opt.setUpperBounds(ub);
		
		double[] ubNew;
		ubNew.length = 2;
		opt.getUpperBounds(ubNew);
		auto result = cmp(ub[], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[2] ub = [1, 0];
		opt.setUpperBounds(ub);
		
		double[2] ubNew;
		opt.getUpperBounds(ubNew);
		auto result = cmp(ub[], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ubPre = [1, 0];
		auto ub = ubPre.sliced(2);
		opt.setUpperBounds(ub);
		
		double[] ubNew;
		ubNew.length = 2;
		opt.getUpperBounds(ubNew);
		auto result = cmp(ub[], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		double[] ubPre = [1, 0];
		auto ub = ubPre.sliced(2);
		opt.setUpperBounds(ub);
		
		double[] ubNewPre;
		ubNewPre.length = 2;
		auto ubNew = ubNewPre.sliced(2);
		
		opt.getUpperBounds(ubNew);
		auto result = cmp(ub[], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		import std.algorithm : cmp;
		
		auto opt = Opt(Algorithm.ldMMA, 2);
		
		opt.setUpperBounds(1);
		
		double[] ubNew;
		ubNew.length = 2;
		opt.getUpperBounds(ubNew);
		auto result = cmp([1, 1], ubNew[]);
		
		assert(result == 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldMMA, 1);
		
		opt.setUpperBounds(1);
		
		double ubNew;
		opt.getUpperBounds(ubNew);
		
		assert(ubNew == 1);
	}	
	
	/**************************************
	 * Remove inequality constraints
	 *
	 * Equivalent in C API: nlopt_remove_inequality_constraints
	 */
	void removeInequalityConstraints()
	{
		import nlopt : nlopt_remove_inequality_constraints;
	
		processResult(
			nlopt_remove_inequality_constraints(_opt));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addInequalityConstraint(&myConstraintC, c_data, 1e-8);
		
		opt.removeInequalityConstraints();
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Add inequality constraint
	 *
	 * Equivalent in C API: nlopt_add_inequality_constraint
	 * ----
	 * Params:
	 *
	 * fc = function pointer representing constraint, must be extern(C)
	 *
	 * fc_data = additional data to pass to function (optional)
	 *
	 * tol = tolerance for constraint (default set to defaultTol.inequality)
	 * ----
	 */
	void addInequalityConstraint(T, U)(
		T fc, ref U fc_data, const double tol = defaultTol.inequality)
	{
		assert(tol >= 0, "tol must be >= zero");

		import nlopt : nlopt_add_inequality_constraint;
	
		processResult(
			nlopt_add_inequality_constraint(
				_opt, processFP(fc), &fc_data, tol));
	}
	
	void addInequalityConstraint(T)(T fc)
	{
		import nlopt : nlopt_add_inequality_constraint;
	
		processResult(
			nlopt_add_inequality_constraint(
				_opt, processFP(fc), null, defaultTol.inequality));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addInequalityConstraint(&myConstraintC, c_data, 1e-8);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addInequalityConstraint(&myConstraintC, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		opt.addInequalityConstraint(&myConstraintC);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Add pre-conditioned inequality constraint.
	 * 
	 * Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Preconditioning_with_approximate_Hessians)
	 * for information on preconditioning. 
	 *
	 * Equivalent in C API: nlopt_add_precond_inequality_constraint
	 * ----
	 * Params: (TODO)
	 *
	 * fc = function pointer representing constraint, must be extern(C)
	 * 
	 * pre = function pointer representing preconditioner function, must be 
			 extern(C)
	 *
	 * fc_data = additional data to pass to functions (optional)
	 *
	 * tol = tolerance for constraint (default set to defaultTol.inequality)
	 * ----
	 */
	void addPrecondInequalityConstraint(T, U, V)(
		T fc, U pre, ref V fc_data, 
			const double tol = defaultTol.inequality)
	{
		assert(tol >= 0, "tol must be >= zero");

		import nlopt : nlopt_add_precond_inequality_constraint;
	
		processResult(
			nlopt_add_precond_inequality_constraint(
				_opt, processFP(fc), processFPPre(pre), &f_data, tol));
	}
	
	void addPrecondInequalityConstraint(T, U, V)(T fc, U pre)
	{
		assert(tol >= 0, "tol must be >= zero");

		import nlopt : nlopt_add_precond_inequality_constraint;
	
		processResult(
			nlopt_add_precond_inequality_constraint(
				_opt, processFP(fc), processFPPre(pre), null, 
				defaultTol.inequality));
	}
	
	//add unittest
	
	/**************************************
	 * Add multiple inequality constraints
	 *
	 * Equivalent in C API: nlopt_add_inequality_mconstraint
	 * ----
	 * Params:
	 *
	 * m = number of constraints
	 *
	 * fc = function pointer representing constraints, must be extern(C)
	 *
	 * fc_data = additional data to pass to function (optional)
	 *
	 * tol = tolerance for constraints (default set to value of
	 *       defaultTol.inequality)
	 * ----
	 */
	void addInequalityMConstraint(T, U, V)(
			uint m, T fc, ref U fc_data, ref const V tol)
		if (isDoubleForeach!V)
	{
		assert(m == tol.length, "tol must be the same dimension as m");
		foreach (i, t; tol)
		{
			assert(t >= 0, "values of tol must be >= 0");
		}

		import nlopt : nlopt_add_inequality_mconstraint;
	
		processResult(
			nlopt_add_inequality_mconstraint(
				_opt, m, processFPM(fc), &fc_data, &tol[0]));
	}
	
	/// ditto
	void addInequalityMConstraint(T, U)(uint m, T fc, ref U fc_data)
	{
		import nlopt : nlopt_add_inequality_mconstraint;
		import std.algorithm : fill;
		
		auto tol = new double[m];
		fill(tol, defaultTol.inequality);
		
		processResult(
			nlopt_add_inequality_mconstraint(
				_opt, m, processFPM(fc), &fc_data, &tol[0]));
	}
	
	/// ditto
	void addInequalityMConstraint(T)(uint m, T fc)
	{
		import nlopt : nlopt_add_inequality_mconstraint;
		import std.algorithm : fill;
		
		auto tol = new double[m];
		fill(tol, defaultTol.inequality);
		
		processResult(
			nlopt_add_inequality_mconstraint(
				_opt, m, processFPM(fc), null, &tol[0]));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];
		double[] ctol = [1e-6, 1e-6];

		opt.addInequalityMConstraint(m, &myConstraintMC, data, ctol);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];
		double[2] ctol = [1e-6, 1e-6];

		opt.addInequalityMConstraint(m, &myConstraintMC, data, ctol);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];

		opt.addInequalityMConstraint(m, &myConstraintMC, data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;

		opt.addInequalityMConstraint(m, &myConstraintMC);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Remove equality constraints
	 *
	 * Equivalent in C API: nlopt_remove_equality_constraints
	 */
	void removeEqualityConstraints()
	{
		import nlopt : nlopt_remove_equality_constraints;
	
		processResult(
			nlopt_remove_equality_constraints(_opt));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addEqualityConstraint(&myConstraintC, c_data, 1e-8);
		opt.removeEqualityConstraints();
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Add equality constraint
	 *
	 * Equivalent in C API: nlopt_add_equality_constraint
	 * ----
	 * Params:
	 *
	 * fc = function pointer representing constraint, must be extern(C)
	 *
	 * fc_data = additional data to pass to function (optional)
	 *
	 * tol = tolerance for constraints (default set to defaultTol.equality)
	 * ----
	 */
	void addEqualityConstraint(T, U)(
		T fc, ref U fc_data, const double tol = defaultTol.equality)
	{
		assert(tol >= 0, "tol must be >= zero");

		import nlopt : nlopt_add_equality_constraint;
	
		processResult(
			nlopt_add_equality_constraint(_opt, processFP(fc), &fc_data, tol));
	}
	
	/// ditto
	void addEqualityConstraint(T)(T fc)
	{
		import nlopt : nlopt_add_equality_constraint;
	
		processResult(
			nlopt_add_equality_constraint(
				_opt, processFP(fc), null, defaultTol.equality));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addEqualityConstraint(&myConstraintC, c_data, 1e-8);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		my_constraint_data c_data = {2.0, 0.0};
		opt.addEqualityConstraint(&myConstraintC, c_data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.ldSLSQP, 2);
		
		opt.addEqualityConstraint(&myConstraintC);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Add pre-conditioned equality constraint
	 * 
	 * Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Preconditioning_with_approximate_Hessians)
	 * for information on preconditioning. 
	 *
	 * Equivalent in C API: nlopt_add_precond_equality_constraint
	 * ----
	 * Params: (TODO)
	 *
	 * fc = function pointer representing constraint, must be extern(C)
	 *
	 * pre = function pointer representing preconditioner function, must be 
		     extern(C)
	 *
	 * fc_data = additional data to pass to functions (optional)
	 *
	 * tol = tolerance for constraint (default set to defaultTol.equality)
	 * ----
	 */
	void addPrecondEqualityConstraint(T, U, V...)(
		T fc, U pre, ref V fc_data, 
		const double tol = defaultTol.equality)
	{
		assert(tol >= 0, "tol must be >= zero");

		import nlopt : nlopt_add_precond_equality_constraint;
	
		processResult(
			nlopt_add_precond_equality_constraint(
				_opt, processFP(fc), processFPPre(pre), f_data, tol));
	}
	
	void addPrecondEqualityConstraint(T, U, V...)(T fc, U pre)
	{
		import nlopt : nlopt_add_precond_equality_constraint;
	
		processResult(
			nlopt_add_precond_equality_constraint(
				_opt, processFP(fc), processFPPre(pre), null, 
				defaultTol.equality));
	}
	
	//add unittest
	
	/**************************************
	 * Add multiple equality constraints
	 *
	 * Equivalent in C API: nlopt_add_equality_mconstraint
	 * ----
	 * Params:
	 *
	 * m = number of constraints
	 *
	 * fc = function pointer representing constraints, must be extern(C)
	 *
	 * fc_data = additional data to pass to function (optional)
	 *
	 * tol = tolerance for constraints (default set to value of
	 *       defaultTol.equality)
	 * ----
	 */
	void addEqualityMConstraint(T, U, V)(
			uint m, T fc, ref U fc_data, ref const V tol)
		if (isDoubleForeach!V)
	{
		assert(m == tol.length, "tol must be the same dimension as m");
		foreach (i, t; tol)
		{
			assert(t >= 0, "values of tol must be >= 0");
		}

		import nlopt : nlopt_add_equality_mconstraint;
		
		processResult(
			nlopt_add_equality_mconstraint(
				_opt, m, processFPM(fc), &fc_data, &tol[0]));
	}
	
	/// ditto
	void addEqualityMConstraint(T, U)(uint m, T fc, ref U fc_data)
	{
		import nlopt : nlopt_add_equality_mconstraint;
		import std.algorithm : fill;
		
		auto tol = new double[m];
		fill(tol, defaultTol.equality);

		processResult(
			nlopt_add_equality_mconstraint(
				_opt, m, processFPM(fc), &fc_data, &tol[0]));
	}
	
	/// ditto
	void addEqualityMConstraint(T)(uint m, T fc)
	{
		import nlopt : nlopt_add_equality_mconstraint;
		import std.algorithm : fill;
		
		auto tol = new double[m];
		fill(tol, defaultTol.equality);
		
		processResult(
			nlopt_add_equality_mconstraint(
				_opt, m, processFPM(fc), null, &tol[0]));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];
		double[] ctol = [1e-6, 1e-6];

		opt.addEqualityMConstraint(m, &myConstraintMC, data, ctol);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];
		double[2] ctol = [1e-6, 1e-6];

		opt.addEqualityMConstraint(m, &myConstraintMC, data, ctol);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;
		my_constraint_data[2] data = [ {2,0}, {-1,1} ];

		opt.addEqualityMConstraint(m, &myConstraintMC, data);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnCOBYLA, 2);
		
		uint m = 2;

		opt.addEqualityMConstraint(m, &myConstraintMC);
		
		assert(opt.getResult() > 0);
	}

/* stopping criteria: */
	/**************************************
	 * Sets the Stopval. 
	 * 
	 * When performing a minimization, optimization stops when below stopval, 
	 * and vice-versa for maximization.
	 *
	 * Equivalent in C API: nlopt_set_stopval
	 * ----
	 * Params:
	 *
	 * stopval = below 
	 * ----
	 */
	void setStopval(const double stopval)
	{
		import nlopt : nlopt_set_stopval;
	
		processResult(
			nlopt_set_stopval(_opt, stopval));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setStopval(2.0);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the Stopval. 
	 *
	 * When performing a minimization, optimization stops when below stopval, 
	 * and vice-versa for maximization.
	 *
	 * Equivalent in C API: nlopt_get_stopval
	 * 
	 * Returns:
	 *		Stopval
	 */
	auto getStopval()
	{
		import nlopt : nlopt_get_stopval;
	
		return nlopt_get_stopval(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setStopval(2.0);
		
		assert(opt.getStopval() == 2.0);
	}
	
	/**************************************
	 * Sets the relative tolerance for objective function f(x).
	 *
	 * Equivalent in C API: nlopt_set_ftol_rel
	 * ----
	 * Params:
	 *
	 * tol = value of tolerance, default determined by enum defaultTol
	 * ----
	 */
	void setFTolRel(const double tol = defaultTol.FTolRel)
	{
		assert(tol >= 0, "tol must be >= 0");

		import nlopt : nlopt_set_ftol_rel;
	
		processResult(
			nlopt_set_ftol_rel(_opt, tol));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setFTolRel(1e-4);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the relative tolerance for objective function f(x).
	 *
	 * Equivalent in C API: nlopt_get_ftol_rel
	 *
	 * Returns:
	 *		FTolRel
	 */
	auto getFTolRel()
	{
		import nlopt : nlopt_get_ftol_rel;
	
		return nlopt_get_ftol_rel(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setFTolRel(1e-4);
		
		assert(opt.getFTolRel() == 1e-4);
	}
	
	/**************************************
	 * Sets the absolute tolerance for objective function f(x).
	 *
	 * Equivalent in C API: nlopt_set_ftol_abs
	 * ----
	 * Params:
	 *
	 * tol = value of tolerance, default determined by enum defaultTol
	 * ----
	 */
	void setFTolAbs(const double tol = defaultTol.FTolAbs)
	{
		assert(tol >= 0, "tol must be >= 0");

		import nlopt : nlopt_set_ftol_abs;
	
		processResult(
			nlopt_set_ftol_abs(_opt, tol));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setFTolAbs(1e-4);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the absolute tolerance for objective function f(x).
	 *
	 * Equivalent in C API: nlopt_get_ftol_abs
	 * 
	 * Returns:
	 * 		FTolAbs
	 */
	auto getFTolAbs()
	{
		import nlopt : nlopt_get_ftol_abs;
	
		return nlopt_get_ftol_abs(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setFTolAbs(1e-4);
		
		assert(opt.getFTolAbs() == 1e-4);
	}
	
	/**************************************
	 * Sets the relative tolerance for x.
	 *
	 * Equivalent in C API: nlopt_set_xtol_rel
	 * ----
	 * Params:
	 *
	 * tol = value of tolerance, default determined by enum defaultTol
	 * ----
	 */
	void setXTolRel(const double tol = defaultTol.XTolRel)
	{
		assert(tol >= 0, "tol must be >= 0");

		import nlopt : nlopt_set_xtol_rel;
		
		processResult(
			nlopt_set_xtol_rel(_opt, tol));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setXTolRel(1e-4);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the relative tolerance for x.
	 *
	 * Equivalent in C API: nlopt_get_xtol_rel
	 *
	 * Returns:
	 *		XTolRel
	 */
	auto getXTolRel()
	{
		import nlopt : nlopt_get_xtol_rel;
		
		return nlopt_get_xtol_rel(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setXTolRel(1e-4);
		
		assert(opt.getXTolRel() == 1e-4);
	}
	
	/**************************************
	 * Sets the absolute tolerance for x.
	 *
	 * Equivalent in C API: nlopt_set_xtol_abs and nlopt_set_xtol_abs1
	 * ----
	 * Params:
	 *
	 * tol = value of tolerance, default determined by enum defaultTol
	 * ----
	 */
	void setXTolAbs(T)(ref const T tol)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == tol.length, "the length of tol must equal n");
		foreach (i, t; tol)
		{
			assert(t >= 0, "values of tol must be >= 0");
		}

		import nlopt : nlopt_set_xtol_abs;
		
		processResult(
			nlopt_set_xtol_abs(_opt, &tol[0]));
	}
	
	/// ditto
	void setXTolAbs(const double tol = defaultTol.XTolAbs)
	{
		assert(tol >= 0, "tol must be >= 0");

		import nlopt : nlopt_set_xtol_abs1;
	
		processResult(
			nlopt_set_xtol_abs1(_opt, tol));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		double[] test = [1e-4, 1e-4];
		
		opt.setXTolAbs(test);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		double[2] test = [1e-4, 1e-4];
		
		opt.setXTolAbs(test);
		
		assert(opt.getResult() > 0);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setXTolAbs(1e-4);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the absolute tolerance for x.
	 *
	 * Equivalent in C API: nlopt_get_xtol_abs
	 * ----
	 * Params:
	 *
	 * tol = place to put tolerance
	 * ----
	 */
	void getXTolAbs(T)(ref T tol)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == tol.length, "the length of tol must equal n");

		import nlopt : nlopt_get_xtol_abs;
		
		processResult(
			nlopt_get_xtol_abs(_opt, &tol[0]));
	}

	///
	unittest
	{
		import std.algorithm : cmp;
	
		auto opt = Opt(Algorithm.auglag, 2);
		
		double[] test = [1e-4, 1e-4];
		opt.setXTolAbs(test);
		
		double[] val = [0, 0];
		opt.getXTolAbs(val);
		
		auto result = cmp(val[], test[]);
		
		assert(result == 0);
	}
	
	///
	unittest
	{
		import std.algorithm : cmp;
	
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setXTolAbs(1e-4);
		
		double[] val = [0, 0];
		opt.getXTolAbs(val);
		
		double[] test = [1e-4, 1e-4];
		auto result = cmp(val[], test[]);
		
		assert(result == 0);
	}
	
	/**************************************
	 * Sets the maximum number of evaluations.
	 *
	 * Equivalent in C API: nlopt_set_maxeval
	 * ----
	 * Params:
	 *
	 * maxeval = maximum number of evaluations
	 * ----
	 */
	void setMaxeval(const int maxeval)
	{
		assert(maxeval >= 0, "maxval must be >= 0");

		import nlopt : nlopt_set_maxeval;
		
		processResult(
			nlopt_set_maxeval(_opt, maxeval));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setMaxeval(20);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the maximum number of evaluations.
	 *
	 * Equivalent in C API: nlopt_get_maxeval
	 *
	 * Returns:
	 *		Maxeval
	 */
	auto getMaxeval()
	{
		import nlopt : nlopt_get_maxeval;
		
		return nlopt_get_maxeval(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setMaxeval(20);
		assert(opt.getMaxeval() == 20);
	}
	
	/**************************************
	 * Sets the maximum amount of time spent on optimization.
	 *
	 * Equivalent in C API: nlopt_set_maxtime
	 * ----
	 * Params:
	 *
	 * maxtime = maximum time for optimization
	 * ----
	 */
	void setMaxtime(const double maxtime)
	{
		assert(maxtime >= 0, "maxtime must be >= 0");

		import nlopt : nlopt_set_maxtime;
		
		processResult(
			nlopt_set_maxtime(_opt, maxtime));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setMaxtime(2.0);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets the maximum amount of time spent on optimization.
	 *
	 * Equivalent in C API: nlopt_get_maxtime
	 * 
	 * Returns:
	 * 		Maxtime
	 */
	auto getMaxtime()
	{
		import nlopt : nlopt_get_maxtime;
		
		return nlopt_get_maxtime(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		
		opt.setMaxtime(2.0);
		assert(opt.getMaxtime() == 2.0);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Force optimization to stop. Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Forced_termination)
	 *
	 * Equivalent in C API: nlopt_force_stop
	 */
	void forceStop()
	{
		import nlopt : nlopt_force_stop;
		
		processResult(
			nlopt_force_stop(_opt));
	}
	
	//not tested
	
	/**************************************
	 * Force optimization to stop with more information. Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Forced_termination)
	 *
	 * Equivalent in C API: nlopt_set_force_stop
	 * ----
	 * Params:
	 *
	 * val = code to provide more information with ForceStop
	 * ----
	 */
	void setForceStop(const int val)
	{
		assert(val >= 0, "val must be >= 0");

		import nlopt : nlopt_set_force_stop;
		
		processResult(
			nlopt_set_force_stop(_opt, val));
	}
	
	//not tested
	
	/**************************************
	 * Refer to
	 * $(LINK http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Forced_termination)
	 *
	 * Equivalent in C API: nlopt_get_force_stop
	 * 
	 * Returns:
	 *		The last force stop value. 
	 */
	auto getForceStop()
	{
		import nlopt : nlopt_get_force_stop;
		
		return nlopt_get_force_stop(_opt);
	}
	
	//not tested

/* more algorithm-specific parameters */

	/**************************************
	 * From NLopt Reference: "Some of the algorithms, especially MLSL and 
	 * AUGLAG, use a different optimization algorithm as a subroutine, typically
	 * for local optimization."
	 *
	 * Equivalent in C API: nlopt_set_local_optimizer
	 * ----
	 * Params:
	 *
	 * local_opt = optimization object
	 * ----
	 */
	void setLocalOptimizer(nlopt_opt local_opt)
	{
		import nlopt : nlopt_set_local_optimizer;
		
		processResult(
			nlopt_set_local_optimizer(_opt, local_opt));
	}
	
	/// ditto
	void setLocalOptimizer(Opt local_opt)
	{
		import nlopt : nlopt_set_local_optimizer;
		
		nlopt_set_local_optimizer(_opt, local_opt.getOptRaw());
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		auto optLocal = Opt(Algorithm.lnNelderMead, 2);
		
		opt.setLocalOptimizer(optLocal.getOptRaw());
		
		assert(opt.getResult() > 0);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.auglag, 2);
		auto optLocal = Opt(Algorithm.lnNelderMead, 2);
		
		opt.setLocalOptimizer(optLocal);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * For use in stochastic search algorithms, size of initial population of
	 * random points. Default is that it is chosen heuristically by algorithm.
	 *
	 * Equivalent in C API: nlopt_set_population
	 * ----
	 * Params:
	 *
	 * pop = value of population
	 * ----
	 */
	void setPopulation(const uint pop)
	{
		import nlopt : nlopt_set_population;
		
		processResult(
			nlopt_set_population(_opt, pop));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.gnCRS2LM, 2);
		opt.setPopulation(100);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Get population in stochastic search algorithms.
	 *
	 * Equivalent in C API: nlopt_get_population
	 *
	 * Returns:
	 *		Size of population
	 */
	auto getPopulation()
	{
		import nlopt : nlopt_get_population;
		
		return nlopt_get_population(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.gnCRS2LM, 2);
		opt.setPopulation(100);
		auto pop = opt.getPopulation();
		assert(pop == 100);
	}
	
	/**************************************
	 * For use with limited-memory algorithms that remember gradients from 
	 * previous iterations.
	 *
	 * Equivalent in C API: nlopt_set_vector_storage
	 * ----
	 * Params:
	 *
	 * M = number of previous optimization steps to store
	 * ----
	 */
	void setVectorStorage(const uint M)
	{
		import nlopt : nlopt_set_vector_storage;
		
		processResult(
			nlopt_set_vector_storage(_opt, M));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnBOBYQA, 2);
		opt.setVectorStorage(1);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Get vector storage.
	 *
	 * Equivalent in C API: nlopt_get_vector_storage
	 * 
	 * Returns:
	 *		Size of vector storage
	 */
	auto getVectorStorage()
	{
		import nlopt : nlopt_get_vector_storage;
		
		return nlopt_get_vector_storage(_opt);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnBOBYQA, 2);
		opt.setVectorStorage(1);
		
		auto vStorage = opt.getVectorStorage();
		
		assert(vStorage == 1);
	}
	
	/**************************************
	 * Sets default initial step for derivative-free local optimization 
	 * algorithms.
	 *
	 * Equivalent in C API: nlopt_set_default_initial_step
	 * ----
	 * Params:
	 *
	 * dx = Sets the default initial step for each dimension
	 * ----
	 */
	void setDefaultInitialStep(T)(ref T dx)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == dx.length, "the length of dx must equal n");

		import nlopt : nlopt_set_default_initial_step;
		
		processResult(
			nlopt_set_default_initial_step(_opt, &dx[0]));
	}
	
	/// ditto
	void setDefaultInitialStep(ref double dx)
	{
		assert(getDimension() == 1, "the length of dx must equal n");

		import nlopt : nlopt_set_default_initial_step;
		
		processResult(
			nlopt_set_default_initial_step(_opt, &dx));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[] dx = [0.1, 0.1];
		opt.setDefaultInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 1);
		double dx = 0.1;
		opt.setDefaultInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		const double[] dx = [0.1, 0.1];
		opt.setDefaultInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[2] dx = [0.1, 0.1];
		opt.setDefaultInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[] dx_pre = [0.1, 0.1];
		auto dx = dx_pre.sliced(2);
		opt.setDefaultInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Sets initial step for derivative-free local optimization algorithms.
	 *
	 * Equivalent in C API: nlopt_set_initial_step
	 * ----
	 * Params:
	 *
	 * dx = Sets the initial step for each dimension
	 * ----
	 */
	void setInitialStep(T)(ref T dx)
		if (isDoubleForeach!T)
	{
		assert(getDimension() == dx.length, "the length of dx must equal n");

		import nlopt : nlopt_set_initial_step;
		
		processResult(
			nlopt_set_initial_step(_opt, &dx[0]));
	}
	
	/// ditto
	void setInitialStep(const double dx)
	{
		assert(dx >= 0, "the value of dx must be >= 0");

		import nlopt : nlopt_set_initial_step1;
		
		processResult(
			nlopt_set_initial_step1(_opt, dx));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[] dx = [0.1, 0.1];
		opt.setInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		const double[] dx = [0.1, 0.1];
		opt.setInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[2] dx = [0.1, 0.1];
		opt.setInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		import std.experimental.ndslice : sliced;
		
		auto opt = Opt(Algorithm.lnPraxis, 2);
		double[] dx_pre = [0.1, 0.1];
		auto dx = dx_pre.sliced(2);
		opt.setInitialStep(dx);
		
		assert(opt.getResult() > 0);
	}

	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		
		assert(opt.getResult() > 0);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 1);
		opt.setInitialStep(0.1);
		
		assert(opt.getResult() > 0);
	}
	
	/**************************************
	 * Gets initial step for derivative-free local optimization algorithms.
	 *
	 * Equivalent in C API: nlopt_get_initial_step
	 * ----
	 * Params:
	 *
	 * x = Same value as initial guess to pass to optimize.
	 *
	 * dx = On successful return will contain the initial step sizes
	 * ----
	 */
	void getInitialStep(T, U)(ref T x, ref U dx)
		if (isDoubleForeach!T && isDoubleForeach!U)
	{
		assert(getDimension() == x.length, 
			"the length of x must equal n");
		assert(x.length == dx.length, 
			"the length of x must equal the length of dx");

		import nlopt : nlopt_get_initial_step;
		
		processResult(
			nlopt_get_initial_step(_opt, &x[0], &dx[0]));
	}
	
	/// ditto
	void getInitialStep(ref double x, ref double dx)
	{
		assert(getDimension() == 1, "the length of dx must equal n");

		import nlopt : nlopt_get_initial_step;
		
		processResult(
			nlopt_get_initial_step(_opt, &x, &dx));
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		double[] x = [1.234, 5.678];
		double[] dx = [0, 0];
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 1);
		opt.setInitialStep(0.1);
		
		double x = 1.234;
		double dx = 0;
		opt.getInitialStep(x, dx);
		assert(dx == 0.1);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		double[] x = [1.234, 5.678];
		double[] dx = [0, 0];
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		double[2] x = [1.234, 5.678];
		double[2] dx = [0, 0];
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		const double[2] x = [1.234, 5.678];
		double[2] dx = [0, 0];
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	unittest
	{
		import std.experimental.ndslice : sliced;
	
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		double[] xPre = [1.234, 5.678];
		auto x = xPre.sliced(2);
		double[] dxPre = [0, 0];
		auto dx = dxPre.sliced(2);
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	unittest
	{
		import std.experimental.ndslice : sliced;
	
		auto opt = Opt(Algorithm.lnPraxis, 2);
		opt.setInitialStep(0.1);
		const double[] xPre = [1.234, 5.678];
		auto x = xPre.sliced(2);
		double[] dxPre = [0, 0];
		auto dx = dxPre.sliced(2);
		
		opt.getInitialStep(x, dx);
		assert(dx[0] == 0.1);
		assert(dx[1] == 0.1);
	}
	
	/**************************************
	 * Gets raw opt object
	 *
	 * Equivalent in C API: nlopt_opt
	 *
	 * Returns:
	 *		The private nlopt_opt object
	 */
	nlopt_opt getOptRaw()
	{
		return _opt;
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		auto optRaw = opt.getOptRaw();
		
		assert(is(typeof(optRaw) == nlopt_opt));
	}
	
	/**************************************
	 * Gets the raw result value.
	 *
	 * Equivalent in C API: nlopt_result
	 *
	 * Returns:
	 *		The value of private member _result
	 */
	auto getResultRaw()
	{
		return _result;
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		
		auto resultRaw = opt.getResultRaw();
	}
	
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		
		assert(opt.getResultRaw() == NLOPT_SUCCESS);
	}
	
	/**************************************
	 * Gets the result value.
	 *
	 * Returns:
	 *		The equivalent of private member _result as a member of Result
	 */
	auto getResult()
	{
		return getResultRaw().lookupEnum!(Result, nlopt_result)();
	}
	
	///
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		
		auto result = opt.getResult();
	}
	unittest
	{
		auto opt = Opt(Algorithm.lnPraxis, 2);
		
		assert(opt.getResult() == Result.success);
	}
	
	/**************************************
	 * Processes a member of nlopt_result. 
	 * 
	 * Throws exceptions under certain circumstances and destroys _opt in the 
	 * event of failure.
	 *
	 */
	void processResult(nlopt_result result)
	{
		if (result > 0)
		{
			_result = result;
		}
		else
		{
			import std.exception : enforce;
			
			scope(failure) destroy(_opt);
			
			enforce(result != -1, 
				"nlopt error code -1: generic");
			enforce(result != -2, 
				"nlopt error code -2: invalid arguments");
			enforce(result != -3, 
				"nlopt error code -3: out of memory");
			enforce(result != -4, 
				"nlopt error code -4: roundoff errors limited progress");
			enforce(result != -5, 
				"nlopt error code -5: forced termination");
			enforce(0, 
				"nlopt_result was negative but not the right code");	
			_result = result;
		}
	}
}

version(unittest)
{
	extern(C) double myFuncC_(
		uint n, const(double)* x, double* gradient, void* func_data)
	{
		import std.math : sqrt;
	
		if (gradient)
		{
			gradient[0] = 0.0;
			gradient[1] = 0.5 / sqrt(x[1]);
		}
		return sqrt(x[1]);
	}
	
	extern(C) double myConstraintC_(
		uint n, const(double)* x, double* gradient, void* data)
	{
		my_constraint_data* d = cast(my_constraint_data*) data;
		double a = d.a;
		double b = d.b;
		if (gradient)
		{
			gradient[0] = 3 * a * (a * x[0] + b) * (a * x[0] + b);
			gradient[1] = -1.0;
		}
		return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
	}
	
	extern(C) void myconstraintMC_(
		uint m, double* result, uint n, const(double)* x, double* gradient, 
			void* func_data)
	{
		  my_constraint_data* d = cast(my_constraint_data*) func_data;
		  uint i;
		  for (i = 0; i < m; ++i) {
			   double a = d[i].a, b = d[i].b;
			   if (gradient)
			   {
					gradient[i*n + 0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
					gradient[i*n + 1] = -1.0;
			   }
			   result[i] = (a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1];
		  }
	}
}

version(unittest)
{
	double myFuncD(
		uint n, const(double)* x, double* gradient, void* func_data)
	{
		import std.math : sqrt;
	
		if (gradient)
		{
			gradient[0] = 0.0;
			gradient[1] = 0.5 / sqrt(x[1]);
		}
		return sqrt(x[1]);
	}
	
	double myConstraintD(
		uint n, const(double)* x, double* gradient, void* data)
	{
		my_constraint_data* d = cast(my_constraint_data*) data;
		double a = d.a;
		double b = d.b;
		if (gradient)
		{
			gradient[0] = 3 * a * (a * x[0] + b) * (a * x[0] + b);
			gradient[1] = -1.0;
		}
		return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
	}
	
	void myconstraintMD(
		uint m, double* result, uint n, const(double)* x, double* gradient, 
			void* func_data)
	{
		  my_constraint_data* d = cast(my_constraint_data*) func_data;
		  uint i;
		  for (i = 0; i < m; ++i) {
			   double a = d[i].a, b = d[i].b;
			   if (gradient)
			   {
					gradient[i*n + 0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
					gradient[i*n + 1] = -1.0;
			   }
			   result[i] = (a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1];
		  }
	}
}

@nogc nothrow version(unittest)
{
	struct my_constraint_data
	{
		double a;
		double b;
	}

	extern(C) double myFuncC(
		uint n, const(double)* x, double* gradient, void* func_data)
	{
		import std.math : sqrt;
	
		if (gradient)
		{
			gradient[0] = 0.0;
			gradient[1] = 0.5 / sqrt(x[1]);
		}
		return sqrt(x[1]);
	}
	
	extern(C) double myConstraintC(
		uint n, const(double)* x, double* gradient, void* data)
	{
		my_constraint_data* d = cast(my_constraint_data*) data;
		double a = d.a;
		double b = d.b;
		if (gradient)
		{
			gradient[0] = 3 * a * (a * x[0] + b) * (a * x[0] + b);
			gradient[1] = -1.0;
		}
		return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
	}
	
	extern(C) void myConstraintMC(uint m, double* result, uint n, 
		const(double)* x, double* gradient, void* func_data)
	{
		  my_constraint_data* d = cast(my_constraint_data*) func_data;
		  uint i;
		  for (i = 0; i < m; ++i)
		  {
			   double a = d[i].a, b = d[i].b;
			   if (gradient)
			   {
					gradient[i*n + 0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
					gradient[i*n + 1] = -1.0;
			   }
			   result[i] = (a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1];
		  }
	}
}

version(unittest)
{
	void nloptOptimizeTest(T, U)(T x, U lb)
	{
		import std.stdio : writefln, writeln;
		
		import std.math : approxEqual;

		auto opt = Opt(Algorithm.ldMMA, 2);
		opt.setLowerBounds(lb);
		opt.setMinObjective(&myFuncC);
		
		my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
		
		opt.addInequalityConstraint(&myConstraintC, c_data[0], 1e-8);
		opt.addInequalityConstraint(&myConstraintC, c_data[1], 1e-8);
		
		double minf;
		
		opt.optimize(x, minf);
		
		assert(opt.getResult() > 0);
		assert(approxEqual(x[0], 0.333333));
		assert(approxEqual(x[1], 0.296296));
		assert(approxEqual(minf, 0.5443310476));
	}
}

///
unittest
{
	import core.stdc.math : HUGE_VAL;
	import std.math : approxEqual;
	
	auto opt = Opt(Algorithm.ldMMA, 2);
	
	double[] lb = [-HUGE_VAL, 0];
	double[] x = [1.234, 5.678];
	double minf;
	my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];

	opt.setLowerBounds(lb);
	
	opt.setMinObjective(&myFuncC);

	opt.addInequalityConstraint(&myConstraintC, c_data[0]);
	opt.addInequalityConstraint(&myConstraintC, c_data[1]);
	
	opt.optimize(x, minf);
	
	assert(opt.getResult() > 0);
	assert(approxEqual(x[0], 0.333333));
	assert(approxEqual(x[1], 0.296296));
	assert(approxEqual(minf, 0.5443310476));
}

unittest
{
	import core.stdc.math : HUGE_VAL;
	import std.math : approxEqual;
	
	auto opt = Opt(Algorithm.ldMMA, 2);
	
	double[] lb = [-HUGE_VAL, 0];
	double[] x = [1.234, 5.678];
	double minf;
	my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];

	opt.setLowerBounds(lb);
	
	opt.setMinObjective(&myFuncC_);

	opt.addInequalityConstraint(&myConstraintC_, c_data[0]);
	opt.addInequalityConstraint(&myConstraintC_, c_data[1]);
	
	opt.optimize(x, minf);
	
	assert(opt.getResult() > 0);
	assert(approxEqual(x[0], 0.333333));
	assert(approxEqual(x[1], 0.296296));
	assert(approxEqual(minf, 0.5443310476));
}

unittest
{
	import core.stdc.math : HUGE_VAL;
	import std.math : approxEqual;
	
	auto opt = Opt(Algorithm.ldMMA, 2);
	
	double[] lb = [-HUGE_VAL, 0];
	double[] x = [1.234, 5.678];
	double minf;
	my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];

	opt.setLowerBounds(lb);
	
	opt.setMinObjective(&myFuncC_);

	opt.addInequalityConstraint(&myConstraintC_, c_data[0]);
	opt.addInequalityConstraint(&myConstraintC_, c_data[1]);
	
	opt.optimize(x, minf);
	
	assert(opt.getResult() > 0);
	assert(approxEqual(x[0], 0.333333));
	assert(approxEqual(x[1], 0.296296));
	assert(approxEqual(minf, 0.5443310476));
}

















