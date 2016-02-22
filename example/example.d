// Written in the D programming language.

/**
 * Sample program implementing nlopt in D
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

import nloptd;

import std.math : sqrt;

extern(C) double myfunc(uint n, const(double)* x, double* grad, void* my_func_data)
{
    if (grad)
	{
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

struct my_constraint_data
{
	double a;
	double b;
}

extern(C) double myconstraint(uint n, const(double)* x, double* grad, void* data)
{
    auto d = cast(my_constraint_data*) data;
    double a = d.a;
	double b = d.b;
    if (grad) {
        grad[0] = 3 * a * (a * x[0] + b) * (a * x[0] + b);
        grad[1] = -1.0;
    }
    return ((a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]);
}



void main()
{
	import std.stdio : writefln, writeln;
	import core.stdc.math : HUGE_VAL;
	import std.experimental.ndslice : sliced;

	double[] lb_pre = [-HUGE_VAL, 0];
	auto lb = lb_pre.sliced(2);
	
	auto opt = Opt(Algorithm.ldMMA, 2);
	opt.setLowerBounds(lb);
	opt.setMinObjective(&myfunc);
	
	my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
	
	opt.addInequalityConstraint(&myconstraint, c_data[0]);
	opt.addInequalityConstraint(&myconstraint, c_data[1]);
	
	double[] xPre = [1.234, 5.678];
	auto x = xPre.sliced(2);
	
	double minf;
	
	opt.optimize(x, minf);
	if (opt.getResult < 0) {
		writeln("nlopt failed!\n");
	}
	else {
		writefln("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
	}
	
	writeln("main ends");

}


