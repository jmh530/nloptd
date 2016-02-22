nloptd
====
**A D-style wrapper over libnlopt**

## Installation Notes
-------
Ensure that the libnlopt dependency is installed properly. This may involve 
building C lib/dll, depending on platform or compiler. 

## Example
-------
```D
#!/usr/bin/env rdmd

import nloptd;

import std.math : sqrt;

extern(C) double myfunc(
	uint n, const(double)* x, double* grad, void* my_func_data)
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

extern(C) double myconstraint(
	uint n, const(double)* x, double* grad, void* data)
{
    my_constraint_data* d = cast(my_constraint_data*) data;
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

	double[] lb = [-HUGE_VAL, 0];
	my_constraint_data[2] c_data = [{2.0, 0.0}, {-1.0, 1.0}];
	double[] x = [1.234, 5.678];
	double minf;
	
	auto opt = Opt(Algorithm.ldMMA, 2);
	opt.setLowerBounds(lb);
	opt.setMinObjective(&myfunc);
	
	opt.addInequalityConstraint(&myconstraint, c_data[0]);
	opt.addInequalityConstraint(&myconstraint, c_data[1]);
	
	opt.optimize(x, minf);
	
	if (opt.getResult < 0) {
		writeln("nlopt failed!\n");
	}
	else {
		writefln("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
	}
}
