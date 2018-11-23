// Written in the D programming language

/**
 * Functions to manipulate function pointers
 * 
 *
 *
 *
 * Macros:
 * Copyright: Copyright © 2016, John Michael Hall
 * License:   LGPL-2.1 or later
 * Authors:   John Michael Hall
 * Date:      2/21/2016
 */

module manipulateFP;


import std.traits : FunctionAttribute, isFunctionPointer, isDelegate, 
					SetFunctionAttributes, functionLinkage, 
					functionAttributes;
					
alias FA = FunctionAttribute;

enum FAattr = (FA.nothrow_ | FA.nogc);

alias ExternC(T) = SetFunctionAttributes!(
	T, "C", functionAttributes!T);
alias NoThrowNoGC(T) = SetFunctionAttributes!(T, functionLinkage!T, FAattr);
alias ExternCNoThrowNoGC(T) = SetFunctionAttributes!(T, "C", FAattr);

/**************************************
 * Adds externC linkage 
 *
 * ----
 * Params:
 *
 * t = function pointer or delegate to add extern(C) to
 * ----
 */
auto addExternC(T)(T t)
	if (isFunctionPointer!T || isDelegate!T)
{
	return cast(ExternC!(T)) t;
}

unittest
{
	int function(int x) foo;
	
	auto c_foo = addExternC(foo);
	
	static assert(functionLinkage!c_foo == "C");
}

/**************************************
 * Adds nothrow and @nogc attributes
 *
 * ----
 * Params:
 *
 * t = function pointer or delegate to add attributes to
 * ----
 */
auto setNoThrowNoGC(T)(T t)
	if (isFunctionPointer!T || isDelegate!T)
{
	return cast(NoThrowNoGC!(T)) t;
}

unittest
{
	int function(int x) foo;
	
	auto attr_foo = setNoThrowNoGC(foo);
	
	static assert(functionLinkage!attr_foo == "D");
	
	static assert(functionAttributes!attr_foo & FA.nothrow_);
	static assert(functionAttributes!attr_foo & FA.nogc);
}

/**************************************
 * Adds extern(C) and nothrow and @nogc attributes
 *
 * ----
 * Params:
 *
 * t = function pointer or delegate to add extern(C) and attributes to
 * ----
 */
auto addExternCsetNoThrowNoGC(T)(T t)
	if (isFunctionPointer!T || isDelegate!T)
{
	return cast(ExternCNoThrowNoGC!(T)) t;
}

unittest
{
	int function(int x) foo;
	
	auto c_foo = addExternCsetNoThrowNoGC(foo);
	
	static assert(functionLinkage!c_foo == "C");
	
	static assert(functionAttributes!c_foo & FA.nothrow_);
	static assert(functionAttributes!c_foo & FA.nogc);
}


