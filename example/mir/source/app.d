static if (__traits(compiles, { import mir.ndslice.slice; }))
{
    import mir.ndslice.slice : sliced;
    import mir.ndslice.topology : assumeContiguous, map, universal;
}
else
    static assert(false, "This example uses mir-algorithm. Run 'dub add mir-algorithm' in your application directory.");
import nloptd;
import std.math : abs;
import std.stdio : writeln;

/// Minimize the squared distance to (1, -2) using the same objective for each layout.
void main()
{
    auto opt = Optimizer(Algorithm.ldSLSQP, 2);
    opt.setMinObjective((coordinates, gradient) {
        // nloptd passes built-in D slices: const(double)[] for the point and double[] for the gradient.
        // Use .sliced to view either as a Mir ndslice.
        auto x = coordinates.sliced;
        const dx = x[0] - 1;
        const dy = x[1] + 2;
        if (gradient.length)
        {
            auto g = gradient.sliced;
            g[0] = 2 * dx;
            g[1] = 2 * dy;
        }
        return dx * dx + dy * dy;
    });
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(100);

    optimizeContiguousPoint(opt);
    optimizeContiguousRow(opt);
    optimizeStridedColumn(opt);
}

/// Solve from the same starting point using each overload. Both the contiguous
/// ndslice and its field refer to storage, which receives the final coordinates.
private void optimizeContiguousPoint(ref Optimizer opt)
{
    double[2] storage = [5, 6];
    auto point = storage[].sliced;
    opt.optimizeInPlace(point);
    assert(abs(storage[0] - 1) < 1e-7 && abs(storage[1] + 2) < 1e-7);
    writeln("Contiguous point: ", storage[]);

    storage[] = [5.0, 6.0];
    assert(point.field is storage[]);
    opt.optimizeInPlace(point.field);
    assert(abs(storage[0] - 1) < 1e-7 && abs(storage[1] + 2) < 1e-7);
    writeln("Backing array: ", storage[]);
}

/// Optimize the rows of a padded canonical matrix individually. The second row
/// can be passed directly. For the first row, demonstrate how to establish
/// contiguity when the caller has a universal view of the same storage.
private void optimizeContiguousRow(ref Optimizer opt)
{
    double[6] entries = [10, 20, 30, 5, 6, 40];
    auto matrix = entries[].sliced(2, 3);
    auto padded = matrix[0 .. $, 0 .. 2];
    opt.optimizeInPlace(padded[1]);
    assert(abs(entries[3] - 1) < 1e-7 && abs(entries[4] + 2) < 1e-7);
    assert(entries[0 .. 3] == [10, 20, 30] && entries[5] == 40);

    auto row = padded[0].universal;
    assert(row.strides[0] == 1);
    opt.optimizeInPlace(row.assumeContiguous);
    assert(abs(entries[0] - 1) < 1e-7 && abs(entries[1] + 2) < 1e-7);
    assert(entries[2] == 30 && entries[5] == 40);
}

/// Optimize the middle column using a two-element work array. Copy the result
/// back through the column view so only the selected entries are updated.
private void optimizeStridedColumn(ref Optimizer opt)
{
    double[6] entries = [10, 5, 20, 30, 6, 40];
    auto matrix = entries[].sliced(2, 3);
    auto column = matrix[0 .. $, 1];

    double[2] work;
    foreach (i; 0 .. column.length)
        work[i] = column[i];
    assert(work[] == [5.0, 6.0]);
    opt.optimizeInPlace(work[]);
    foreach (i; 0 .. column.length)
        column[i] = work[i];

    assert(abs(matrix[0, 1] - 1) < 1e-7 && abs(matrix[1, 1] + 2) < 1e-7);
    assert(matrix[0, 0] == 10 && matrix[0, 2] == 20);
    assert(matrix[1, 0] == 30 && matrix[1, 2] == 40);
    writeln("Updated rows: ", entries[0 .. 3], " ", entries[3 .. 6]);
}

// Compile-time checks for views that cannot be passed directly.
private void checkUnsupportedViews(ref Optimizer opt, double[] storage)
{
    const(double)[] readOnly = storage;
    auto point = storage.sliced;
    auto matrix = storage.sliced(2, 3);
    static assert(!__traits(compiles, opt.optimizeInPlace(readOnly.sliced)));
    static assert(!__traits(compiles, opt.optimizeInPlace(point.map!"a * 2")));
    static assert(!__traits(compiles, opt.optimizeInPlace(point.universal)));
    static assert(!__traits(compiles, opt.optimizeInPlace(matrix[0 .. $, 0 .. 2])));
    static assert(!__traits(compiles, opt.optimizeInPlace(matrix[0 .. $, 1])));
}
