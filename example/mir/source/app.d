import mir.ndslice.slice : sliced;
import nloptd;
import std.math : abs;
import std.stdio : writeln;

void main()
{
    auto opt = Optimizer(Algorithm.ldSLSQP, 2);
    opt.setMinObjective((coordinates, gradient) {
        // Both Mir views borrow NLopt's buffers for this callback.
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
    optimizeStridedColumn(opt);
}

private void optimizeContiguousPoint(ref Optimizer opt)
{
    double[2] storage = [5, 6];
    auto point = storage[].sliced;
    assert(point.field is storage[]);
    opt.optimizeInPlace(point.field);
    assert(abs(point[0] - 1) < 1e-7 && abs(point[1] + 2) < 1e-7);
    writeln("Contiguous point: ", storage[]);
}

private void optimizeStridedColumn(ref Optimizer opt)
{
    double[6] entries = [10, 5, 20, 30, 6, 40];
    auto matrix = entries[].sliced(2, 3);
    auto column = matrix[0 .. $, 1];

    // Gather the middle column into contiguous storage for NLopt.
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
