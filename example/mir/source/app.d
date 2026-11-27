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
