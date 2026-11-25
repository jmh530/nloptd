import nloptd;
import std.stdio : writeln;

void main()
{
    double target = 2;
    auto opt = Optimizer(Algorithm.ldSLSQP, 1);
    opt.setMinObjective((x, gradient) {
        const delta = x[0] - target;
        if (gradient.length) gradient[0] = 2 * delta;
        return delta * delta;
    });
    opt.setXTolRel(1e-9);
    opt.setMaxEvaluations(200);
    double[1] point = [0.25];
    auto result = opt.optimizeInPlace(point[]);
    writeln(point[], " ", result);
}
