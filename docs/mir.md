# Using Mir with nloptd

For background on how nloptd passes points and derivatives to D functions, see [Objective and constraint callbacks in NLopt and D](callbacks.md).

## From an ndslice to the solver

Mir describes the shape and strides of a view as well as its elements. The strides determine whether the selected elements occupy consecutive memory in the order the solver expects.

For a `Contiguous` ndslice whose `.field` is a writable `double[]`, set the optimizer dimension to the total element count. Either call below operates on that storage.

| Pass the ndslice | Pass its backing array |
| --- | --- |
| `opt.optimizeInPlace(point)` | `opt.optimizeInPlace(point.field)` |
| Direct overload on the development branch | Also available in nloptd 0.3.0 |

For a view into a larger array, use the layout of the selected view:

- A row of a `Canonical` matrix is contiguous. Pass `matrix[i]` directly, including when there is padding between rows.
- A `Universal` vector over mutable doubles can use the same path once you establish that its stride is `+1`. Then pass `point.assumeContiguous`.
- For a strided column, copy its values into a contiguous work array, optimize the array, and write the result back to the column. Both copies happen outside the objective.

## Using ndslices in the objective

Use the objective’s D slices directly, or call `.sliced` to obtain Mir views over the same memory. If the input point represents a matrix, use `.sliced(rows, cols)` to restore its shape in the original storage order.

When NLopt requests a gradient, fill it with the partial derivatives in the same order as the input. Writes through a gradient view go directly into NLopt’s array. The slices and any views derived from them are valid only for that evaluation.

## Setup and examples

Include `mir-algorithm` alongside nloptd in your application's build. If you use DUB, run `dub add mir-algorithm` in your application directory.

The [Mir examples](../example/mir/source/app.d) show how to optimize contiguous views and a strided matrix column. To run them with DUB, use `dub run --root=example/mir` from the nloptd repository root.
