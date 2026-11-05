# Legal handwritten plans

`t0_1.json` through `t8_2.json` use the exact node ordering, IDs, operator types,
names, and edges of the corresponding `data/patent/handwritten/` plans.
The legal operator instructions are preserved. The t6 dependency correction below
is an intentional exception to the original patent topology.
The identical t1 and t5 variant graphs follow the patent references intentionally.

The same source-column substitution used for patent applies to both legal
variants, because the `_2` graphs place generated records above the semantic join:

- t7 node 1 joins `orders_text` to `reasons_text`, instead of `relief_requirements`.
- t8 node 3 joins `orders_text` to `reasons_text`, instead of the two generated contracts.

The existing legal join conditions are preserved. Both variants have identical
operator parameters, so their differences are graph placement only. Their joins
use source excerpts rather than the base plans' compact generated records, and
are subject to the engine's 2,000-character limit per join side.

All 18 plans were checked for exact reference topology, single-root connectivity,
cycles, source-column availability, projection dependencies, and config coverage.
No model execution or semantic equivalence measurement was performed.
See `config/legal/README.md` for configuration behavior and limitations.

## t6 dependency correction

Node 5 now filters `reasoning_kind`, which node 1 produces from `reasons_text`.
The base t6 and t6_1 graphs therefore place node 5 immediately above node 1 on
the reasons branch; it cannot run on the orders branch. The t6_2 graph already
places node 5 after the required projection and join, so it is unchanged.
This preserves the updated predicate rather than reverting it to an orders filter.
