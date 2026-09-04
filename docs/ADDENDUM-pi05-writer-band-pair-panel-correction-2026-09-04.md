# Addendum: position-matched Object panel correction

**Frozen:** 2026-09-04, before inspecting any action metric or internal-state metric from the interrupted confirmation attempt  
**Scope:** input-validity correction only

The `v2` execution stopped at directed pair `task 2 → task 4` because its two
prompts have different valid instruction positions. The run had already written
70 rows for `task 0 ↔ task 2`; those row-level results were not inspected and the
incomplete artifact remains preserved.

A behavior-blind audit then tokenized every LIBERO Object task and checked object
co-presence in both simulator directions. It did not call the prefix transformer,
the action expert, or the policy action sampler. The complete set of eligible
unordered pairs was:

- `task 0 ↔ task 1`;
- `task 0 ↔ task 2`;
- `task 3 ↔ task 8`;
- `task 8 ↔ task 9`.

The corrected confirmation panel is exactly these four pairs in both directions.
This replaces the invalid `task 2 ↔ task 4` and `task 3 ↔ task 4` relations with
the two other pairs returned by the exhaustive metadata rule. It preserves eight
directed cells, five initial states, layers 6–8, control layers 14–16, all metrics,
and every frozen success threshold. The corrected run starts from an empty output
directory; no rows from the interrupted attempt are reused.

The audit record is `artifacts/pi05_object_pair_metadata_2026-09-04.json` and the
corrected pair list is
`configs/pi05_attention_resolution_holdout_pairs_v2.json`.
