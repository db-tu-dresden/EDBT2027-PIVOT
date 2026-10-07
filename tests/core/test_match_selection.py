from pivot.passes.translation.match_selection import MatchCandidate, select_matches


def _match(
    match_id: str, intrinsics: str, cost: int = 1, constants: int = 0, order: int = 0,
    recomputed: int = 0, result: str | None = None, inputs: str = "",
) -> MatchCandidate:
    return MatchCandidate(
        match_id=match_id,
        intrinsics=frozenset(intrinsics),
        cost=cost,
        constants=constants,
        order_key=(order, match_id),
        recomputed=recomputed,
        removed=frozenset(intrinsics) - {result or intrinsics[-1]},
        inputs=frozenset(inputs),
    )


def _selected(*candidates: MatchCandidate) -> set[str]:
    selected, _ = select_matches(list(candidates))
    return set(selected)


def test_exact_cover_beats_greedy_on_the_forced_pivot():
    # X is covered only by A, yet B translates more intrinsics.
    assert _selected(_match("A", "XY"), _match("B", "YZW")) == {"B"}


def test_coverage_outranks_statement_count():
    assert _selected(_match("big", "XYZ", cost=5), _match("small", "XY", cost=1)) == {"big"}
    assert _selected(_match("big", "XYZ", cost=5), _match("small", "XY", cost=1), _match("z", "Z")) == {"small", "z"}


def test_fewer_statements_among_equal_coverage():
    assert _selected(_match("fused", "XY", cost=1), _match("x", "X"), _match("y", "Y")) == {"fused"}


def test_constants_break_a_statement_tie():
    # _mm_cmp_ps(a, b, _CMP_LT_OQ): VLT fixes the predicate, CMP takes it as input.
    assert _selected(_match("CMP", "X", constants=0), _match("VLT", "X", constants=1)) == {"VLT"}


def test_constants_never_outrank_statements():
    assert _selected(_match("fixed", "X", cost=2, constants=3), _match("plain", "X", cost=1)) == {"plain"}


def test_remaining_ties_prefer_the_earlier_match():
    assert _selected(_match("second", "X", order=1), _match("first", "X", order=0)) == {"first"}
    assert _selected(_match("first", "X", order=0), _match("second", "X", order=1)) == {"first"}


def test_selected_matches_are_disjoint_and_components_are_independent():
    candidates = [
        _match("A", "XY"), _match("B", "YZW"),
        _match("C", "PQ", cost=1), _match("p", "P"), _match("q", "Q"),
        _match("lone", "L"),
    ]
    selected, stats = select_matches(candidates)
    assert set(selected) == {"B", "C", "lone"}
    assert stats.to_json()["components"] == 3
    assert stats.to_json()["max_component_matches"] == 3
    assert stats.max_matches_per_intrinsic == 2


def test_selection_is_deterministic():
    candidates = [_match(f"m{i}", "XYZ"[i % 3] + "XYZ"[(i + 1) % 3], order=i) for i in range(6)]
    first = select_matches(candidates)[0]
    assert all(select_matches(list(reversed(candidates)))[0] == first for _ in range(5))


def test_recomputing_fusion_beats_one_to_one_when_it_saves_statements():
    # abs (A) and lzcnt (B) stay for the overflow check; the conversion
    # replaces the rest (C, D) and recomputes A and B.
    one_to_one = [_match(n.lower(), n) for n in "ABCD"]
    fused = _match("convert", "CD", recomputed=2, inputs="")
    assert _selected(fused, *one_to_one) == {"convert", "a", "b"}


def test_recomputing_fusion_that_saves_nothing_loses():
    assert _selected(_match("fused", "C", recomputed=2), _match("c", "C")) == {"c"}


def test_recomputation_never_outranks_statements():
    assert _selected(_match("fused", "C", cost=1, recomputed=5), _match("c", "C", cost=2)) == {"fused"}


def test_a_match_must_not_remove_a_call_another_replacement_reads():
    # `m` recomputes a kept call and reads Q; `wide` deletes Q on the way to K.
    m = _match("m", "XY", recomputed=1, inputs="Q")
    wide = _match("wide", "QK")
    assert _selected(m, wide, _match("q", "Q"), _match("k", "K")) == {"m", "q", "k"}


def test_the_result_of_another_match_may_feed_a_replacement():
    m = _match("m", "XY", recomputed=1, inputs="K")
    assert _selected(m, _match("wide", "QK")) == {"m", "wide"}


def test_recomputation_stats():
    selected, stats = select_matches([_match("convert", "CD", recomputed=2), _match("c", "C"), _match("d", "D")])
    assert selected == ["convert"]
    stats = stats.to_json()
    assert (stats["recomputing_candidates"], stats["selected_recomputing_matches"], stats["recomputed_intrinsics"]) == (1, 1, 2)
