import solution


def test_official_solution_entry_reexports_all_public_interfaces():
    for name in ("prepare", "run", "run_direct", "run_prepared", "run_prepared_direct"):
        assert callable(getattr(solution, name))
