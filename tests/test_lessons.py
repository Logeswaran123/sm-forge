from smforge import lessons


def test_ledger(tmp_path):
    path = lessons.ledger_path(tmp_path, "NVIDIA GeForce RTX 5080", "p")
    assert path == tmp_path / "lessons" / "nvidia_geforce_rtx_5080" / "p.md"
    assert lessons.read(path) is None
    lessons.start(path, "pipeline")
    assert lessons.read(path) is None
    lessons.add(path, "autotuning attempt 0: parity 1.01x over its input:  bigger\n tiles")
    lessons.start(path, "trials")
    for i in range(3):
        lessons.add(path, f"t{i + 1} (from t0): accepted{lessons.speedup(1.5 + i, 't0')}: s{i}")
    text = path.read_text()
    assert text.startswith("# p\n") and text.count("\n## ") == 2
    assert "- autotuning attempt 0: parity 1.01x over its input: bigger tiles\n" in text
    full = lessons.read(path)
    assert full.count("## ") == 2 and full.endswith("- t3 (from t0): accepted 3.50x over t0: s2")
    recent = lessons.read(path, lines=2)
    assert recent.count("## ") == 1 and recent.count("\n- ") == 2 and "t1 " not in recent
