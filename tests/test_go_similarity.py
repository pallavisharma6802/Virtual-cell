import gzip

from scripts.go_similarity import jaccard, load_go_bp_annotations, top_k_neighbors


def _write_gaf(tmp_path, rows):
    path = tmp_path / "test.gaf.gz"
    header = "!gaf-version: 2.2\n"
    lines = []
    for symbol, go_id, aspect in rows:
        cols = ["UniProtKB", "P00001", symbol, "", go_id, "PMID:1", "IEA", "", aspect,
                "", "", "protein", "taxon:9606", "20260101", "GO_Central"]
        lines.append("\t".join(cols))
    path.write_bytes(gzip.compress((header + "\n".join(lines) + "\n").encode()))
    return path


def test_load_go_bp_annotations_filters_to_biological_process(tmp_path):
    path = _write_gaf(tmp_path, [
        ("GENE_A", "GO:0001", "P"),
        ("GENE_A", "GO:0002", "F"),
        ("GENE_B", "GO:0001", "P"),
    ])
    gene_to_go = load_go_bp_annotations(path)
    assert gene_to_go["GENE_A"] == {"GO:0001"}
    assert gene_to_go["GENE_B"] == {"GO:0001"}


def test_jaccard_similarity():
    assert jaccard({"GO:1", "GO:2"}, {"GO:2", "GO:3"}) == 1 / 3
    assert jaccard(set(), {"GO:1"}) == 0.0
    assert jaccard({"GO:1"}, {"GO:2"}) == 0.0


def test_top_k_neighbors_excludes_self_and_zero_similarity():
    gene_to_go = {
        "QUERY": {"GO:1", "GO:2"},
        "CLOSE": {"GO:1", "GO:2", "GO:3"},
        "FAR": {"GO:9"},
        "UNANNOTATED": set(),
    }
    neighbors = top_k_neighbors("QUERY", ["CLOSE", "FAR", "UNANNOTATED", "QUERY"], gene_to_go, k=5)
    assert neighbors == [("CLOSE", 2 / 3)]


def test_top_k_neighbors_returns_empty_for_unannotated_query():
    gene_to_go = {"CLOSE": {"GO:1"}}
    assert top_k_neighbors("QUERY", ["CLOSE"], gene_to_go, k=5) == []
