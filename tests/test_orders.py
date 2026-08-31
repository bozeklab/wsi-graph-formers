"""The order files and `ordered_files`, which decide the cross-validation folds.

A wrong order does not crash, it produces plausible numbers on the wrong splits,
so `ordered_files` deliberately raises instead of falling back. These tests pin
that behaviour down.
"""

import pytest

from conftest import tracked_matching
from models.order_utils import ordered_files

ORDERS = tracked_matching("configs/orders/*.txt")


@pytest.mark.parametrize("path", ORDERS, ids=lambda p: p.stem)
def test_order_file_is_a_clean_list_of_graphs(path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    entries = [line for line in lines if line]

    assert entries, f"{path.name} is empty"
    assert all(e.endswith(".pt") for e in entries), (
        f"{path.name} has entries that are not .pt files"
    )
    assert not any(e.startswith("._") for e in entries), (
        f"{path.name} lists macOS resource-fork files"
    )

    duplicates = {e for e in entries if entries.count(e) > 1}
    assert not duplicates, f"{path.name} lists {sorted(duplicates)[:3]} more than once"


# --- ordered_files ----------------------------------------------------------

def _folder(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


def test_defaults_to_alphabetical_order(tmp_path):
    _folder(tmp_path, ["b.pt", "a.pt", "c.pt"])
    assert ordered_files(tmp_path) == ["a.pt", "b.pt", "c.pt"]


def test_ignores_other_extensions_and_macos_junk(tmp_path):
    _folder(tmp_path, ["a.pt", "._a.pt", "notes.txt", ".DS_Store"])
    assert ordered_files(tmp_path) == ["a.pt"]


def test_order_file_wins_over_alphabetical(tmp_path):
    _folder(tmp_path, ["a.pt", "b.pt", "c.pt"])
    order = tmp_path / "order.txt"
    order.write_text("c.pt\na.pt\nb.pt\n")
    assert ordered_files(tmp_path, order_file=order) == ["c.pt", "a.pt", "b.pt"]


def test_rejects_an_order_file_with_duplicates(tmp_path):
    _folder(tmp_path, ["a.pt", "b.pt"])
    order = tmp_path / "order.txt"
    order.write_text("a.pt\nb.pt\na.pt\n")
    with pytest.raises(ValueError, match="duplicated"):
        ordered_files(tmp_path, order_file=order)


def test_rejects_an_order_file_naming_an_absent_graph(tmp_path):
    _folder(tmp_path, ["a.pt"])
    order = tmp_path / "order.txt"
    order.write_text("a.pt\nghost.pt\n")
    with pytest.raises(ValueError, match="does not match"):
        ordered_files(tmp_path, order_file=order)


def test_rejects_an_order_file_missing_a_present_graph(tmp_path):
    _folder(tmp_path, ["a.pt", "b.pt"])
    order = tmp_path / "order.txt"
    order.write_text("a.pt\n")
    with pytest.raises(ValueError, match="does not match"):
        ordered_files(tmp_path, order_file=order)


def test_returns_every_graph_exactly_once(tmp_path):
    names = [f"wsigraph_{i}.pt" for i in range(10)]
    _folder(tmp_path, names)
    assert sorted(ordered_files(tmp_path)) == sorted(names)
