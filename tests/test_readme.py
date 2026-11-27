"""M7 checks (laptop): README blocks match the results files, local links resolve, videos are small."""
import re
from pathlib import Path

from scripts.make_readme import README, TABLE, headline, section
from scripts.make_results_table import load_rows


def block(name):
    return re.search(rf"<!-- BEGIN {name} -->\n(.*?)\n<!-- END {name} -->", README.read_text(), re.S).group(1)


def test_tables_match_ablation_table():
    t = TABLE.read_text()
    assert block("per_config") == section(t, "Per config (seeds pooled)")
    assert block("per_model") == section(t, "Per model")
    assert block("per_task") in t


def test_headline_matches_csvs():
    assert block("headline") == headline(load_rows())


def test_local_links_resolve():
    links = re.findall(r"\]\(([^)#]+)\)", README.read_text())
    missing = [l for l in links if not l.startswith("http") and not Path(l).exists()]
    assert links and not missing, missing


def test_videos_small():
    gifs = re.findall(r"\((docs/media/[^)]+\.gif)\)", README.read_text())
    assert len(gifs) == 3 and all(Path(g).stat().st_size < 2_000_000 for g in gifs)
