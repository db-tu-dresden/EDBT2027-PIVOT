import os

from pivot.ir.graph import Graph
from pivot.lang.languages import get_language
from pivot.utils.corpus_cache import load_corpus_per_file


# Bump whenever the pickled Graph/Node fields change.
_CACHE_VERSION = "6"
_CACHE_FILENAME = ".graph_cache.pkl"


class GraphLoader:
    def __init__(self):
        """Initialize graph loader state from the active language manifest."""
        self.base_dir: str = get_language().graphs_yaml_dir
        self.graphs: list[Graph] = []
        self._is_loaded = False

    def load_all_graphs(self):
        """Load every pattern graph under the base directory, cached per file.
        Files keep their ``os.walk`` order, on which the occurrence numbering
        depends.  ``PIVOT_NO_GRAPH_CACHE`` bypasses the cache."""
        if self._is_loaded:
            return
        per_file = load_corpus_per_file(
            self.base_dir,
            cache_filename=_CACHE_FILENAME,
            version=_CACHE_VERSION,
            parse_file=lambda path, _name: Graph.load_all_from_yaml(path),
            enabled=not os.environ.get("PIVOT_NO_GRAPH_CACHE"),
        )
        self.graphs = [graph for _rel, graphs in per_file for graph in graphs]
        self._is_loaded = True

    def get_by_source(self, source: str, *, arch: str | None = None, nto1: bool = True) -> list[Graph]:
        """The graphs of source family ``source``, without those that block
        ``arch`` and, unless ``nto1``, the n:1 ones."""
        return [
            graph for graph in self.graphs
            if graph.isa.lower() == source.lower()
            and not (arch and arch in {a.lower() for a in graph.blocked_arches})
            and (nto1 or not graph.nto1)
        ]
