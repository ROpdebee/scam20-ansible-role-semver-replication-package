"""Structural model provenance."""
from typing import ClassVar, Optional, Sequence

import abc

from pathlib import Path
from pprint import pformat as pformat  # re-export  # noqa

import graphviz as gv


class GraphvizMixin:

    _gv_color: ClassVar[str]
    _gv_shape: ClassVar[str]

    def __init_subclass__(
            cls, *args: object,
            gv_color: Optional[str] = None,
            gv_shape: Optional[str] = None,
            **kwargs: object
    ) -> None:
        cls._gv_color = gv_color or 'black'
        cls._gv_shape = gv_shape or 'rect'

    @abc.abstractmethod
    def gv_visit(self, graph: 'SMGraph') -> None:
        """Visit the object and dump it and its children to a graph."""
        ...

    def gv_visit_child(self, graph: 'SMGraph', attr_name: str) -> None:
        child = getattr(self, attr_name)
        child.gv_visit(graph)
        graph.add_edge(self, child, label=attr_name)

    def gv_visit_children(
            self, graph: 'SMGraph',
            attr_name: str,
            children: Optional[Sequence['GraphvizMixin']] = None,
    ) -> None:
        if children is None:
            children = getattr(self, attr_name)
        for child_pos, child in enumerate(children):
            child.gv_visit(graph)
            graph.add_edge(self, child, label=f'{attr_name}[{child_pos}]')

    def dump_to_dot(
            self, dot_path: Path, format: 'gv.backend._FormatValue'
    ) -> Path:
        print(str(dot_path))
        g = SMGraph(filename=str(dot_path), format=format)
        self.gv_visit(g)
        return Path(g.render())


class SMGraph(gv.Digraph):
    """Custom Digraph for structural model."""

    def add_node(self, obj: GraphvizMixin, label: str) -> None:
        lbl = f'{obj.__class__.__name__}:{label}'
        self.node(
                str(id(obj)), label=lbl, shape=obj._gv_shape,
                color=obj._gv_color)

    def add_edge(
            self, parent: GraphvizMixin, child: GraphvizMixin, label: str
    ) -> None:
        self.edge(str(id(parent)), str(id(child)), label=label)
