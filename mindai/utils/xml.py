# mindai/utils/xml.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from dataclasses import dataclass
import logging
import re
logger = logging.getLogger(__name__)

@dataclass
class XmlLeaf:
    """
    A simple XML leaf node.
    """
    depth: int
    name: str
    content: str
    attrs: dict
    nowrap: bool

    @property
    def format_attrs(self) -> str:
        """
        Format a dictionary of attributes into a string of XML attributes.

        Args:
            attrs (dict): Dictionary of attribute names and values

        Returns:
            str: Formatted XML attributes string
        """
        if not self.attrs:
            return ""
        return " " + " ".join(f'{k}="{v}"' for k, v in self.attrs.items())


@dataclass
class XmlNode:
    """
    A simple XML tree node.
    """
    name: str
    leaves: list[XmlLeaf]
    children: dict[str, "XmlNode"]
    depth: int = 0

    def add_leaf(self, name: str, content: str, attrs: dict = {}, nowrap: bool = False) -> None:
        """
        Add a leaf node to the current node.
        """
        self.leaves.append(XmlLeaf(depth=self.depth, name=name, content=content, attrs=attrs, nowrap=nowrap))

    def add_child(self, name: str) -> "XmlNode":
        """
        Add a child node to the current node.
        """
        if name not in self.children:
            self.children[name] = XmlNode(depth=self.depth + 1, name=name, leaves=[], children={})
            #logger.info(f"Added child node {name} to {self.name}")
        return self.children[name]
        
    def render(self, indent_str: str = "  ") -> list[str]:
        """
        Render the XML node and its children to a string.
        """
        result = []
        if self.depth == 0:
            for child_name, child in self.children.items():
                result.extend(child.render(indent_str))
            return result
        
        indent = indent_str * self.depth
        if len(self.leaves) == 0 and len(self.children) == 0:
            result.append(f"{indent}<{self.name}/>")
            return result
        if len(self.leaves) == 1 and len(self.children) == 0:
            result.append(f"{indent}<{self.name}{self.leaves[0].format_attrs}>{self.leaves[0].content}</{self.name}>")
            return result

        close_tag = False

        if len(self.leaves) <= 1:
            attr = self.leaves[0].format_attrs if self.leaves else ""
            content = ("\n" + self.leaves[0].content) if self.leaves else ""
            result.append(f"{indent}<{self.name}{attr}>{content}")
            close_tag = True
        else:
            for leaf in self.leaves:
                attrs = leaf.format_attrs
                if leaf.nowrap:
                    result.append(f"{indent}<{leaf.name}{attrs}>{leaf.content}</{leaf.name}>")
                else:
                    result.append(f"{indent}<{leaf.name}{attrs}>\n{leaf.content}\n{indent}</{leaf.name}>")
        
        for child_name, child in self.children.items():
            result.extend(child.render(indent_str))
        
        if close_tag:
            result.append(f"{indent}</{self.name}>")

        return result

class XmlRoot:
    """
    The root node of an XML document.
    """
    root: XmlNode

    def __init__(self):
        self.root = XmlNode(depth=-1, name="root", leaves=[], children={})

    def drill(self, path: list[str]) -> XmlNode | None:
        """
        Drill down into the tree structure based on the given path.
        """
        current_node = self.root
        for part in path:
            if part not in current_node.children:
                #logger.warning(f"Path {part} not found in current node {current_node.name}")
                current_node.add_child(part)
            current_node = current_node.children[part]

        return current_node

    def render(self) -> str:
        """
        Render the XML document to a string.
        """
        return "\n".join(self.root.render())

class XmlFormatter:
    """
    A flexible XML document formatter that tracks total content length.

    This formatter supports building nested XML structures with attributes while
    maintaining an accurate count of the total content length. The formatter
    uses a tree-based structure to maintain hierarchy and supports arbitrary
    nesting depths.

    Attributes:
        tree (dict): Internal representation of the XML document structure
        indent (str): String used for each level of indentation
        _current_length (int): Running total of content length in characters
    """

    def __init__(self, base_indent: str = "  "):
        """
        Initialize a new XML formatter.

        Args:
            base_indent (str): String to use for each level of indentation
                             Defaults to 4 spaces.
        """
        self.tree = XmlRoot()
        self.indent = base_indent
        self._current_length = 0

    @property
    def current_length(self) -> int:
        """
        Total length of all content and attributes in characters.

        Returns:
            int: Current length of all content in the document
        """
        return self._current_length

    def add_element(self, *path: str, content: str = None, nowrap: bool = False, **attrs):
        """
        Add an element with content and optional attributes at the specified path.

        Tracks the length of added content and attributes in current_length.

        Args:
            *path (str): Variable length path to the element location
            content (str): The text content of the element (optional)
            **attrs: Optional attributes for the element as keyword arguments
        """
        if not path:
            raise ValueError("Path must contain at least one element name")

        current = self.tree.drill(path)

        # Create or update the final element
        last = path[-1]

        # Update content and attributes
        if content is not None:
            # Add new content length
            current.add_leaf(name=last, content=content, attrs=attrs, nowrap=nowrap)
            new_attrs_length = sum(len(str(k)) + len(str(v)) + 4 for k, v in attrs.items())
            self._current_length += len(str(content)) + new_attrs_length

        logger.info(
            f"Added element {'/'.join(path)} "
            f"with content length {len(str(content)) if content else 0} "
            f"(total {self._current_length})"
        )

    def substitute(self, match_regex: str, replacement: str) -> None:
        """
        Substitute all occurrences of a regex pattern with a replacement string.
        """
        for leaf in self.tree.root.leaves:
            if leaf.content:
                leaf.content = re.sub(match_regex, replacement, leaf.content)

    def render(self) -> str:
        """
        Render the complete XML document to a string.

        Returns:
            str: The formatted XML document as a string with proper indentation
        """
        return self.tree.render()