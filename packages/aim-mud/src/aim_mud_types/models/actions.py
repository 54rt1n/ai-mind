# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD action model for agent commands."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field


class MUDAction(BaseModel):
    """An action the agent wants to take in the MUD.

    Actions are emitted by the agent worker to the mud:actions stream
    and consumed by Evennia to execute commands in the world.

    The to_command() method converts the structured action into an
    Evennia command string for execution.

    Attributes:
        tool: Tool name (e.g., "say", "emote", "move").
        args: Tool arguments as a dictionary.
        metadata: Metadata dict for action flags (non_published, non_reactive, etc.).
        priority: Execution priority (lower = first). Default 5.
    """

    # Metadata key constants for action flags
    META_NON_PUBLISHED: ClassVar[str] = "non_published"  # Event won't be routed, no echo expected
    META_NON_REACTIVE: ClassVar[str] = "non_reactive"    # Event won't trigger reactions from others
    META_SLEEP_AWARE: ClassVar[str] = "sleep_aware"      # Event bypasses sleep filtering

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5

    def expects_echo(self) -> bool:
        """Return True if this action expects an echo event from the MUD.

        NON_PUBLISHED actions don't produce routed events, so no echo is expected.
        """
        return not self.metadata.get(self.META_NON_PUBLISHED, False)

    def to_command(self) -> str:
        """Convert action to Evennia command string.

        Handles all player-level and builder-level commands defined
        in the ANDIMUD specification.

        Returns:
            Command string ready for execution in Evennia.
        """
        # Player-level commands
        if self.tool == "say":
            return f"say {self.args.get('message', '')}"

        elif self.tool == "emote":
            return f"emote {self.args.get('action', '')}"

        elif self.tool == "whisper":
            target = self.args.get("target", "")
            message = self.args.get("message", "")
            return f"whisper {target} = {message}"

        elif self.tool == "pose":
            return f"pose {self.args.get('action', '')}".strip()

        elif self.tool == "home":
            return "home"

        elif self.tool == "setdesc":
            if "description" not in self.args:
                return ""
            return f"setdesc {self.args.get('description', '')}".strip()

        elif self.tool == "desc_room":
            description = self.args.get("description")
            if not description:
                return ""
            return f"@desc here = {description}".strip()

        elif self.tool == "desc_object":
            target = self.args.get("target")
            description = self.args.get("description")
            if not target or not description:
                return ""
            return f"@desc {target} = {description}".strip()

        elif self.tool == "move":
            location = self.args.get("location")
            if not location:
                location = self.args.get("direction", "")
            if not location:
                return ""

            mood = self.args.get("mood", "").strip()
            if mood:
                return f"{location}, {mood}"
            return location

        elif self.tool == "get":
            obj = self.args.get('object', '')
            mood = self.args.get("mood", "").strip()
            if mood:
                return f"get {obj}, {mood}"
            return f"get {obj}"

        elif self.tool == "drop":
            obj = self.args.get('object', '')
            mood = self.args.get("mood", "").strip()
            if mood:
                return f"drop {obj}, {mood}"
            return f"drop {obj}"

        elif self.tool == "give":
            obj = self.args.get("object", "")
            target = self.args.get("target", "")
            mood = self.args.get("mood", "").strip()
            if mood:
                return f"give {obj} = {target}, {mood}"
            return f"give {obj} = {target}"

        elif self.tool == "use":
            return f"use {self.args.get('object', '')}"

        elif self.tool == "ring":
            obj = self.args.get("object", "")
            return f"ring {obj}".strip()

        elif self.tool == "speak":
            text = self.args.get("text", "")
            if not text:
                return ""
            # Encode newlines to avoid command parser splitting
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            encoded = normalized.replace("\n", "\\n")
            return f"act {encoded}".rstrip()

        # Builder-level commands (Andi only)
        elif self.tool == "dig":
            if "room" not in self.args or "exits" not in self.args:
                return ""
            return f"@dig {self.args.get('room', '')} = {self.args.get('exits', '')}".strip()

        elif self.tool == "tunnel":
            if "direction" not in self.args or "room" not in self.args:
                return ""
            return f"@tunnel {self.args.get('direction', '')} = {self.args.get('room', '')}".strip()

        elif self.tool == "open":
            if "name" not in self.args or "destination" not in self.args:
                return ""
            return f"@open {self.args.get('name', '')} = {self.args.get('destination', '')}".strip()

        elif self.tool == "alias":
            if "object" not in self.args or "aliases" not in self.args:
                return ""
            delete = self.args.get("delete", False)
            suffix = "/delete" if delete else ""
            return f"@alias{suffix} {self.args.get('object', '')} = {self.args.get('aliases', '')}".strip()

        elif self.tool == "copy" and "source" in self.args and "destination" in self.args:
            # Builder @copy command
            return f"@copy {self.args.get('source', '')} = {self.args.get('destination', '')}".strip()

        elif self.tool == "cpattr":
            if (
                "source" not in self.args
                or "source_attr" not in self.args
                or "target" not in self.args
                or "target_attr" not in self.args
            ):
                return ""
            return (
                f"@cpattr {self.args.get('source', '')}/"
                f"{self.args.get('source_attr', '')} = "
                f"{self.args.get('target', '')}/"
                f"{self.args.get('target_attr', '')}"
            ).strip()

        elif self.tool == "mvattr":
            if (
                "source" not in self.args
                or "source_attr" not in self.args
                or "target" not in self.args
                or "target_attr" not in self.args
            ):
                return ""
            return (
                f"@mvattr {self.args.get('source', '')}/"
                f"{self.args.get('source_attr', '')} = "
                f"{self.args.get('target', '')}/"
                f"{self.args.get('target_attr', '')}"
            ).strip()

        elif self.tool == "create":
            if "object" not in self.args:
                return ""
            return f"@create {self.args.get('object', '')}".strip()

        elif self.tool == "desc":
            if "target" not in self.args or "description" not in self.args:
                return ""
            return f"@desc {self.args.get('target', '')} = {self.args.get('description', '')}".strip()

        elif self.tool == "teleport":
            if "destination" not in self.args:
                return ""
            target = self.args.get("target", "me")
            return f"@teleport {target} = {self.args.get('destination', '')}".strip()

        elif self.tool == "destroy":
            if "object" not in self.args:
                return ""
            return f"@destroy {self.args.get('object', '')}".strip()

        elif self.tool == "link":
            if "exit" not in self.args or "destination" not in self.args:
                return ""
            return f"@link {self.args.get('exit', '')} = {self.args.get('destination', '')}".strip()

        elif self.tool == "unlink":
            if "exit" not in self.args:
                return ""
            return f"unlink {self.args.get('exit', '')}".strip()

        elif self.tool == "lock":
            if "object" not in self.args or "lockstring" not in self.args:
                return ""
            return f"@lock {self.args.get('object', '')} = {self.args.get('lockstring', '')}".strip()

        elif self.tool == "name":
            if "object" not in self.args or "new_name" not in self.args:
                return ""
            return f"@name {self.args.get('object', '')} = {self.args.get('new_name', '')}".strip()

        elif self.tool == "set":
            if (
                "object" not in self.args
                or "attribute" not in self.args
                or "value" not in self.args
            ):
                return ""
            return (
                f"@set {self.args.get('object', '')}/"
                f"{self.args.get('attribute', '')} = "
                f"{self.args.get('value', '')}"
            ).strip()

        elif self.tool == "sethome":
            if "object" not in self.args or "home" not in self.args:
                return ""
            return f"@sethome {self.args.get('object', '')} = {self.args.get('home', '')}".strip()

        elif self.tool == "cmdsets":
            if "target" not in self.args:
                return ""
            return f"@cmdsets {self.args.get('target', '')}".strip()

        elif self.tool == "typeclass":
            if "target" not in self.args or "typeclass" not in self.args:
                return ""
            return f"@typeclass {self.args.get('target', '')} = {self.args.get('typeclass', '')}".strip()

        elif self.tool == "wipe":
            if "object" not in self.args:
                return ""
            attribute = self.args.get("attribute")
            if attribute:
                return f"@wipe {self.args.get('object', '')}/{attribute}".strip()
            return f"@wipe {self.args.get('object', '')}".strip()

        elif self.tool == "examine":
            if "target" not in self.args:
                return ""
            return f"@examine {self.args.get('target', '')}".strip()

        elif self.tool == "find":
            if "query" not in self.args:
                return ""
            return f"@find {self.args.get('query', '')}".strip()

        elif self.tool == "tag":
            if "object" not in self.args or "tags" not in self.args:
                return ""
            category = self.args.get("category")
            delete = self.args.get("delete", False)
            suffix = "/del" if delete else ""
            tag_str = self.args.get("tags", "")
            if category:
                tag_str = f"{tag_str}:{category}"
            return f"@tag{suffix} {self.args.get('object', '')} = {tag_str}".strip()

        elif self.tool == "spawn":
            if "prototype" not in self.args:
                return ""
            return f"@spawn {self.args.get('prototype', '')}".strip()

        # Aura tool commands - agent uses same commands as users
        elif self.tool == "sync":
            return "sync"

        elif self.tool == "scan":
            pattern = self.args.get("pattern", "")
            return f"scan {pattern}".strip()

        elif self.tool == "grep":
            pattern = self.args.get("pattern", "")
            return f"grep {pattern}".strip()

        elif self.tool == "cat":
            file = self.args.get("file", "")
            return f"cat {file}".strip()

        elif self.tool == "sed":
            file = self.args.get("file", "")
            range_expr = self.args.get("range", "")
            if not file or not range_expr:
                return ""
            return f"sed {file} = {range_expr}".strip()

        elif self.tool == "head":
            file = self.args.get("file", "")
            lines = self.args.get("lines", "")
            if not file or lines == "":
                return ""
            return f"head {file} = {lines}".strip()

        elif self.tool == "tail":
            file = self.args.get("file", "")
            lines = self.args.get("lines", "")
            if not file or lines == "":
                return ""
            return f"tail {file} = {lines}".strip()

        elif self.tool == "meta":
            file = self.args.get("file", "")
            return f"meta {file}".strip()

        elif self.tool == "refresh":
            target = self.args.get("target", "")
            return f"refresh {target}".strip()

        elif self.tool == "diff":
            file = self.args.get("file", "")
            other = self.args.get("other", "")
            if not file or not other:
                return ""
            return f"diff {file} = {other}".strip()

        elif self.tool == "gitdiff":
            file = self.args.get("file", "")
            ref = self.args.get("ref", "")
            if not file:
                return ""
            if ref:
                return f"gitdiff {file} = {ref}".strip()
            return f"gitdiff {file}".strip()

        elif self.tool == "clone":
            file = self.args.get("file", "")
            new_file = self.args.get("new_file", "")
            if not file or not new_file:
                return ""
            return f"clone {file} = {new_file}".strip()

        elif self.tool == "patch":
            file = self.args.get("file", "")
            if not file:
                return ""
            return f"patch {file}".strip()

        elif self.tool == "touch":
            path = self.args.get("path", "")
            return f"touch {path}".strip()

        elif self.tool == "uv":
            args = self.args.get("args", "")
            return f"uv {args}".strip()

        elif self.tool == "py_exec":
            code = self.args.get("code", "")
            escaped = code.replace("\n", "\\n")
            return f"py_exec {escaped}"

        elif self.tool == "bash_exec":
            command = self.args.get("command", "")
            return f"bash_exec {command}"

        elif self.tool == "web_search":
            query = self.args.get("query", "")
            return f"web_search {query}"

        elif self.tool == "visit_webpage":
            url = self.args.get("url", "")
            return f"visit_webpage {url}"

        elif self.tool == "get_feed":
            return "get_feed"

        elif self.tool == "research":
            query = self.args.get("query", "")
            return f"research {query}"

        elif self.tool == "read_doc":
            doc_id = self.args.get("doc_id", "")
            return f"read_doc {doc_id}"

        elif self.tool == "show_list":
            return "show_list"

        elif self.tool == "add_item":
            text = self.args.get("text", "")
            return f"add_item {text}"

        elif self.tool == "check_item":
            item_id = self.args.get("item_id", "")
            return f"check_item {item_id}"

        elif self.tool == "stock_quote":
            symbol = self.args.get("symbol", "")
            return f"stock_quote {symbol}"

        # Paper/book commands
        elif self.tool == "read":
            book = self.args.get("book", "")
            page = self.args.get("page")
            if page is not None:
                return f"read {book} = {page}"
            return f"read {book}"

        elif self.tool == "unbind":
            book = self.args.get("book", "")
            return f"unbind {book}"

        elif self.tool == "write":
            paper = self.args.get("paper", "")
            content = self.args.get("content", "")
            return f"write {paper} = {content}"

        elif self.tool == "bind":
            folder = self.args.get("folder", "")
            title = self.args.get("title")
            if title:
                return f'bind {folder} as "{title}"'
            return f"bind {folder}"

        elif self.tool == "print":
            terminal_type = self.args.get("terminal_type", "")
            identifier = self.args.get("identifier", "")
            name = self.args.get("name", "")
            return f"print {terminal_type} = {identifier}, {name}"

        elif self.tool == "scan":
            terminal_type = self.args.get("terminal_type", "")
            filename = self.args.get("filename", "")
            return f"scan {terminal_type} = {filename}"

        elif self.tool == "copy":
            # Paper copy command - no args, uses paper in copier
            return "copy"

        # Web tools (container-mcp)
        elif self.tool == "web_scrape":
            url = self.args.get("url", "")
            selector = self.args.get("selector")
            if selector:
                return f"web_scrape {url} {selector}"
            return f"web_scrape {url}"

        elif self.tool == "web_browse":
            url = self.args.get("url", "")
            return f"web_browse {url}"

        else:
            # Generic fallback for unknown aura tools
            # Format: tool_name arg1=val1 arg2=val2
            if self.args:
                args_str = " ".join(f"{k}={v}" for k, v in self.args.items())
                return f"{self.tool} {args_str}".strip()
            return self.tool

    def to_redis_dict(self, agent_id: str) -> dict[str, Any]:
        """Convert to dictionary for Redis stream publishing.

        Args:
            agent_id: ID of the agent emitting this action.

        Returns:
            Dictionary ready for JSON serialization to Redis.
        """
        return {
            "agent_id": agent_id,
            "command": self.to_command(),
            "tool": self.tool,
            "args": self.args,
            "metadata": self.metadata,
            "priority": self.priority,
        }
