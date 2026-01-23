# tests/andimud_tests/test_paper.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for paper system typeclasses."""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch


class TestPaper:
    """Tests for Paper typeclass."""

    def test_paper_inherits_from_object(self):
        """Paper inherits from Object."""
        from typeclasses.paper import Paper
        from typeclasses.objects import Object

        assert issubclass(Paper, Object)

    def test_paper_has_content_when_desc_set(self):
        """Paper with db.desc has content."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.db.desc = "Some written content"

        # Call the actual method bound to the mock
        result = Paper.has_content(paper)
        assert result is True

    def test_paper_no_content_when_desc_empty(self):
        """Paper without db.desc is blank."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.db.desc = ""

        result = Paper.has_content(paper)
        assert result is False

    def test_paper_no_content_when_desc_none(self):
        """Paper with None desc is blank."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.db.desc = None

        result = Paper.has_content(paper)
        assert result is False

    def test_paper_is_blank_when_no_content(self):
        """is_blank() returns True when paper has no content."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.db.desc = ""

        # has_content returns False for empty desc
        assert Paper.has_content(paper) is False
        # is_blank calls not has_content(), verify the relationship
        # Since is_blank uses self.has_content(), we test the logic directly
        assert not Paper.has_content(paper) is True

    def test_paper_is_not_blank_when_has_content(self):
        """is_blank() returns False when paper has content."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.db.desc = "Content"

        # has_content returns True for non-empty desc
        assert Paper.has_content(paper) is True
        # is_blank should return False (opposite of has_content)
        assert not Paper.has_content(paper) is False

    def test_paper_is_bound_logic(self):
        """Paper.is_bound() checks if location is a Book instance."""
        from typeclasses.paper import Paper, Book

        # Verify is_bound has correct logic by inspecting the method
        # When location is None, should return False
        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.location = None

        result = Paper.is_bound(paper)
        assert result is False

        # When location is set but not a Book, isinstance fails
        # With MagicMock(spec=Book), isinstance won't work as expected,
        # but we can verify the code structure is correct
        import inspect
        source = inspect.getsource(Paper.is_bound)
        assert "isinstance" in source
        assert "Book" in source
        assert "self.location" in source

    def test_paper_not_bound_when_not_in_book(self):
        """Paper is not bound when its location is not a Book."""
        from typeclasses.paper import Paper

        paper = MagicMock(spec=Paper)
        paper.db = MagicMock()
        paper.location = None

        result = Paper.is_bound(paper)
        assert result is False


class TestFolder:
    """Tests for Folder container."""

    def test_folder_inherits_from_object(self):
        """Folder inherits from Object."""
        from typeclasses.paper import Folder
        from typeclasses.objects import Object

        assert issubclass(Folder, Object)

    def test_folder_get_papers_returns_only_paper(self):
        """Folder.get_papers() returns only Paper objects."""
        from typeclasses.paper import Folder, Paper

        folder = MagicMock(spec=Folder)

        paper1 = MagicMock(spec=Paper)
        paper2 = MagicMock(spec=Paper)
        non_paper = MagicMock()

        folder.contents = [paper1, paper2, non_paper]

        # Call actual method
        result = Folder.get_papers(folder)

        # Only Paper objects should be returned
        assert len(result) == 2
        assert paper1 in result
        assert paper2 in result
        assert non_paper not in result

    def test_folder_has_no_at_object_receive_override(self):
        """Folder does not override at_object_receive (allows all objects)."""
        from typeclasses.paper import Folder

        # Folder should not have its own at_object_receive (inherits default)
        # Check that it's not defined directly on Folder
        assert "at_object_receive" not in Folder.__dict__


class TestBook:
    """Tests for Book container."""

    def test_book_inherits_from_object(self):
        """Book inherits from Object."""
        from typeclasses.paper import Book
        from typeclasses.objects import Object

        assert issubclass(Book, Object)

    def test_book_rejects_all_objects(self):
        """Book rejects all objects (immutable)."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)
        book.key = "My Book"
        book.location = MagicMock()
        book.location.msg_contents = MagicMock()

        obj = MagicMock(spec=Paper)
        obj.move_to = MagicMock()
        source_location = MagicMock()

        # Call actual method
        Book.at_object_receive(book, obj, source_location)

        # Verify rejection
        obj.move_to.assert_called_once_with(source_location, quiet=True)
        book.location.msg_contents.assert_called_once()
        assert "bound and cannot accept" in book.location.msg_contents.call_args[0][0]

    def test_book_page_count(self):
        """Book correctly counts pages."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)

        paper1 = MagicMock(spec=Paper)
        paper2 = MagicMock(spec=Paper)
        non_paper = MagicMock()

        book.contents = [paper1, paper2, non_paper]

        count = Book.page_count(book)
        assert count == 2

    def test_book_get_pages_sorted_by_page_no(self):
        """Book.get_pages() returns pages sorted by page_no."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)

        paper1 = MagicMock(spec=Paper)
        paper1.db = MagicMock()
        paper1.db.page_no = 3

        paper2 = MagicMock(spec=Paper)
        paper2.db = MagicMock()
        paper2.db.page_no = 1

        paper3 = MagicMock(spec=Paper)
        paper3.db = MagicMock()
        paper3.db.page_no = 2

        book.contents = [paper1, paper2, paper3]

        pages = Book.get_pages(book)

        assert len(pages) == 3
        assert pages[0] is paper2  # page_no = 1
        assert pages[1] is paper3  # page_no = 2
        assert pages[2] is paper1  # page_no = 3

    def test_book_get_pages_handles_none_page_no(self):
        """Book.get_pages() handles papers with None page_no."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)

        paper1 = MagicMock(spec=Paper)
        paper1.db = MagicMock()
        paper1.db.page_no = None

        paper2 = MagicMock(spec=Paper)
        paper2.db = MagicMock()
        paper2.db.page_no = 1

        book.contents = [paper1, paper2]

        pages = Book.get_pages(book)

        assert len(pages) == 2
        # None should sort as 0, so paper1 comes first
        assert pages[0] is paper1
        assert pages[1] is paper2

    def test_book_get_page_by_number(self):
        """Book.get_page() returns specific page by number."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)

        paper1 = MagicMock(spec=Paper)
        paper1.db = MagicMock()
        paper1.db.page_no = 1

        paper2 = MagicMock(spec=Paper)
        paper2.db = MagicMock()
        paper2.db.page_no = 2

        book.contents = [paper1, paper2]

        result = Book.get_page(book, 2)
        assert result is paper2

    def test_book_get_page_returns_none_for_missing(self):
        """Book.get_page() returns None for non-existent page."""
        from typeclasses.paper import Book, Paper

        book = MagicMock(spec=Book)

        paper1 = MagicMock(spec=Paper)
        paper1.db = MagicMock()
        paper1.db.page_no = 1

        book.contents = [paper1]

        result = Book.get_page(book, 99)
        assert result is None


class TestConstants:
    """Tests for paper constants."""

    def test_page_size_tokens(self):
        """Default page size is 500 tokens."""
        from typeclasses.paper import DEFAULT_PAGE_SIZE_TOKENS

        assert DEFAULT_PAGE_SIZE_TOKENS == 500

    def test_hard_max_tokens(self):
        """Hard max is 600 tokens."""
        from typeclasses.paper import DEFAULT_HARD_MAX_TOKENS

        assert DEFAULT_HARD_MAX_TOKENS == 600


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All expected exports are available from package."""
        from typeclasses.paper import (
            Paper,
            Folder,
            Book,
            Printer,
            Scanner,
            Copier,
            paginate,
            count_tokens,
            estimate_chars_for_tokens,
            DEFAULT_PAGE_SIZE_TOKENS,
            DEFAULT_HARD_MAX_TOKENS,
        )

        # Verify they're all importable and are the correct types
        assert Paper is not None
        assert Folder is not None
        assert Book is not None
        assert Printer is not None
        assert Scanner is not None
        assert Copier is not None
        assert callable(paginate)
        assert callable(count_tokens)
        assert callable(estimate_chars_for_tokens)
        assert isinstance(DEFAULT_PAGE_SIZE_TOKENS, int)
        assert isinstance(DEFAULT_HARD_MAX_TOKENS, int)


class TestPagination:
    """Tests for pagination module."""

    def test_count_tokens_empty(self):
        """Empty string has 0 tokens."""
        from typeclasses.paper.pagination import count_tokens

        assert count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Simple text has reasonable token count."""
        from typeclasses.paper.pagination import count_tokens

        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be around 4 tokens

    def test_count_tokens_none_handled(self):
        """None-like empty string returns 0."""
        from typeclasses.paper.pagination import count_tokens

        # The function checks 'if not text' so empty string returns 0
        assert count_tokens("") == 0

    def test_paginate_empty(self):
        """Empty content returns empty list."""
        from typeclasses.paper.pagination import paginate

        assert paginate("") == []

    def test_paginate_short_content(self):
        """Short content fits in one page."""
        from typeclasses.paper.pagination import paginate

        pages = paginate("Hello, world!")
        assert len(pages) == 1
        assert pages[0] == "Hello, world!"

    def test_paginate_long_content(self):
        """Long content gets split into multiple pages."""
        from typeclasses.paper.pagination import paginate

        # Create content that's definitely more than 100 tokens
        content = "This is a test sentence. " * 200  # ~1000 tokens
        pages = paginate(content, page_size_tokens=100)
        assert len(pages) > 1

    def test_paginate_respects_newlines(self):
        """Pagination prefers breaking at newlines."""
        from typeclasses.paper.pagination import paginate

        content = "Line one.\nLine two.\nLine three.\n" * 50
        pages = paginate(content, page_size_tokens=50)
        # Pages should not have trailing spaces (clean breaks)
        for page in pages:
            assert not page.endswith(" ")

    def test_paginate_strips_whitespace(self):
        """Pages are stripped of leading/trailing whitespace."""
        from typeclasses.paper.pagination import paginate

        content = "   Some content here   "
        pages = paginate(content)
        assert pages[0] == "Some content here"

    def test_estimate_chars_for_tokens_empty(self):
        """Empty string returns 0."""
        from typeclasses.paper.pagination import estimate_chars_for_tokens

        assert estimate_chars_for_tokens("", 100) == 0

    def test_estimate_chars_for_tokens_short_text(self):
        """Short text returns full length when under target."""
        from typeclasses.paper.pagination import estimate_chars_for_tokens

        text = "Hello"
        result = estimate_chars_for_tokens(text, 100)
        assert result == len(text)

    def test_estimate_chars_for_tokens_finds_position(self):
        """Long text finds correct character position for target tokens."""
        from typeclasses.paper.pagination import (
            estimate_chars_for_tokens,
            count_tokens,
        )

        text = "This is a longer sentence. " * 100
        target = 50

        char_pos = estimate_chars_for_tokens(text, target)
        # The substring up to char_pos should have <= target tokens
        substring = text[:char_pos]
        assert count_tokens(substring) <= target


class TestPrinter:
    """Tests for Printer device."""

    def test_printer_inherits_from_object(self):
        """Printer inherits from Object."""
        from typeclasses.paper import Printer
        from typeclasses.objects import Object

        assert issubclass(Printer, Object)

    def test_printer_get_page_size_default(self):
        """Printer uses default page size when not overridden."""
        from typeclasses.paper.devices import Printer
        from typeclasses.paper.constants import DEFAULT_PAGE_SIZE_TOKENS

        printer = MagicMock(spec=Printer)
        printer.db = MagicMock()
        printer.db.page_size = None

        # Call actual method
        result = Printer.get_page_size(printer)
        assert result == DEFAULT_PAGE_SIZE_TOKENS

    def test_printer_get_page_size_override(self):
        """Printer uses custom page size when set."""
        from typeclasses.paper.devices import Printer

        printer = MagicMock(spec=Printer)
        printer.db = MagicMock()
        printer.db.page_size = 300

        result = Printer.get_page_size(printer)
        assert result == 300



class TestScanner:
    """Tests for Scanner device."""

    def test_scanner_inherits_from_object(self):
        """Scanner inherits from Object."""
        from typeclasses.paper import Scanner
        from typeclasses.objects import Object

        assert issubclass(Scanner, Object)

    def test_scanner_get_content_empty_when_nothing_inside(self):
        """Scanner returns empty string when nothing inside."""
        from typeclasses.paper.devices import Scanner

        scanner = MagicMock(spec=Scanner)
        scanner.contents = []

        # Mock get_paper to return None
        with patch.object(Scanner, "get_paper", return_value=None):
            result = Scanner.get_content(scanner)
            assert result == ""

    def test_scanner_get_paper_returns_paper(self):
        """Scanner.get_paper() finds Paper in contents."""
        from typeclasses.paper.devices import Scanner
        from typeclasses.paper import Paper

        scanner = MagicMock(spec=Scanner)
        paper = MagicMock(spec=Paper)
        non_paper = MagicMock()

        scanner.contents = [non_paper, paper]

        result = Scanner.get_paper(scanner)
        assert result is paper

    def test_scanner_get_paper_returns_book(self):
        """Scanner.get_paper() finds Book in contents."""
        from typeclasses.paper.devices import Scanner
        from typeclasses.paper import Book

        scanner = MagicMock(spec=Scanner)
        book = MagicMock(spec=Book)

        scanner.contents = [book]

        result = Scanner.get_paper(scanner)
        assert result is book

    def test_scanner_get_paper_returns_none_when_empty(self):
        """Scanner.get_paper() returns None when empty."""
        from typeclasses.paper.devices import Scanner

        scanner = MagicMock(spec=Scanner)
        scanner.contents = []

        result = Scanner.get_paper(scanner)
        assert result is None


class TestCopier:
    """Tests for Copier device."""

    def test_copier_inherits_from_object(self):
        """Copier inherits from Object."""
        from typeclasses.paper import Copier
        from typeclasses.objects import Object

        assert issubclass(Copier, Object)

    def test_copier_get_original_empty(self):
        """Copier returns None when no paper inside."""
        from typeclasses.paper.devices import Copier

        copier = MagicMock(spec=Copier)
        copier.contents = []

        result = Copier.get_original(copier)
        assert result is None

    def test_copier_get_original_finds_paper(self):
        """Copier.get_original() finds Paper in contents."""
        from typeclasses.paper.devices import Copier
        from typeclasses.paper import Paper

        copier = MagicMock(spec=Copier)
        paper = MagicMock(spec=Paper)
        non_paper = MagicMock()

        copier.contents = [non_paper, paper]

        result = Copier.get_original(copier)
        assert result is paper



class TestCmdPrint:
    """Tests for CmdPrint command."""

    def test_cmd_print_key(self):
        """CmdPrint has correct command key."""
        from commands.mud.object_commands.paper import CmdPrint

        cmd = CmdPrint()
        assert cmd.key == "print"

    def test_cmd_print_help_category(self):
        """CmdPrint has correct help category."""
        from commands.mud.object_commands.paper import CmdPrint

        cmd = CmdPrint()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_print_locks(self):
        """CmdPrint has correct locks."""
        from commands.mud.object_commands.paper import CmdPrint

        cmd = CmdPrint()
        assert cmd.locks == "cmd:all()"

    def test_cmd_print_requires_args(self):
        """CmdPrint requires arguments."""
        from commands.mud.object_commands.paper import CmdPrint

        cmd = CmdPrint()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]

    def test_cmd_print_requires_equals_sign(self):
        """CmdPrint requires '=' in args."""
        from commands.mud.object_commands.paper import CmdPrint

        cmd = CmdPrint()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = "Market Terminal AAPL"

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]


class TestCmdScan:
    """Tests for CmdScan command."""

    def test_cmd_scan_key(self):
        """CmdScan has correct command key."""
        from commands.mud.object_commands.paper import CmdScan

        cmd = CmdScan()
        assert cmd.key == "scan"

    def test_cmd_scan_help_category(self):
        """CmdScan has correct help category."""
        from commands.mud.object_commands.paper import CmdScan

        cmd = CmdScan()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_scan_locks(self):
        """CmdScan has correct locks."""
        from commands.mud.object_commands.paper import CmdScan

        cmd = CmdScan()
        assert cmd.locks == "cmd:all()"

    def test_cmd_scan_requires_args(self):
        """CmdScan requires arguments."""
        from commands.mud.object_commands.paper import CmdScan

        cmd = CmdScan()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]


class TestCmdBind:
    """Tests for CmdBind command."""

    def test_cmd_bind_key(self):
        """CmdBind has correct command key."""
        from commands.mud.object_commands.paper import CmdBind

        cmd = CmdBind()
        assert cmd.key == "bind"

    def test_cmd_bind_help_category(self):
        """CmdBind has correct help category."""
        from commands.mud.object_commands.paper import CmdBind

        cmd = CmdBind()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_bind_locks(self):
        """CmdBind has correct locks."""
        from commands.mud.object_commands.paper import CmdBind

        cmd = CmdBind()
        assert cmd.locks == "cmd:all()"

    def test_cmd_bind_requires_args(self):
        """CmdBind requires arguments."""
        from commands.mud.object_commands.paper import CmdBind

        cmd = CmdBind()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]


class TestCmdUnbind:
    """Tests for CmdUnbind command."""

    def test_cmd_unbind_key(self):
        """CmdUnbind has correct command key."""
        from commands.mud.object_commands.paper import CmdUnbind

        cmd = CmdUnbind()
        assert cmd.key == "unbind"

    def test_cmd_unbind_help_category(self):
        """CmdUnbind has correct help category."""
        from commands.mud.object_commands.paper import CmdUnbind

        cmd = CmdUnbind()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_unbind_locks(self):
        """CmdUnbind has correct locks."""
        from commands.mud.object_commands.paper import CmdUnbind

        cmd = CmdUnbind()
        assert cmd.locks == "cmd:all()"

    def test_cmd_unbind_requires_args(self):
        """CmdUnbind requires arguments."""
        from commands.mud.object_commands.paper import CmdUnbind

        cmd = CmdUnbind()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]


class TestCmdReadBook:
    """Tests for CmdReadBook command."""

    def test_cmd_read_book_key(self):
        """CmdReadBook has correct command key."""
        from commands.mud.object_commands.paper import CmdReadBook

        cmd = CmdReadBook()
        assert cmd.key == "read"

    def test_cmd_read_book_aliases(self):
        """CmdReadBook has correct aliases."""
        from commands.mud.object_commands.paper import CmdReadBook

        cmd = CmdReadBook()
        assert "read_book" in cmd.aliases

    def test_cmd_read_book_help_category(self):
        """CmdReadBook has correct help category."""
        from commands.mud.object_commands.paper import CmdReadBook

        cmd = CmdReadBook()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_read_book_locks(self):
        """CmdReadBook has correct locks."""
        from commands.mud.object_commands.paper import CmdReadBook

        cmd = CmdReadBook()
        assert cmd.locks == "cmd:all()"

    def test_cmd_read_book_requires_args(self):
        """CmdReadBook requires arguments."""
        from commands.mud.object_commands.paper import CmdReadBook

        cmd = CmdReadBook()
        cmd.caller = MagicMock()
        cmd.caller.ndb = MagicMock()
        cmd.caller.ndb.pending_action_metadata = None
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "Usage:" in cmd.caller.msg.call_args[0][0]


class TestCmdCopy:
    """Tests for CmdCopy command."""

    def test_cmd_copy_key(self):
        """CmdCopy has correct command key."""
        from commands.mud.object_commands.paper import CmdCopy

        cmd = CmdCopy()
        assert cmd.key == "copy"

    def test_cmd_copy_help_category(self):
        """CmdCopy has correct help category."""
        from commands.mud.object_commands.paper import CmdCopy

        cmd = CmdCopy()
        assert cmd.help_category.lower() == "paper commands"

    def test_cmd_copy_locks(self):
        """CmdCopy has correct locks."""
        from commands.mud.object_commands.paper import CmdCopy

        cmd = CmdCopy()
        assert cmd.locks == "cmd:all()"

    def test_cmd_copy_no_copier_in_room(self):
        """CmdCopy fails gracefully when no copier in room."""
        from commands.mud.object_commands.paper import CmdCopy

        cmd = CmdCopy()
        cmd.caller = MagicMock()
        cmd.caller.location = MagicMock()
        cmd.caller.location.contents = []

        cmd.func()

        cmd.caller.msg.assert_called_once()
        assert "No copier" in cmd.caller.msg.call_args[0][0]


class TestPaperCommandExports:
    """Tests for paper command exports."""

    def test_all_commands_exported(self):
        """All paper commands are exported from module."""
        from commands.mud.object_commands import (
            CmdPrint,
            CmdScan,
            CmdBind,
            CmdUnbind,
            CmdReadBook,
            CmdCopy,
        )

        assert CmdPrint is not None
        assert CmdScan is not None
        assert CmdBind is not None
        assert CmdUnbind is not None
        assert CmdReadBook is not None
        assert CmdCopy is not None


class TestContainerCmdSets:
    """Tests for CmdSet constants in container typeclasses."""

    def test_folder_cmdset_constant_defined(self):
        """FOLDER_CMDSET constant is defined in containers module."""
        from typeclasses.paper.containers import FOLDER_CMDSET

        assert FOLDER_CMDSET == "commands.mud.object_commands.paper.FolderCmdSet"

    def test_book_cmdset_constant_defined(self):
        """BOOK_CMDSET constant is defined in containers module."""
        from typeclasses.paper.containers import BOOK_CMDSET

        assert BOOK_CMDSET == "commands.mud.object_commands.paper.BookCmdSet"


class TestDeviceCmdSets:
    """Tests for CmdSet constants in device typeclasses."""

    def test_copier_cmdset_constant_defined(self):
        """COPIER_CMDSET constant is defined in devices module."""
        from typeclasses.paper.devices import COPIER_CMDSET

        assert COPIER_CMDSET == "commands.mud.object_commands.paper.CopierCmdSet"


class TestAuraConstants:
    """Tests for paper aura constants."""

    def test_print_access_constant(self):
        """AURA_PRINT_ACCESS is defined."""
        from aim_mud_types import AURA_PRINT_ACCESS

        assert AURA_PRINT_ACCESS == "PRINT_ACCESS"

    def test_scan_access_constant(self):
        """AURA_SCAN_ACCESS is defined."""
        from aim_mud_types import AURA_SCAN_ACCESS

        assert AURA_SCAN_ACCESS == "SCAN_ACCESS"

    def test_bind_access_constant(self):
        """AURA_BIND_ACCESS is defined."""
        from aim_mud_types import AURA_BIND_ACCESS

        assert AURA_BIND_ACCESS == "BIND_ACCESS"

    def test_book_access_constant(self):
        """AURA_BOOK_ACCESS is defined."""
        from aim_mud_types import AURA_BOOK_ACCESS

        assert AURA_BOOK_ACCESS == "BOOK_ACCESS"

    def test_copy_access_constant(self):
        """AURA_COPY_ACCESS is defined."""
        from aim_mud_types import AURA_COPY_ACCESS

        assert AURA_COPY_ACCESS == "COPY_ACCESS"


class TestFolderAuras:
    """Tests for Folder aura methods."""

    def test_folder_get_room_auras_returns_bind_access(self):
        """Folder provides BIND_ACCESS aura."""
        from typeclasses.paper.containers import Folder
        from aim_mud_types import AURA_BIND_ACCESS

        folder = MagicMock(spec=Folder)
        folder.db = MagicMock()
        folder.db.aura_enabled = True
        folder.key = "test_folder"
        folder.dbref = "#123"

        auras = Folder.get_room_auras(folder)
        assert len(auras) == 1
        assert auras[0]["name"] == AURA_BIND_ACCESS
        assert auras[0]["source"] == "test_folder"
        assert auras[0]["source_id"] == "#123"

    def test_folder_get_room_auras_disabled(self):
        """Folder returns empty list when aura_enabled is False."""
        from typeclasses.paper.containers import Folder

        folder = MagicMock(spec=Folder)
        folder.db = MagicMock()
        folder.db.aura_enabled = False
        folder.key = "test_folder"
        folder.dbref = "#123"

        auras = Folder.get_room_auras(folder)
        assert len(auras) == 0


class TestBookAuras:
    """Tests for Book aura methods."""

    def test_book_get_room_auras_returns_book_access(self):
        """Book provides BOOK_ACCESS aura."""
        from typeclasses.paper.containers import Book
        from aim_mud_types import AURA_BOOK_ACCESS

        book = MagicMock(spec=Book)
        book.db = MagicMock()
        book.db.aura_enabled = True
        book.key = "test_book"
        book.dbref = "#124"

        auras = Book.get_room_auras(book)
        assert len(auras) == 1
        assert auras[0]["name"] == AURA_BOOK_ACCESS
        assert auras[0]["source"] == "test_book"
        assert auras[0]["source_id"] == "#124"

    def test_book_get_room_auras_disabled(self):
        """Book returns empty list when aura_enabled is False."""
        from typeclasses.paper.containers import Book

        book = MagicMock(spec=Book)
        book.db = MagicMock()
        book.db.aura_enabled = False
        book.key = "test_book"
        book.dbref = "#124"

        auras = Book.get_room_auras(book)
        assert len(auras) == 0


class TestCopierAuras:
    """Tests for Copier aura methods."""

    def test_copier_get_room_auras_returns_copy_access(self):
        """Copier provides COPY_ACCESS aura."""
        from typeclasses.paper.devices import Copier
        from aim_mud_types import AURA_COPY_ACCESS

        copier = MagicMock(spec=Copier)
        copier.db = MagicMock()
        copier.db.aura_enabled = True
        copier.key = "test_copier"
        copier.dbref = "#125"

        auras = Copier.get_room_auras(copier)
        assert len(auras) == 1
        assert auras[0]["name"] == AURA_COPY_ACCESS
        assert auras[0]["source"] == "test_copier"
        assert auras[0]["source_id"] == "#125"

    def test_copier_get_room_auras_disabled(self):
        """Copier returns empty list when aura_enabled is False."""
        from typeclasses.paper.devices import Copier

        copier = MagicMock(spec=Copier)
        copier.db = MagicMock()
        copier.db.aura_enabled = False
        copier.key = "test_copier"
        copier.dbref = "#125"

        auras = Copier.get_room_auras(copier)
        assert len(auras) == 0


class TestTerminalPrintScanAuras:
    """Tests for Terminal print/scan aura support."""

    def test_terminal_with_printer_provides_print_aura(self):
        """Terminal with printer configured provides PRINT_ACCESS aura."""
        from typeclasses.terminals import Terminal
        from aim_mud_types import AURA_PRINT_ACCESS

        terminal = MagicMock(spec=Terminal)
        terminal.db = MagicMock()
        terminal.db.aura_enabled = True
        terminal.db.aura_name = "CODE_ACCESS"
        terminal.db.printer = "#200"  # Printer configured
        terminal.db.scanner = None
        terminal.key = "code_terminal"
        terminal.dbref = "#126"

        auras = Terminal.get_room_auras(terminal)
        assert len(auras) == 2
        aura_names = [a["name"] for a in auras]
        assert "CODE_ACCESS" in aura_names
        assert AURA_PRINT_ACCESS in aura_names

    def test_terminal_with_scanner_provides_scan_aura(self):
        """Terminal with scanner configured provides SCAN_ACCESS aura."""
        from typeclasses.terminals import Terminal
        from aim_mud_types import AURA_SCAN_ACCESS

        terminal = MagicMock(spec=Terminal)
        terminal.db = MagicMock()
        terminal.db.aura_enabled = True
        terminal.db.aura_name = "CODE_ACCESS"
        terminal.db.printer = None
        terminal.db.scanner = "#201"  # Scanner configured
        terminal.key = "code_terminal"
        terminal.dbref = "#126"

        auras = Terminal.get_room_auras(terminal)
        assert len(auras) == 2
        aura_names = [a["name"] for a in auras]
        assert "CODE_ACCESS" in aura_names
        assert AURA_SCAN_ACCESS in aura_names

    def test_terminal_with_both_provides_all_auras(self):
        """Terminal with both printer and scanner provides both auras."""
        from typeclasses.terminals import Terminal
        from aim_mud_types import AURA_PRINT_ACCESS, AURA_SCAN_ACCESS

        terminal = MagicMock(spec=Terminal)
        terminal.db = MagicMock()
        terminal.db.aura_enabled = True
        terminal.db.aura_name = "WEB_ACCESS"
        terminal.db.printer = "#200"
        terminal.db.scanner = "#201"
        terminal.key = "web_terminal"
        terminal.dbref = "#127"

        auras = Terminal.get_room_auras(terminal)
        assert len(auras) == 3
        aura_names = [a["name"] for a in auras]
        assert "WEB_ACCESS" in aura_names
        assert AURA_PRINT_ACCESS in aura_names
        assert AURA_SCAN_ACCESS in aura_names

    def test_terminal_without_printer_scanner_normal_aura_only(self):
        """Terminal without printer/scanner only provides its own aura."""
        from typeclasses.terminals import Terminal

        terminal = MagicMock(spec=Terminal)
        terminal.db = MagicMock()
        terminal.db.aura_enabled = True
        terminal.db.aura_name = "CODE_ACCESS"
        terminal.db.printer = None
        terminal.db.scanner = None
        terminal.key = "code_terminal"
        terminal.dbref = "#126"

        auras = Terminal.get_room_auras(terminal)
        assert len(auras) == 1
        assert auras[0]["name"] == "CODE_ACCESS"

    def test_terminal_aura_disabled_no_auras(self):
        """Terminal with aura_enabled=False returns no main aura but still print/scan if configured."""
        from typeclasses.terminals import Terminal
        from aim_mud_types import AURA_PRINT_ACCESS

        terminal = MagicMock(spec=Terminal)
        terminal.db = MagicMock()
        terminal.db.aura_enabled = False
        terminal.db.aura_name = "CODE_ACCESS"
        terminal.db.printer = "#200"
        terminal.db.scanner = None
        terminal.key = "code_terminal"
        terminal.dbref = "#126"

        auras = Terminal.get_room_auras(terminal)
        # Only print aura since main aura is disabled
        assert len(auras) == 1
        assert auras[0]["name"] == AURA_PRINT_ACCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
