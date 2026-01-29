#!/usr/bin/env python3

import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Helper to unparse AST nodes to string (requires Python 3.9+)
# For older Pythons, you might need a backport or more manual string construction.
try:
    from ast import unparse
except ImportError:
    # Basic fallback for older Python versions for simple names/attributes
    # This is NOT a full unparse implementation.
    def unparse(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{unparse(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant): # Python 3.8+ for ast.Constant
            return repr(node.value)
        elif isinstance(node, (ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis)): # Older AST nodes
            if isinstance(node, ast.Num): return repr(node.n)
            if isinstance(node, ast.Str): return repr(node.s)
            if isinstance(node, ast.Bytes): return repr(node.s)
            if isinstance(node, ast.NameConstant): return repr(node.value)
            if isinstance(node, ast.Ellipsis): return "..."
        return "..." # Placeholder for complex nodes if ast.unparse is not available


class SignatureVisitor(ast.NodeVisitor):
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.current_class_name: Optional[str] = None
        self.signatures: Dict[str, Dict[str, Any]] = {
            "classes": {},
            "functions": {},
        }

    def _format_arg(self, arg: ast.arg, default_expr: Optional[ast.AST] = None) -> str:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {unparse(arg.annotation).strip()}"
            except Exception:
                arg_str += ": ?" # Fallback if unparsing annotation fails
        if default_expr:
            try:
                arg_str += f" = {unparse(default_expr).strip()}"
            except Exception:
                arg_str += " = ?" # Fallback if unparsing default fails
        return arg_str

    def _format_arguments(self, args_node: ast.arguments) -> str:
        args_list: List[str] = []

        # Positional-only arguments (Python 3.8+)
        if hasattr(args_node, 'posonlyargs') and args_node.posonlyargs:
            for i, arg in enumerate(args_node.posonlyargs):
                # Defaults for posonlyargs are in args_node.defaults
                # They appear at the end of combined posonlyargs and args
                # This calculation is a bit simplified for clarity; real alignment is complex
                # For simplicity, we assume defaults apply to the *end* of regular args first.
                # True default alignment: defaults apply to last len(defaults) of (posonlyargs + args)
                # However, ast.unparse handles this correctly. We try to emulate part of it.
                args_list.append(self._format_arg(arg))
            args_list.append("/")

        # Regular arguments (positional or keyword)
        num_regular_args = len(args_node.args)
        num_defaults = len(args_node.defaults)
        for i, arg in enumerate(args_node.args):
            default_expr: Optional[ast.AST] = None
            # Defaults are aligned from the right of args_node.args
            default_idx = i - (num_regular_args - num_defaults)
            if default_idx >= 0:
                default_expr = args_node.defaults[default_idx]
            args_list.append(self._format_arg(arg, default_expr))

        # *args
        if args_node.vararg:
            args_list.append(f"*{self._format_arg(args_node.vararg)}")

        # Keyword-only arguments
        if args_node.kwonlyargs:
            if not args_node.vararg: # if no *args, need a standalone *
                args_list.append("*")
            for i, arg in enumerate(args_node.kwonlyargs):
                args_list.append(self._format_arg(arg, args_node.kw_defaults[i]))

        # **kwargs
        if args_node.kwarg:
            args_list.append(f"**{self._format_arg(args_node.kwarg)}")

        return ", ".join(args_list)

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        base_classes_str = ""
        if node.bases:
            base_classes_str = f"({', '.join(unparse(b).strip() for b in node.bases)})"
        
        signature = f"class {class_name}{base_classes_str}:"
        
        self.signatures["classes"][class_name] = {
            "signature": signature,
            "methods": {},
            "docstring": ast.get_docstring(node, clean=False) # Get raw docstring
        }
        
        # Important: set current class context for methods
        original_class_name = self.current_class_name
        self.current_class_name = class_name
        self.generic_visit(node) # Visit methods within this class
        self.current_class_name = original_class_name # Restore context

    def _process_function_def(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        func_name = node.name
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        params_str = self._format_arguments(node.args)
        
        return_annotation_str = ""
        if node.returns:
            try:
                return_annotation_str = f" -> {unparse(node.returns).strip()}"
            except Exception:
                return_annotation_str = " -> ?"

        prefix = "async def" if is_async else "def"
        signature = f"{prefix} {func_name}({params_str}){return_annotation_str}:"
        docstring = ast.get_docstring(node, clean=False)

        if self.current_class_name:
            # This is a method
            if self.current_class_name in self.signatures["classes"]:
                self.signatures["classes"][self.current_class_name]["methods"][func_name] = {
                    "signature": signature,
                    "docstring": docstring
                }
            else:
                # Should not happen if logic is correct, but good for robustness
                print(f"Warning: Method {func_name} found outside known class {self.current_class_name} in {self.filepath}")
        else:
            # This is a top-level function
            self.signatures["functions"][func_name] = {
                "signature": signature,
                "docstring": docstring
            }
        
        # Do not call generic_visit(node) here for functions unless you want to
        # dive into nested functions/classes defined inside this function.
        # For just signatures, we usually stop at the function definition.

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function_def(node)

def gather_signatures_from_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Parses a Python file and extracts class and function signatures.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as source_file:
            source_code = source_file.read()
        tree = ast.parse(source_code, filename=str(filepath))
        visitor = SignatureVisitor(filepath)
        visitor.visit(tree)
        return visitor.signatures
    except SyntaxError as e:
        print(f"Error: Could not parse {filepath}. Syntax error: {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Scan Python files in a folder and gather class and method signatures using AST."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="The folder containing Python files to scan."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scan recursively into subfolders."
    )
    parser.add_argument(
        "--include-docstrings",
        action="store_true",
        help="Include docstrings in the output."
    )

    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return

    print(f"Scanning folder: {folder_path}{' (recursively)' if args.recursive else ''}\n")

    all_results: Dict[str, Dict[str, Any]] = {}

    glob_pattern = "*.py"
    file_iterator = folder_path.rglob(glob_pattern) if args.recursive else folder_path.glob(glob_pattern)

    for py_file in file_iterator:
        if py_file.is_file():
            file_signatures = gather_signatures_from_file(py_file)
            if file_signatures and (file_signatures["classes"] or file_signatures["functions"]):
                all_results[str(py_file.relative_to(folder_path.parent))] = file_signatures
            elif file_signatures:
                 print(f"  No classes or functions found in {py_file}")
            print("-" * 20)


    print("\n" + "=" * 40)
    print("         Collected Signatures")
    print("=" * 40 + "\n")

    if not all_results:
        print("No Python files with classes or functions found.")
        return

    for filepath_str, data in all_results.items():
        print(f"File: {filepath_str}")
        
        if data["functions"]:
            print("  Top-level Functions:")
            for func_name, func_data in data["functions"].items():
                print(f"    {func_data['signature']}")
                if args.include_docstrings and func_data['docstring']:
                    doc_lines = func_data['docstring'].strip().split('\n')
                    print(f"      \"\"\"{doc_lines[0]}{'...' if len(doc_lines) > 1 else ''}\"\"\"")
            print()

        if data["classes"]:
            print("  Classes:")
            for class_name, class_data in data["classes"].items():
                print(f"    {class_data['signature']}")
                if args.include_docstrings and class_data['docstring']:
                    doc_lines = class_data['docstring'].strip().split('\n')
                    print(f"      \"\"\"{doc_lines[0]}{'...' if len(doc_lines) > 1 else ''}\"\"\"")

                if class_data["methods"]:
                    print("      Methods:")
                    for method_name, method_data in class_data["methods"].items():
                        print(f"        {method_data['signature']}")
                        if args.include_docstrings and method_data['docstring']:
                            doc_lines = method_data['docstring'].strip().split('\n')
                            print(f"          \"\"\"{doc_lines[0]}{'...' if len(doc_lines) > 1 else ''}\"\"\"")
                print()
        print("-" * 30)


if __name__ == "__main__":
    # Ensure ast.unparse is available or provide a note
    if 'unparse' not in dir(ast):
        print("Warning: ast.unparse not found (requires Python 3.9+).")
        print("Signature formatting for annotations and defaults will be basic ('?').\n")
    
    # For type hint Union in _process_function_def
    from typing import Union

    main()