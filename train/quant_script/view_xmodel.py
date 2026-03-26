"""
view_xmodel.py
==============
A command-line tool for inspecting Vitis-AI .xmodel files using the
XIR Python API (PyXIR). Compatible with Vitis-AI 1.3+ environments
where xir.so is installed.

Usage:
    python3 view_xmodel.py <path_to_model.xmodel> [options]

Examples:
    python3 view_xmodel.py resnet50.xmodel
    python3 view_xmodel.py resnet50.xmodel --ops --tensors --attrs
    python3 view_xmodel.py resnet50.xmodel --dot graph.dot

References:
    [1] Xilinx / AMD Vitis-AI GitHub Repository
        https://github.com/Xilinx/Vitis-AI

    [2] Vitis-AI Documentation (GitHub Pages)
        https://xilinx.github.io/Vitis-AI/

    [3] XIR (Xilinx Intermediate Representation) Python API Reference
        https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/xir.html

    [4] Vitis-AI Model Zoo
        https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo

    [5] Vitis-AI Quantizer & Compiler User Guide (UG1414)
        https://docs.xilinx.com/r/en-US/ug1414-vitis-ai

    [6] PyXIR: Xilinx Intermediate Representation for Python
        https://github.com/Xilinx/pyxir
"""

import sys
import argparse

# ── Verify that the xir module is available ───────────────────────────────────
try:
    import xir
except ImportError:
    print("[ERROR] The 'xir' module was not found.")
    print("        Please run this script inside the Vitis-AI Docker container,")
    print("        or ensure that xir.so is installed in your Python site-packages.")
    sys.exit(1)


# ==============================================================================
# Helper utilities
# ==============================================================================

SEP_MAJOR = "=" * 70
SEP_MINOR = "-" * 60


def section(title: str) -> None:
    """Print a major section header."""
    print(f"\n{SEP_MAJOR}")
    print(f"  {title}")
    print(SEP_MAJOR)


def subsection(title: str) -> None:
    """Print a minor subsection header."""
    print(f"\n{SEP_MINOR}")
    print(f"  {title}")
    print(SEP_MINOR)


def safe_attrs(obj) -> dict:
    """
    Safely retrieve all attributes from an Op or Subgraph object.
    Returns an empty dict if the object exposes no attribute interface.
    """
    try:
        keys = obj.get_attr_names()
        result = {}
        for k in keys:
            try:
                result[k] = obj.get_attr(k)
            except Exception:
                result[k] = "<unreadable>"
        return result
    except Exception:
        return {}


def format_shape(shape) -> str:
    """Format a shape tuple/list as a human-readable string (e.g. 1x3x224x224)."""
    if shape is None:
        return "unknown"
    return "x".join(str(d) for d in shape)


# ==============================================================================
# Core inspection functions
# ==============================================================================

def show_graph_summary(graph) -> None:
    """Display high-level graph statistics and op-type distribution."""
    section("Graph Summary")
    ops = graph.get_ops()
    print(f"  Graph name  : {graph.get_name()}")
    print(f"  Total ops   : {len(ops)}")

    # Count how many times each op type appears
    type_count: dict = {}
    for op in ops:
        t = op.get_type()
        type_count[t] = type_count.get(t, 0) + 1

    print(f"\n  Op type distribution ({len(type_count)} unique types):")
    for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"    {t:<40s} x {c}")


def show_ops(graph, show_attrs: bool = False, show_tensors: bool = False) -> None:
    """
    List every Op in topological order.

    Args:
        graph       : xir.Graph object loaded from the .xmodel file.
        show_attrs  : When True, print all key-value attributes for each Op.
        show_tensors: When True, print detailed input tensor information.
    """
    section("Op List (Topological Order)")
    ops = graph.topological_sort()

    for i, op in enumerate(ops):
        op_name    = op.get_name()
        op_type    = op.get_type()
        out_tensor = op.get_output_tensor()

        shape_str = format_shape(out_tensor.get_shape()) if out_tensor else "-"
        dtype_str = str(out_tensor.get_data_type())      if out_tensor else "-"

        print(f"\n  [{i:04d}] {op_name}")
        print(f"         type   : {op_type}")
        print(f"         output : shape={shape_str}  dtype={dtype_str}")

        # Show upstream (producer) ops
        try:
            in_ops_dict = op.get_input_ops()
            if in_ops_dict:
                parts = []
                for arg_name, in_op_list in in_ops_dict.items():
                    for in_op in in_op_list:
                        parts.append(f"{arg_name}->{in_op.get_name()}")
                print(f"         inputs : {', '.join(parts) if parts else '(none)'}")
        except Exception:
            pass

        # Detailed input tensor info (verbose mode)
        if show_tensors:
            try:
                in_tensors = op.get_input_tensors()
                for t in in_tensors:
                    print(f"           in_tensor: {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
            except Exception:
                pass

        # Op attributes (verbose mode)
        if show_attrs:
            attrs = safe_attrs(op)
            if attrs:
                print(f"         attrs  :")
                for k, v in attrs.items():
                    print(f"           {k}: {v}")


def show_subgraphs(graph, show_attrs: bool = False) -> None:
    """
    Display the subgraph hierarchy and DPU/CPU partition information.

    After quantization and compilation with Vitis-AI, a graph is split
    into DPU subgraphs (accelerated on the FPGA) and CPU subgraphs
    (executed on the ARM/x86 host). This function shows the tree and
    summarises each DPU subgraph.

    Args:
        graph      : xir.Graph object.
        show_attrs : When True, also print non-device attributes.
    """
    section("Subgraph Structure")

    root = graph.get_root_subgraph()

    def _print_subgraph(sg, depth: int = 0) -> None:
        """Recursively print the subgraph tree."""
        indent    = "  " * depth
        name      = sg.get_name()
        op_num    = sg.get_op_num()
        child_num = len(sg.get_children())

        # 'device' attribute is set only in compiled xmodels
        device = "-"
        try:
            if sg.has_attr("device"):
                device = sg.get_attr("device")
        except Exception:
            pass

        print(f"{indent}> {name}")
        print(f"{indent}  op_num={op_num}  children={child_num}  device={device}")

        if show_attrs:
            attrs = safe_attrs(sg)
            if attrs:
                for k, v in attrs.items():
                    if k != "device":
                        print(f"{indent}  attr: {k} = {v}")

        for child in sg.get_children():
            _print_subgraph(child, depth + 1)

    _print_subgraph(root)

    # Summarise DPU subgraphs specifically
    subsection("DPU Subgraph Summary")
    try:
        child_subgraphs = root.toposort_child_subgraph()
        dpu_sgs = [
            sg for sg in child_subgraphs
            if sg.has_attr("device") and
               sg.get_attr("device").upper() == "DPU"
        ]

        if dpu_sgs:
            print(f"  Found {len(dpu_sgs)} DPU subgraph(s):")
            for sg in dpu_sgs:
                inputs  = sg.get_input_tensors()
                outputs = sg.get_output_tensors()
                print(f"\n    [DPU] {sg.get_name()}")
                print(f"      Op count       : {sg.get_op_num()}")
                print(f"      Input tensors ({len(inputs)}):")
                for t in inputs:
                    print(f"        - {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
                print(f"      Output tensors ({len(outputs)}):")
                for t in outputs:
                    print(f"        - {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
        else:
            print("  No DPU subgraphs found "
                  "(the model may be quantized but not yet compiled).")
    except Exception as e:
        print(f"  Could not retrieve DPU subgraph information: {e}")


def show_io_tensors(graph) -> None:
    """
    Display the graph-level input and output tensors.

    These are derived from the root subgraph and represent the overall
    model interface (e.g. image input batch and class-score output).
    """
    section("Graph Input / Output Tensors")

    try:
        root           = graph.get_root_subgraph()
        input_tensors  = root.get_input_tensors()
        output_tensors = root.get_output_tensors()

        print(f"\n  Input tensors ({len(input_tensors)}):")
        for t in input_tensors:
            print(f"    - {t.get_name()}")
            print(f"      shape={format_shape(t.get_shape())}  "
                  f"dtype={t.get_data_type()}")

        print(f"\n  Output tensors ({len(output_tensors)}):")
        for t in output_tensors:
            print(f"    - {t.get_name()}")
            print(f"      shape={format_shape(t.get_shape())}  "
                  f"dtype={t.get_data_type()}")
    except Exception as e:
        print(f"  Could not retrieve IO tensors: {e}")


# ==============================================================================
# Graphviz DOT export
# ==============================================================================

def export_dot(graph, dot_path: str) -> None:
    """
    Export the computation graph as a Graphviz DOT file.

    Each node represents one Op (labelled with name, type, and output shape).
    Directed edges represent data flow between ops.

    The resulting DOT file can be rendered with:
        dot -Tpng output.dot -o graph.png
        dot -Tsvg output.dot -o graph.svg

    Args:
        graph    : xir.Graph object.
        dot_path : Destination file path for the DOT output.
    """
    ops = graph.topological_sort()

    def node_id(name: str) -> str:
        """Wrap a node name in quotes, escaping any internal quote characters."""
        return '"' + name.replace('"', '\\"') + '"'

    lines = [
        "digraph xmodel {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fillcolor="#AED6F1", '
        'fontname="Arial", fontsize=10];',
        "  edge [fontsize=9];",
        "",
    ]

    # Declare all nodes
    for op in ops:
        label = f"{op.get_name()}\\n[{op.get_type()}]"
        out_t = op.get_output_tensor()
        if out_t:
            label += f"\\n{format_shape(out_t.get_shape())}"
        lines.append(f"  {node_id(op.get_name())} [label={node_id(label)}];")

    lines.append("")

    # Declare all edges
    for op in ops:
        try:
            in_ops_dict = op.get_input_ops()
            for _, in_op_list in in_ops_dict.items():
                for in_op in in_op_list:
                    lines.append(
                        f"  {node_id(in_op.get_name())} -> "
                        f"{node_id(op.get_name())};"
                    )
        except Exception:
            pass

    lines.append("}")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [DOT] Exported to : {dot_path}")
    print(f"  [DOT] Render PNG  : dot -Tpng {dot_path} -o graph.png")
    print(f"  [DOT] Render SVG  : dot -Tsvg {dot_path} -o graph.svg")


# ==============================================================================
# Command-line interface
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vitis-AI xmodel Inspector (PyXIR / xir)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic summary (graph info + subgraph tree + IO tensors)
  python3 view_xmodel.py resnet50.xmodel

  # List all ops in topological order
  python3 view_xmodel.py resnet50.xmodel --ops

  # Full detail: ops + attributes + tensor info
  python3 view_xmodel.py resnet50.xmodel --ops --attrs --tensors

  # Export a Graphviz DOT diagram
  python3 view_xmodel.py resnet50.xmodel --dot output.dot
        """
    )
    parser.add_argument(
        "xmodel",
        help="Path to the .xmodel file to inspect."
    )
    parser.add_argument(
        "--ops",
        action="store_true",
        help="List all ops in topological order with shape/dtype information."
    )
    parser.add_argument(
        "--attrs",
        action="store_true",
        help="Print all key-value attributes for each op / subgraph (use with --ops)."
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="Print detailed input tensor information for each op (use with --ops)."
    )
    parser.add_argument(
        "--dot",
        metavar="FILE",
        help="Export the computation graph to a Graphviz DOT file."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{SEP_MAJOR}")
    print(f"  Vitis-AI xmodel Inspector  (PyXIR / xir)")
    print(f"{SEP_MAJOR}")
    print(f"  Loading: {args.xmodel}")

    # ── Load the xmodel ───────────────────────────────────────────────────────
    try:
        graph = xir.Graph.deserialize(args.xmodel)
    except Exception as e:
        print(f"\n[ERROR] Failed to load xmodel: {e}")
        sys.exit(1)

    print(f"  Loaded successfully.")

    # ── Run selected inspection modules ──────────────────────────────────────
    show_graph_summary(graph)
    show_io_tensors(graph)
    show_subgraphs(graph, show_attrs=args.attrs)

    if args.ops:
        show_ops(graph, show_attrs=args.attrs, show_tensors=args.tensors)

    if args.dot:
        export_dot(graph, args.dot)

    print(f"\n{SEP_MAJOR}")
    print(f"  Done.")
    print(f"{SEP_MAJOR}\n")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
view_xmodel.py
==============
A command-line tool for inspecting Vitis-AI .xmodel files using the
XIR Python API (PyXIR). Compatible with Vitis-AI 1.3+ environments
where xir.so is installed.

Usage:
    python3 view_xmodel.py <path_to_model.xmodel> [options]

Examples:
    python3 view_xmodel.py resnet50.xmodel
    python3 view_xmodel.py resnet50.xmodel --ops --tensors --attrs
    python3 view_xmodel.py resnet50.xmodel --dot graph.dot

References:
    [1] Xilinx / AMD Vitis-AI GitHub Repository
        https://github.com/Xilinx/Vitis-AI

    [2] Vitis-AI Documentation (GitHub Pages)
        https://xilinx.github.io/Vitis-AI/

    [3] XIR (Xilinx Intermediate Representation) Python API Reference
        https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/xir.html

    [4] Vitis-AI Model Zoo
        https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo

    [5] Vitis-AI Quantizer & Compiler User Guide (UG1414)
        https://docs.xilinx.com/r/en-US/ug1414-vitis-ai

    [6] PyXIR: Xilinx Intermediate Representation for Python
        https://github.com/Xilinx/pyxir
"""

import sys
import argparse

# ── Verify that the xir module is available ───────────────────────────────────
try:
    import xir
except ImportError:
    print("[ERROR] The 'xir' module was not found.")
    print("        Please run this script inside the Vitis-AI Docker container,")
    print("        or ensure that xir.so is installed in your Python site-packages.")
    sys.exit(1)


# ==============================================================================
# Helper utilities
# ==============================================================================

SEP_MAJOR = "=" * 70
SEP_MINOR = "-" * 60


def section(title: str) -> None:
    """Print a major section header."""
    print(f"\n{SEP_MAJOR}")
    print(f"  {title}")
    print(SEP_MAJOR)


def subsection(title: str) -> None:
    """Print a minor subsection header."""
    print(f"\n{SEP_MINOR}")
    print(f"  {title}")
    print(SEP_MINOR)


def safe_attrs(obj) -> dict:
    """
    Safely retrieve all attributes from an Op or Subgraph object.
    Returns an empty dict if the object exposes no attribute interface.
    """
    try:
        keys = obj.get_attr_names()
        result = {}
        for k in keys:
            try:
                result[k] = obj.get_attr(k)
            except Exception:
                result[k] = "<unreadable>"
        return result
    except Exception:
        return {}


def format_shape(shape) -> str:
    """Format a shape tuple/list as a human-readable string (e.g. 1x3x224x224)."""
    if shape is None:
        return "unknown"
    return "x".join(str(d) for d in shape)


# ==============================================================================
# Core inspection functions
# ==============================================================================

def show_graph_summary(graph) -> None:
    """Display high-level graph statistics and op-type distribution."""
    section("Graph Summary")
    ops = graph.get_ops()
    print(f"  Graph name  : {graph.get_name()}")
    print(f"  Total ops   : {len(ops)}")

    # Count how many times each op type appears
    type_count: dict = {}
    for op in ops:
        t = op.get_type()
        type_count[t] = type_count.get(t, 0) + 1

    print(f"\n  Op type distribution ({len(type_count)} unique types):")
    for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"    {t:<40s} x {c}")


def show_ops(graph, show_attrs: bool = False, show_tensors: bool = False) -> None:
    """
    List every Op in topological order.

    Args:
        graph       : xir.Graph object loaded from the .xmodel file.
        show_attrs  : When True, print all key-value attributes for each Op.
        show_tensors: When True, print detailed input tensor information.
    """
    section("Op List (Topological Order)")
    ops = graph.topological_sort()

    for i, op in enumerate(ops):
        op_name    = op.get_name()
        op_type    = op.get_type()
        out_tensor = op.get_output_tensor()

        shape_str = format_shape(out_tensor.get_shape()) if out_tensor else "-"
        dtype_str = str(out_tensor.get_data_type())      if out_tensor else "-"

        print(f"\n  [{i:04d}] {op_name}")
        print(f"         type   : {op_type}")
        print(f"         output : shape={shape_str}  dtype={dtype_str}")

        # Show upstream (producer) ops
        try:
            in_ops_dict = op.get_input_ops()
            if in_ops_dict:
                parts = []
                for arg_name, in_op_list in in_ops_dict.items():
                    for in_op in in_op_list:
                        parts.append(f"{arg_name}->{in_op.get_name()}")
                print(f"         inputs : {', '.join(parts) if parts else '(none)'}")
        except Exception:
            pass

        # Detailed input tensor info (verbose mode)
        if show_tensors:
            try:
                in_tensors = op.get_input_tensors()
                for t in in_tensors:
                    print(f"           in_tensor: {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
            except Exception:
                pass

        # Op attributes (verbose mode)
        if show_attrs:
            attrs = safe_attrs(op)
            if attrs:
                print(f"         attrs  :")
                for k, v in attrs.items():
                    print(f"           {k}: {v}")


def show_subgraphs(graph, show_attrs: bool = False) -> None:
    """
    Display the subgraph hierarchy and DPU/CPU partition information.

    After quantization and compilation with Vitis-AI, a graph is split
    into DPU subgraphs (accelerated on the FPGA) and CPU subgraphs
    (executed on the ARM/x86 host). This function shows the tree and
    summarises each DPU subgraph.

    Args:
        graph      : xir.Graph object.
        show_attrs : When True, also print non-device attributes.
    """
    section("Subgraph Structure")

    root = graph.get_root_subgraph()

    def _print_subgraph(sg, depth: int = 0) -> None:
        """Recursively print the subgraph tree."""
        indent    = "  " * depth
        name      = sg.get_name()
        op_num    = sg.get_op_num()
        child_num = len(sg.get_children())

        # 'device' attribute is set only in compiled xmodels
        device = "-"
        try:
            if sg.has_attr("device"):
                device = sg.get_attr("device")
        except Exception:
            pass

        print(f"{indent}> {name}")
        print(f"{indent}  op_num={op_num}  children={child_num}  device={device}")

        if show_attrs:
            attrs = safe_attrs(sg)
            if attrs:
                for k, v in attrs.items():
                    if k != "device":
                        print(f"{indent}  attr: {k} = {v}")

        for child in sg.get_children():
            _print_subgraph(child, depth + 1)

    _print_subgraph(root)

    # Summarise DPU subgraphs specifically
    subsection("DPU Subgraph Summary")
    try:
        child_subgraphs = root.toposort_child_subgraph()
        dpu_sgs = [
            sg for sg in child_subgraphs
            if sg.has_attr("device") and
               sg.get_attr("device").upper() == "DPU"
        ]

        if dpu_sgs:
            print(f"  Found {len(dpu_sgs)} DPU subgraph(s):")
            for sg in dpu_sgs:
                inputs  = sg.get_input_tensors()
                outputs = sg.get_output_tensors()
                print(f"\n    [DPU] {sg.get_name()}")
                print(f"      Op count       : {sg.get_op_num()}")
                print(f"      Input tensors ({len(inputs)}):")
                for t in inputs:
                    print(f"        - {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
                print(f"      Output tensors ({len(outputs)}):")
                for t in outputs:
                    print(f"        - {t.get_name()}  "
                          f"shape={format_shape(t.get_shape())}  "
                          f"dtype={t.get_data_type()}")
        else:
            print("  No DPU subgraphs found "
                  "(the model may be quantized but not yet compiled).")
    except Exception as e:
        print(f"  Could not retrieve DPU subgraph information: {e}")


def show_io_tensors(graph) -> None:
    """
    Display the graph-level input and output tensors.

    These are derived from the root subgraph and represent the overall
    model interface (e.g. image input batch and class-score output).
    """
    section("Graph Input / Output Tensors")

    try:
        root           = graph.get_root_subgraph()
        input_tensors  = root.get_input_tensors()
        output_tensors = root.get_output_tensors()

        print(f"\n  Input tensors ({len(input_tensors)}):")
        for t in input_tensors:
            print(f"    - {t.get_name()}")
            print(f"      shape={format_shape(t.get_shape())}  "
                  f"dtype={t.get_data_type()}")

        print(f"\n  Output tensors ({len(output_tensors)}):")
        for t in output_tensors:
            print(f"    - {t.get_name()}")
            print(f"      shape={format_shape(t.get_shape())}  "
                  f"dtype={t.get_data_type()}")
    except Exception as e:
        print(f"  Could not retrieve IO tensors: {e}")


# ==============================================================================
# Graphviz DOT export
# ==============================================================================

def export_dot(graph, dot_path: str) -> None:
    """
    Export the computation graph as a Graphviz DOT file.

    Each node represents one Op (labelled with name, type, and output shape).
    Directed edges represent data flow between ops.

    The resulting DOT file can be rendered with:
        dot -Tpng output.dot -o graph.png
        dot -Tsvg output.dot -o graph.svg

    Args:
        graph    : xir.Graph object.
        dot_path : Destination file path for the DOT output.
    """
    ops = graph.topological_sort()

    def node_id(name: str) -> str:
        """Wrap a node name in quotes, escaping any internal quote characters."""
        return '"' + name.replace('"', '\\"') + '"'

    lines = [
        "digraph xmodel {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fillcolor="#AED6F1", '
        'fontname="Arial", fontsize=10];',
        "  edge [fontsize=9];",
        "",
    ]

    # Declare all nodes
    for op in ops:
        label = f"{op.get_name()}\\n[{op.get_type()}]"
        out_t = op.get_output_tensor()
        if out_t:
            label += f"\\n{format_shape(out_t.get_shape())}"
        lines.append(f"  {node_id(op.get_name())} [label={node_id(label)}];")

    lines.append("")

    # Declare all edges
    for op in ops:
        try:
            in_ops_dict = op.get_input_ops()
            for _, in_op_list in in_ops_dict.items():
                for in_op in in_op_list:
                    lines.append(
                        f"  {node_id(in_op.get_name())} -> "
                        f"{node_id(op.get_name())};"
                    )
        except Exception:
            pass

    lines.append("}")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [DOT] Exported to : {dot_path}")
    print(f"  [DOT] Render PNG  : dot -Tpng {dot_path} -o graph.png")
    print(f"  [DOT] Render SVG  : dot -Tsvg {dot_path} -o graph.svg")


# ==============================================================================
# Command-line interface
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vitis-AI xmodel Inspector (PyXIR / xir)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic summary (graph info + subgraph tree + IO tensors)
  python3 view_xmodel.py resnet50.xmodel

  # List all ops in topological order
  python3 view_xmodel.py resnet50.xmodel --ops

  # Full detail: ops + attributes + tensor info
  python3 view_xmodel.py resnet50.xmodel --ops --attrs --tensors

  # Export a Graphviz DOT diagram
  python3 view_xmodel.py resnet50.xmodel --dot output.dot
        """
    )
    parser.add_argument(
        "xmodel",
        help="Path to the .xmodel file to inspect."
    )
    parser.add_argument(
        "--ops",
        action="store_true",
        help="List all ops in topological order with shape/dtype information."
    )
    parser.add_argument(
        "--attrs",
        action="store_true",
        help="Print all key-value attributes for each op / subgraph (use with --ops)."
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="Print detailed input tensor information for each op (use with --ops)."
    )
    parser.add_argument(
        "--dot",
        metavar="FILE",
        help="Export the computation graph to a Graphviz DOT file."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{SEP_MAJOR}")
    print(f"  Vitis-AI xmodel Inspector  (PyXIR / xir)")
    print(f"{SEP_MAJOR}")
    print(f"  Loading: {args.xmodel}")

    # ── Load the xmodel ───────────────────────────────────────────────────────
    try:
        graph = xir.Graph.deserialize(args.xmodel)
    except Exception as e:
        print(f"\n[ERROR] Failed to load xmodel: {e}")
        sys.exit(1)

    print(f"  Loaded successfully.")

    # ── Run selected inspection modules ──────────────────────────────────────
    show_graph_summary(graph)
    show_io_tensors(graph)
    show_subgraphs(graph, show_attrs=args.attrs)

    if args.ops:
        show_ops(graph, show_attrs=args.attrs, show_tensors=args.tensors)

    if args.dot:
        export_dot(graph, args.dot)

    print(f"\n{SEP_MAJOR}")
    print(f"  Done.")
    print(f"{SEP_MAJOR}\n")


if __name__ == "__main__":
    main()