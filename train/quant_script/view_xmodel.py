#!/usr/bin/env python3
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


def tensor_name(t) -> str:
    """
    Return the name of an xir.Tensor object.

    Vitis-AI releases differ in their Python binding:
      - Older builds expose  t.get_name()
      - Newer builds expose  t.name  (read-only property)
    This wrapper tries both so the script works across versions.
    """
    if hasattr(t, "get_name"):
        try:
            return t.get_name()
        except Exception:
            pass
    if hasattr(t, "name"):
        return t.name
    return str(t)


def tensor_shape(t) -> str:
    """
    Return the formatted shape string of an xir.Tensor object.

    Tries get_shape() → dims → shape in order, to cover all known
    XIR Python binding versions.
    """
    for attr in ("get_shape", "dims", "shape"):
        try:
            val = getattr(t, attr)
            result = val() if callable(val) else val
            if result is not None:
                return format_shape(result)
        except Exception:
            pass
    return "unknown"


def tensor_dtype(t) -> str:
    """
    Return the data-type string of an xir.Tensor object.

    Tries get_data_type() → data_type → dtype in order.
    """
    for attr in ("get_data_type", "data_type", "dtype"):
        try:
            val = getattr(t, attr)
            result = val() if callable(val) else val
            if result is not None:
                return str(result)
        except Exception:
            pass
    return "unknown"


def get_ops_sorted(graph) -> list:
    """
    Return all ops in topological order, working across XIR versions.

    XIR API differences:
      - graph.get_ops()                  – unordered, available on xir.Graph
      - root_subgraph.topological_sort() – ordered, available on xir.Subgraph
      - graph.topological_sort()         – only exists in some older builds

    We try each variant in order and fall back gracefully.
    """
    # Preferred: topological order from root subgraph
    try:
        root = graph.get_root_subgraph()
        return list(root.topological_sort())
    except Exception:
        pass

    # Fallback: topological_sort directly on graph (older XIR)
    try:
        return list(graph.topological_sort())
    except Exception:
        pass

    # Last resort: unordered op list from graph
    try:
        return list(graph.get_ops())
    except Exception:
        return []


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
    ops = get_ops_sorted(graph)

    for i, op in enumerate(ops):
        op_name    = op.get_name()
        op_type    = op.get_type()
        out_tensor = op.get_output_tensor()

        shape_str = tensor_shape(out_tensor) if out_tensor else "-"
        dtype_str = tensor_dtype(out_tensor) if out_tensor else "-"

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
                    print(f"           in_tensor: {tensor_name(t)}  "
                          f"shape={tensor_shape(t)}  "
                          f"dtype={tensor_dtype(t)}")
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

    def _get_device(sg) -> str:
        """
        Robustly retrieve the device assignment of a subgraph.

        Different Vitis-AI / XIR versions store the device under
        slightly different attribute names or as a plain Python
        property.  We try every known variant in priority order:
          1. get_attr("device")          – standard compiled xmodel
          2. get_attr("device_core_id")  – some DPU split flows
          3. sg.device                   – property-style binding
          4. Scan all attr names for any key containing "device"
        """
        # Variant 1 & 2: named attributes
        for key in ("device", "device_core_id"):
            try:
                if sg.has_attr(key):
                    val = sg.get_attr(key)
                    if val not in (None, ""):
                        return str(val)
            except Exception:
                pass

        # Variant 3: property-style binding
        try:
            val = sg.device                 # type: ignore[attr-defined]
            if val not in (None, ""):
                return str(val)
        except Exception:
            pass

        # Variant 4: scan all attribute names for anything device-like
        try:
            for k in sg.get_attr_names():
                if "device" in k.lower():
                    try:
                        val = sg.get_attr(k)
                        if val not in (None, ""):
                            return f"{k}={val}"
                    except Exception:
                        pass
        except Exception:
            pass

        return "-"

    def _print_subgraph(sg, depth: int = 0, inherited_device: str = "-") -> None:
        """
        Recursively print the subgraph tree.

        Leaf subgraphs inside a DPU partition do not carry their own
        'device' attribute; the attribute is only set on the direct
        children of root.  We pass the parent's device value down so
        every node in the tree shows a meaningful device label.
        """
        indent    = "  " * depth
        name      = sg.get_name()
        op_num    = sg.get_op_num()
        child_num = len(sg.get_children())

        own_device = _get_device(sg)
        # Use own device if found, otherwise inherit from parent
        device = own_device if own_device != "-" else inherited_device

        # Mark inherited values so the user knows it came from the parent
        device_label = device if own_device != "-" else f"{device} (inherited)"

        print(f"{indent}> {name}")
        print(f"{indent}  op_num={op_num}  children={child_num}  device={device_label}")

        if show_attrs:
            attrs = safe_attrs(sg)
            if attrs:
                for k, v in attrs.items():
                    if "device" not in k.lower():
                        print(f"{indent}  attr: {k} = {v}")

        for child in sg.get_children():
            _print_subgraph(child, depth + 1, inherited_device=device)

    _print_subgraph(root)

    # Summarise DPU subgraphs specifically
    subsection("DPU Subgraph Summary")
    try:
        child_subgraphs = root.toposort_child_subgraph()
        dpu_sgs = [
            sg for sg in child_subgraphs
            if "DPU" in _get_device(sg).upper()
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
                    print(f"        - {tensor_name(t)}  "
                          f"shape={tensor_shape(t)}  "
                          f"dtype={tensor_dtype(t)}")
                print(f"      Output tensors ({len(outputs)}):")
                for t in outputs:
                    print(f"        - {tensor_name(t)}  "
                          f"shape={tensor_shape(t)}  "
                          f"dtype={tensor_dtype(t)}")
        else:
            print("  No DPU subgraphs found "
                  "(the model may be quantized but not yet compiled).")
            # Diagnostic: dump all attribute names found on child subgraphs
            # so the user can see what is actually present.
            print("\n  [DIAG] Attribute names found on each child subgraph:")
            for sg in child_subgraphs[:10]:   # cap at 10 to avoid flooding
                try:
                    attr_names = list(sg.get_attr_names())
                except Exception:
                    attr_names = []
                print(f"    {sg.get_name()}")
                print(f"      attrs: {attr_names if attr_names else '(none)'}")
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
            print(f"    - {tensor_name(t)}")
            print(f"      shape={tensor_shape(t)}  "
                  f"dtype={tensor_dtype(t)}")

        print(f"\n  Output tensors ({len(output_tensors)}):")
        for t in output_tensors:
            print(f"    - {tensor_name(t)}")
            print(f"      shape={tensor_shape(t)}  "
                  f"dtype={tensor_dtype(t)}")
    except Exception as e:
        print(f"  Could not retrieve IO tensors: {e}")


# ==============================================================================
# Graphviz DOT export
# ==============================================================================

def export_dot(graph, dot_path: str, rankdir: str = "LR") -> None:
    """
    Export the computation graph as a Graphviz DOT file with device clusters.

    Ops are grouped into labelled, colour-coded cluster boxes according to
    which device their parent subgraph is assigned to:

        DPU   -> blue   (#AED6F1)
        CPU   -> orange (#FAD7A0)
        USER  -> green  (#A9DFBF)
        other -> grey   (#D5D8DC)

    Each node is labelled with:  op_name / [op_type] / output_shape

    Render with:
        dot -Tsvg output.dot -o graph.svg   (recommended for large graphs)
        dot -Tpng output.dot -o graph.png

    Args:
        graph    : xir.Graph object.
        dot_path : Destination file path for the DOT output.
    """

    # ── device colour palette ─────────────────────────────────────────────────
    DEVICE_STYLE: dict = {
        "DPU":     {"fillcolor": "#AED6F1", "color": "#1A5276"},  # blue
        "CPU":     {"fillcolor": "#FAD7A0", "color": "#784212"},  # orange
        "USER":    {"fillcolor": "#A9DFBF", "color": "#1E8449"},  # green
        "DEFAULT": {"fillcolor": "#D5D8DC", "color": "#616A6B"},  # grey
    }

    def node_id(name: str) -> str:
        """Escape and quote a name for use as a DOT node identifier."""
        return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def get_device_key(device_str: str) -> str:
        """Normalise a device string to one of the palette keys."""
        d = device_str.upper().split("(")[0].strip()   # strip "(inherited)"
        for key in ("DPU", "CPU", "USER"):
            if key in d:
                return key
        return "DEFAULT"

    # ── build op -> device mapping via subgraph tree ──────────────────────────
    op_device: dict = {}   # op_name -> device key

    def _walk(sg, inherited: str = "DEFAULT") -> None:
        """Walk the subgraph tree and record each op's effective device."""
        # Determine this subgraph's own device
        own = "DEFAULT"
        for key in ("device", "device_core_id"):
            try:
                if sg.has_attr(key):
                    val = sg.get_attr(key)
                    if val not in (None, ""):
                        own = get_device_key(str(val))
                        break
            except Exception:
                pass
        if own == "DEFAULT":
            try:
                val = sg.device          # type: ignore[attr-defined]
                if val not in (None, ""):
                    own = get_device_key(str(val))
            except Exception:
                pass

        effective = own if own != "DEFAULT" else inherited

        children = sg.get_children()
        if children:
            for child in children:
                _walk(child, inherited=effective)
        else:
            # Leaf subgraph: assign its ops to the effective device
            try:
                for op in sg.get_ops():
                    op_device[op.get_name()] = effective
            except Exception:
                pass

    try:
        _walk(graph.get_root_subgraph())
    except Exception:
        pass

    # Ops that were not reached by the subgraph walk keep DEFAULT
    all_ops = get_ops_sorted(graph)
    for op in all_ops:
        op_device.setdefault(op.get_name(), "DEFAULT")

    # ── group ops by device, preserving topological order within each group ───
    from collections import defaultdict
    device_ops: dict = defaultdict(list)
    for op in all_ops:
        device_ops[op_device[op.get_name()]].append(op)

    # ── build DOT source ──────────────────────────────────────────────────────
    lines = [
        "digraph xmodel {",
        f"  rankdir={rankdir};",
        '  graph [fontname="Arial", fontsize=12, bgcolor="white"];',
        '  node  [fontname="Arial", fontsize=9, style=filled, shape=box];',
        '  edge  [fontsize=8, color="#555555"];',
        "",
    ]

    # One cluster per device
    for cluster_idx, (dev_key, ops) in enumerate(device_ops.items()):
        style  = DEVICE_STYLE.get(dev_key, DEVICE_STYLE["DEFAULT"])
        fc     = style["fillcolor"]
        bc     = style["color"]
        label  = dev_key if dev_key != "DEFAULT" else "UNKNOWN"

        lines.append(f"  subgraph cluster_{cluster_idx} {{")
        lines.append(f'    label     = "Device: {label}";')
        lines.append(f'    style     = filled;')
        lines.append(f'    fillcolor = "{fc}44";')   # transparent fill for cluster bg
        lines.append(f'    color     = "{bc}";')
        lines.append(f'    fontcolor = "{bc}";')
        lines.append(f'    fontname  = "Arial Bold";')
        lines.append(f'    fontsize  = 11;')
        lines.append("")

        for op in ops:
            op_name = op.get_name()
            op_type = op.get_type()
            out_t   = op.get_output_tensor()
            shape   = tensor_shape(out_t) if out_t else "-"

            # Short display name: last segment after "__"
            short_name = op_name.split("__")[-1] if "__" in op_name else op_name

            # Use HTML-like label so <BR/> produces real line breaks in SVG/PNG.
            # Characters that need escaping inside HTML labels: & < >
            def _esc(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            html_label = (
                f'<<FONT POINT-SIZE="9"><B>{_esc(short_name)}</B><BR/>'
                f'<FONT COLOR="#444444">[{_esc(op_type)}]</FONT><BR/>'
                f'<FONT COLOR="#666666">{_esc(shape)}</FONT></FONT>>'
            )

            lines.append(
                f'    {node_id(op_name)} ['
                f'label={html_label}, '
                f'fillcolor="{fc}", '
                f'color="{bc}"'
                f'];'
            )

        lines.append("  }")
        lines.append("")

    # All edges (drawn outside clusters so cross-device arrows render correctly)
    lines.append("  // Edges")
    for op in all_ops:
        try:
            in_ops_dict = op.get_input_ops()
            for _, in_op_list in in_ops_dict.items():
                for in_op in in_op_list:
                    src_dev = op_device.get(in_op.get_name(), "DEFAULT")
                    dst_dev = op_device.get(op.get_name(),    "DEFAULT")
                    # Cross-device edges get a distinct colour
                    edge_color = "#E74C3C" if src_dev != dst_dev else "#555555"
                    lines.append(
                        f'  {node_id(in_op.get_name())} -> {node_id(op.get_name())} '
                        f'[color="{edge_color}"];'
                    )
        except Exception:
            pass

    lines.append("}")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [DOT] Exported to : {dot_path}")
    print(f"  [DOT] Render SVG  : dot -Tsvg {dot_path} -o graph.svg")
    print(f"  [DOT] Render PNG  : dot -Tpng {dot_path} -o graph.png")
    print(f"  [DOT] Legend      : blue=DPU  orange=CPU  green=USER  "
          f"grey=unknown  red-edge=cross-device")


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
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB", "RL", "BT"],
        default="LR",
        help="Graph layout direction: LR=landscape (default), TB=portrait, "
             "RL=right-to-left, BT=bottom-to-top."
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
        export_dot(graph, args.dot, rankdir=args.rankdir)

    print(f"\n{SEP_MAJOR}")
    print(f"  Done.")
    print(f"{SEP_MAJOR}\n")


if __name__ == "__main__":
    main()