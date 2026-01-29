# src/netlist.py
def generate_netlist(components):
    lines = ["* Auto-generated netlist", ""]
    for i, comp in enumerate(components):
        lines.append(f"X{i} UNKNOWN")
    lines.append(".end")
    return "\n".join(lines)
