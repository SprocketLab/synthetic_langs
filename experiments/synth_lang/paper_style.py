"""Shared figure style: bold sans-serif text, flat fills with black outlines, square markers, no grid.

    from paper_style import apply, COLOR, LAYOUT, ACL_COL
"""
import matplotlib.pyplot as plt

ACL_COL = 3.03          # one ACL column, inches

# palette sampled from Figure 1
BLUE, ORANGE, YELLOW, PURPLE, RED = "#7fcaf9", "#e47d55", "#e4c255", "#cc55e4", "#e45555"
GREY, DARK = "#d9d9d9", "#4a4a4a"
GUIDE = "#b5b5b5"

# layer family -> colour: attention only, contains Mamba, contains DeltaProduct
FAMILY = {"T": DARK, "S": BLUE, "D": ORANGE}

LAYOUT = {
    "pure_tf": "TTTTTTTT", "4s4t_alternate": "STSTSTST", "6s2t_start": "TTSSSSSS",
    "6s2t_even": "SSSTSSST", "7s1t_mid": "SSSTSSSS", "pure_ssm": "SSSSSSSS",
    "pure_dp": "DDDDDDDD", "4d4t_alternate": "DTDTDTDT", "6d2t_start": "TTDDDDDD",
    "6s2t_end": "SSSSSSTT", "4s4t_ssm_first": "SSSSTTTT", "4s4t_tf_first": "TTTTSSSS",
    "dsts_alternate": "DSTSDSTS",
}


# one colour per architecture, same in every figure
COLOR = {
    "pure_tf": "#8c9a2c", "4s4t_alternate": BLUE, "6s2t_start": YELLOW, "6s2t_even": PURPLE,
    "7s1t_mid": "#5cc98a", "pure_ssm": "#2f6fb3", "pure_dp": RED, "4d4t_alternate": ORANGE,
    "6d2t_start": "#ef9fcf", "6s2t_end": "#49b8b0", "4s4t_ssm_first": "#9b7653",
    "4s4t_tf_first": "#a3a3a3", "dsts_alternate": "#8f7ae6",
}


def color(cfg):
    return COLOR[cfg]


def family_color(cfg):
    lay = LAYOUT[cfg]
    return FAMILY["D"] if "D" in lay else FAMILY["S"] if "S" in lay else FAMILY["T"]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "axes.edgecolor": "black", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


MARKER = dict(marker="s", markersize=5.5, markeredgecolor="black", markeredgewidth=0.9)
