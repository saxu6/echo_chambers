#!/usr/bin/env python3
"""
Generate missing paper figures:
- Figure 1: Conceptual neutrosophic framework diagram
- Figure 2: 4x3 grid of turn-wise EchoIndex and LocalDrift

Matches the exact style from hedge/dissonant experiment figures.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# Set matplotlib style to match sample
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model and topic mappings
MODELS = ["ChatGPT", "Claude", "Gemini", "Deepseek"]
TOPICS = ["menstrual", "labour", "immigration"]

# Exact colors from sample image
RED = '#e74c3c'
GREEN = '#2ecc71'
PURPLE = '#9b59b6'
ORANGE = '#f39c12'

MODEL_COLORS = {
    'ChatGPT': RED,
    'Claude': GREEN,
    'Gemini': PURPLE,
    'Deepseek': ORANGE
}


def generate_figure1():
    """
    Figure 1: The neutrosophic dynamical framework.
    (A) Three dimensions (F, T, I)
    (B) EchoIndex formula
    (C) LocalDrift formula
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === Panel A: Three dimensions ===
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(A) Turn-Level Dimensions', fontsize=12, pad=20)

    # Three boxes for F, T, I
    box_width = 2.2
    box_height = 3
    box_y = 4.5
    positions = [1.2, 4.0, 6.8]
    labels = ['$F_t$', '$T_t$', '$I_t$']
    subtitles = ['Stance\nReinforcement', 'Perspective\nDiversity', 'Expressed\nIndeterminacy']
    colors = [RED, PURPLE, GREEN]

    for pos, label, subtitle, color in zip(positions, labels, subtitles, colors):
        rect = Rectangle((pos, box_y), box_width, box_height,
                         facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(pos + box_width/2, box_y + box_height/2 + 0.3, label,
                fontsize=18, ha='center', va='center', color='white', fontweight='bold')
        ax.text(pos + box_width/2, box_y + box_height/2 - 0.7, subtitle,
                fontsize=8, ha='center', va='center', color='white')

    # Turn indicator box
    ax.add_patch(Rectangle((3.5, 8.5), 3, 1, facecolor='white', edgecolor='black', linewidth=1))
    ax.text(5, 9, 'Turn $t$', fontsize=11, ha='center', va='center')

    # Arrows
    for pos in positions:
        ax.annotate('', xy=(pos + box_width/2, box_y + box_height),
                   xytext=(5, 8.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # === Panel B: EchoIndex formula ===
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(B) EchoIndex Composite', fontsize=12, pad=20)

    # Formula box
    ax.add_patch(Rectangle((0.5, 4), 9, 3, facecolor='#f5f5f5', edgecolor='black', linewidth=1))
    ax.text(5, 5.8, r'$EC_t = F_t + (1 - T_t) + (1 - I_t)$',
            fontsize=14, ha='center', va='center')
    ax.text(5, 4.5, 'Range: [0, 3]', fontsize=10, ha='center', va='center', color='gray')

    # Interpretation
    ax.annotate('', xy=(2, 3.8), xytext=(2, 2.5),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax.text(2, 2, 'Lower = Less\nEcho Chamber', fontsize=9, ha='center', va='top', color=GREEN)

    ax.annotate('', xy=(8, 3.8), xytext=(8, 2.5),
               arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    ax.text(8, 2, 'Higher = More\nEcho Chamber', fontsize=9, ha='center', va='top', color=RED)

    ax.text(5, 8, 'Combines agreement, low diversity,\nand low uncertainty into single metric',
            fontsize=10, ha='center', va='center', style='italic', color='gray')

    # === Panel C: LocalDrift formula ===
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(C) LocalDrift Dynamics', fontsize=12, pad=20)

    # Formula box
    ax.add_patch(Rectangle((0.5, 4), 9, 3, facecolor='#f5f5f5', edgecolor='black', linewidth=1))
    ax.text(5, 5.8, r'$d_t = \Delta F_t - \Delta T_t - \Delta I_t$',
            fontsize=14, ha='center', va='center')
    ax.text(5, 4.5, 'Turn-to-turn change', fontsize=10, ha='center', va='center', color='gray')

    # Interpretation
    ax.annotate('', xy=(2, 3.8), xytext=(2, 2.5),
               arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    ax.text(2, 2, '$d_t > 0$\nForming\nEcho Chamber', fontsize=9, ha='center', va='top', color=RED)

    ax.annotate('', xy=(8, 3.8), xytext=(8, 2.5),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax.text(8, 2, '$d_t < 0$\nRecovering\n(Breaking Echo)', fontsize=9, ha='center', va='top', color=GREEN)

    ax.text(5, 8, 'Tracks whether conversation\nis drifting toward or away from echo',
            fontsize=10, ha='center', va='center', style='italic', color='gray')

    plt.tight_layout()
    path = OUTPUT_DIR / "figure1_framework.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def load_conversation_data():
    """Load turn-wise data from raw_turns.jsonl."""
    data = {}
    jsonl_path = PROJECT_DIR / "outputs" / "raw_turns.jsonl"

    if jsonl_path.exists():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                turn = json.loads(line.strip())
                model = turn.get('model', '')
                topic = turn.get('topic', '')

                # Normalize model names
                model_normalized = model.lower()
                if 'chatgpt' in model_normalized:
                    model = 'ChatGPT'
                elif 'claude' in model_normalized:
                    model = 'Claude'
                elif 'gemini' in model_normalized:
                    model = 'Gemini'
                elif 'deepseek' in model_normalized:
                    model = 'Deepseek'

                # Normalize topic names
                topic_normalized = topic.lower()
                if 'menstrual' in topic_normalized:
                    topic = 'menstrual'
                elif 'labour' in topic_normalized or 'labor' in topic_normalized:
                    topic = 'labour'
                elif 'immigration' in topic_normalized:
                    topic = 'immigration'

                key = (model, topic)
                if key not in data:
                    data[key] = {'turns': [], 'EC': [], 'd': []}

                data[key]['turns'].append(turn.get('turn', len(data[key]['turns'])+1))
                data[key]['EC'].append(turn.get('EC', turn.get('EchoIndex', 0)))
                data[key]['d'].append(turn.get('d', turn.get('LocalDrift', 0)))

    return data


def generate_figure2():
    """
    Figure 2: 4x3 grid showing turn-wise EchoIndex and LocalDrift.
    Rows: ChatGPT, Claude, Gemini, DeepSeek
    Columns: menstrual leave, labour laws, immigration
    """
    data = load_conversation_data()

    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.10, hspace=0.4, wspace=0.4)

    # Column headers
    topic_labels = ['Menstrual Leave', 'Labour Laws', 'Immigration']
    for col, topic_label in enumerate(topic_labels):
        axes[0, col].set_title(topic_label, fontsize=11, pad=10)

    for row, model in enumerate(MODELS):
        # Row labels
        axes[row, 0].set_ylabel(model, fontsize=11, rotation=90, labelpad=10)

        for col, topic in enumerate(TOPICS):
            ax = axes[row, col]
            key = (model, topic)

            if key in data and len(data[key]['turns']) > 0:
                turns = data[key]['turns']
                ec_values = data[key]['EC']
                d_values = data[key]['d']

                color = MODEL_COLORS.get(model, 'black')

                # Plot EchoIndex line
                ax.plot(turns, ec_values, 'o-', color=color, linewidth=2, markersize=6)

                # Baseline at 2.0
                ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

                ax.set_ylim(0, 3.5)
                ax.set_xlabel('Turn', fontsize=9)
                ax.set_xticks(turns)
                ax.tick_params(labelsize=8)

                # LocalDrift as secondary axis
                ax2 = ax.twinx()
                bar_colors = [RED if d > 0 else GREEN for d in d_values]
                ax2.bar(turns, d_values, color=bar_colors, alpha=0.4, width=0.5)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                ax2.set_ylim(-2, 2)
                ax2.tick_params(labelsize=8, colors='gray')

            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')

    # Y-axis labels
    fig.text(0.04, 0.5, 'EchoIndex', va='center', rotation='vertical', fontsize=11)
    fig.text(0.96, 0.5, 'LocalDrift', va='center', rotation='vertical', fontsize=11, color='gray')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', marker='o', linestyle='-', linewidth=2, markersize=6, label='EchoIndex'),
        mpatches.Patch(facecolor=RED, alpha=0.4, label='LocalDrift > 0'),
        mpatches.Patch(facecolor=GREEN, alpha=0.4, label='LocalDrift < 0'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='EC = 2.0')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=False)

    path = OUTPUT_DIR / "figure2_grid_12conversations.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    print("Generating paper figures...")
    print("=" * 50)

    print("\n[1/2] Figure 1: Neutrosophic Framework...")
    fig1_path = generate_figure1()

    print("\n[2/2] Figure 2: 4x3 Grid...")
    fig2_path = generate_figure2()

    print("\n" + "=" * 50)
    print("COMPLETE!")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
