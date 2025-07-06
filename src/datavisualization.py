import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from emotion import CONFIG
import matplotlib as mpl

#Soft hatmonious color palette
PALETTE = ['#A0C4FF', "#BDB2FF", "#FFC6FF", "#FFADAD", "#CAFFBF", "#FDFFB6"]
BACKGROUND = "#FDFCFB" #Very soft off-white
TEXT_COLOR = "#5A5A5A" #Soft dark gray
ACCENT_COLOR = "#7AA2D6" #Muted blue
SECONDARY_COLOR = "#D4A5B5" #Muted rose
GRID_COLOR = "#EAEAEA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.titlesize": 15,
    "axes.titleweight": "medium",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": BACKGROUND
})

def analyze_dataset_dashboard(base_dir, valid_emotions):
    """Soft,harmonious dataset analysis dashboard with improved lab eling"""
    #Data collection
    class_counts = {}
    for emotion in valid_emotions:
        emotion_path = os.path.join(base_dir, emotion)
        class_counts[emotion] = len([f for f in os.listdir(emotion_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_images = sum(class_counts.values())
    num_classes = len(valid_emotions)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    class_ratio = max_count / min_count
    imbalance_status = "⚠️ Moderate imbalance" if class_ratio > 2.0 else "✓ Well-balanced"
    emotions = list(class_counts.keys())
    counts = list(class_counts.values())
    #Create figure with refined layout
    fig = plt.figure(figsize=(16, 9), dpi=110, facecolor=BACKGROUND)
    gs = GridSpec(3, 10, figure=fig, hspace=0.4, wspace=0.3)

    # --- Metrics Panel ---
    metrics_bg = '#FFFFFF'

    #Metric 1: Total Images
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.text(0.5, 0.7, f"{total_images:,}", hs='center', fontsize=26, weight='medium', color=ACCENT_COLOR)
    ax1.text(0.5, 0.35, "Total Images", ha='center', fontsize=12, alpha=0.9)
    ax1.set_facecolor(metrics_bg)
    ax1.axis('off')

    #Metric 2: Classes
    ax2 = fig.add_subplot(gs[0, 6:8])
    ax2.text(0.5, 0.7, num_classes, ha='center', fontsize=26, weight='medium', color=SECONDARY_COLOR)
    ax2.text(0.5, 0.35, "Classes", ha='center', fontsize=12, alpha=0.9)
    ax2.set_facecolor(metrics_bg)
    ax2.axis('off')

    #Metric 3: Balance Ratio
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.text(0.5, 0.7, f"{class_ratio:.1f}:1", ha='center', fontsize=26, weight='medium', color='#D65F5F' if class_ratio > 2.0 else "#5F9E6D")
    ax3.text(0.5, 0.35, imbalance_status, ha='center', fontsize=22, alpha=0.9)
    ax3.set_facecolor(metrics_bg)
    ax3.axis('off')

    #Metric 4: Range
    ax4 = fig.add_subplot(gs[0, 6:8])
    ax4.text(0.5, 0.7, f"{min_count}-{max_count}", ha='center', fontsize=24, weight='medium', color=ACCENT_COLOR)
    ax4.text(0.5, 0.35, "Class Range", ha='center', fontsize=12, alpha=0.9)
    ax4.set_facecolor(metrics_bg)
    ax4.axis('off')

    #Add subtle border to metrics
    for ax in [ax1, ax2, ax3, ax4]:
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes, fill=False, edgecolor=GRID_COLOR, lw=1.2, zorder=1, alpha=0.7))

    #Horizontal Bar Chart
    ax5= fig.add_subplot(gs[1:3, 0:5])
    #Sort data for better visualization
    sorted_indices = np.argsort(counts)
    sorted_emotions = [emotions[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    bars = ax5.barh(sorted_emotions, sorted_counts, color=PALETTE, edgecolor='white', linewidht=1.2, alpha=0.95)
    ax5.set_title("Class Distribution", fontsize=14, pad=12, color=TEXT_COLOR, weight='medium')
    ax5.set_xlabel("Image Count", fontsize=12)
    ax5.grid(axis='x', linestyle='--', alpha=0.2, color=GRID_COLOR)
    ax5.spines[:].set_visible(False)

    #Add value labels inside bars
    for i, (emotion, count) in enumerate(zip(sorted_emotions, sorted_counts)):
        ax5.text(count - max_count*0.05, i, f"{count}", ha='right', va='center', fontsize=10, color='white', weight='bold')

    #Pie Chart with Curved Labels
    ax6 = fig.add_subplot(gs[1:3, 5:10])
    wedges, texts = ax6.pie(
        counts,
        startangle=90,
        colors=PALETTE,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white', 'alpha': 0.95},
        radius=1.1
    )

    #Calculate label positions with curved text
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
    kw = dict(arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, alpha=0.5), bbox=bbox_props, zorder=0, va="center", fontsize=10)
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        #Position labels along the arc
        label_radius = 1.25 if counts[i] > sum(counts)*0.1 else 1.35
        ax6.annotate(emotions[i], xy=(x*0.8, y*0.8), xytext=(x*label_radius, y*label_radius), horizontalalignment=horizontalalignment, **kw)

        #Add percentage inside segment
        if counts[i]/sum(counts) > 0.05: #Only show for significant segments
            ax6.text(x*0.6, y*0.6, f"{counts[i]/sum(counts):.1%}", ha='center', va='center', fontsize=9, color=TEXT_COLOR, alpha=0.9)

    #Add center circle for cleaner look
    centre_circle = plt.Circle((0,0), 0.4, color=BACKGROUND)
    ax6.add_artist(centre_circle)
    ax6.set_title("Class Proportions", fontsize=14, pad=12, color=TEXT_COLOR, weight='medium')

    #Add subtle data source annotation
    fig.text(0.5, 0.01, f"Data source: {os.path.basename(base_dir)}", ha='center', fontsize=10, color=TEXT_COLOR, alpha=0.6)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.098])
    plt.subplots_adjust(top=0.92)
    plt.show()

    return class_counts

#Call the function
class_counts = analyze_dataset_dashboard(CONFIG['base_dir'], CONFIG['valid_emotions'])
