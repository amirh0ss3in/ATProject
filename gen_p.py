import matplotlib.pyplot as plt
import numpy as np
import math

# Set style for clean presentation graphics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def plot_permutation_split(V=5):
    """Generates a diagram explaining the Permutation Split (Alternative A)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Total space dimensions
    total_perms = math.factorial(V)
    train_perms = int(total_perms * 0.8)
    test_perms = total_perms - train_perms
    
    # Draw bars
    ax.barh(0, total_perms, color='#e0e0e0', height=0.4, label='Total Permutations (V!)')
    ax.barh(-0.6, train_perms, color='#4a90e2', height=0.4, label=f'Train Set ({train_perms} perms)')
    ax.barh(-1.2, test_perms, color='#f5a623', height=0.4, label=f'Test Set ({test_perms} perms)')
    
    # Text annotations
    ax.text(total_perms/2, 0, f"All Possible Mappings (120 unique permutations)", ha='center', va='center', color='black', fontweight='bold')
    ax.text(train_perms/2, -0.6, "Seen Mappings (80%) - Used for Weights Optimization", ha='center', va='center', color='white', fontweight='bold')
    ax.text(test_perms/2, -1.2, "Unseen Mappings (20%) - Evaluation Only", ha='center', va='center', color='black', fontweight='bold')
    
    # Vocabulary explanation text box
    info_text = (
        "Key Concept: Vocabulary Sharing\n"
        "• Vocabularies X and Y are fully shared: {1, 2, 3, 4, 5}\n"
        "• Embeddings for ALL tokens are trained.\n"
        "• ONLY the specific mapping functions (permutations) are unseen."
    )
    ax.text(total_perms * 0.45, -1.9, info_text, bbox=dict(facecolor='#f9f9f9', edgecolor='#ccc', boxstyle='round,pad=1'), fontsize=10, linespacing=1.4)
    
    ax.set_title("Alternative A: Permutation Space Partitioning (V = 5)", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlim(-5, total_perms + 5)
    ax.set_ylim(-2.5, 0.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('1_permutation_split.png', dpi=300)
    plt.close()

def plot_causal_attention_mask(V=5):
    """Generates a diagram visualizing the Causal Attention Mask."""
    seq_len = 2 * V - 1
    labels = []
    for i in range(1, V + 1):
        labels.append(f"x{i}")
        if i < V:
            labels.append(f"y{i}")
            
    # Create causal mask matrix (1 for allowed, 0 for blocked)
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # Custom colormap: Gray for blocked, Soft Teal for allowed
    cmap = plt.cm.colors.ListedColormap(['#f0f0f0', '#80cbc4'])
    ax.imshow(mask, cmap=cmap, aspect='equal')
    
    # Ticks and labels
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    
    # Add grid lines to separate cells clearly
    ax.set_xticks(np.arange(seq_len) - 0.5, minor=True)
    ax.set_yticks(np.arange(seq_len) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Annotate allowed/blocked regions
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                ax.text(j, i, "Attend", ha='center', va='center', fontsize=8, color='#004d40', fontweight='bold')
            else:
                ax.text(j, i, "Mask", ha='center', va='center', fontsize=8, color='#9e9e9e')
                
    ax.set_title("Causal Attention Mask (Sequence Look-Back)", fontsize=13, pad=15, fontweight='bold')
    ax.set_xlabel("Keys / Values (Past)", fontsize=11, labelpad=10)
    ax.set_ylabel("Queries (Present)", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    plt.savefig('2_causal_mask.png', dpi=300)
    plt.close()

def plot_autoregressive_alignment(V=5):
    """Generates a diagram visualizing inputs, targets, and loss masks."""
    seq_len = 2 * V - 1
    inputs_labels = []
    targets_labels = []
    mask = []
    
    for i in range(1, V + 1):
        inputs_labels.append(f"x{i}")
        targets_labels.append(f"y{i}")
        mask.append(1)
        if i < V:
            inputs_labels.append(f"y{i}")
            targets_labels.append(f"x{i+1}")
            mask.append(0)

    mask_array = np.array(mask).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.cm.colors.ListedColormap(['#e0e0e0', '#a5d6a7'])
    
    ax.imshow(mask_array, cmap=cmap, aspect='auto', extent=[-0.5, seq_len-0.5, -0.5, 0.5])
    
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(seq_len) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', size=0)
    
    # Labels
    for idx in range(seq_len):
        ax.text(idx, 0.28, f"In: {inputs_labels[idx]}", ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(idx, 0.0, f"Target: {targets_labels[idx]}", ha='center', va='center', fontsize=11, fontweight='bold')
        
        status = "Compute Loss" if mask[idx] == 1 else "Masked (0)"
        color = "#2e7d32" if mask[idx] == 1 else "#757575"
        ax.text(idx, -0.28, status, ha='center', va='center', fontsize=9.5, color=color, fontweight='bold')

    # Add explanatory arrows below
    ax.text(seq_len/2 - 0.5, -0.7, 
            "Autoregressive Flow: Target at step (t) becomes Input at step (t+1)", 
            ha='center', va='center', fontsize=10.5, style='italic', color='#1976d2')

    ax.set_title("Shifted Sequence Alignment & Loss Target Masking", fontsize=14, pad=15, fontweight='bold')
    ax.set_ylim(-0.9, 0.5)
    
    # Hide outer spine
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    ax.tick_params(left=False, labelleft=False)
    plt.savefig('3_autoregressive_alignment.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_permutation_split(V=5)
    plot_causal_attention_mask(V=5)
    plot_autoregressive_alignment(V=5)
    print("Presentation diagrams successfully generated and saved to your working directory:")
    print("1. 1_permutation_split.png")
    print("2. 2_causal_mask.png")
    print("3. 3_autoregressive_alignment.png")