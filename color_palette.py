"""Quick color palette viewer for purple-grey tones"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Purple-grey tone options
colors = {
    'Deep Purple': '#5a4d7d',      # Deeper, more saturated purple
    'Muted Purple': '#6b5b95',     # Classic muted purple
    'Cool Purple': '#6d6d8f',      # More grey in the purple
    'Slate Purple': '#5d5d7d',     # Greyish purple
    'Dusty Purple': '#7d7d9d',     # Lighter, dustier
    'Steel Purple': '#565d7f',     # Steel-like tone
    'Mauve': '#7d5f8f',            # Warm purple-grey
    'Lavender Grey': '#8d7d9f',    # Very muted, almost grey
    'Current Blue': '#0d47a1',     # For comparison
}

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle('Purple-Grey Tone Options for Button Colors', fontsize=14, fontweight='bold')
fig.patch.set_facecolor('#3d3d3d')

axes_flat = axes.flatten()
for idx, (name, color) in enumerate(colors.items()):
    ax = axes_flat[idx]
    ax.set_facecolor('#3d3d3d')
    
    # Draw large color block
    rect = patches.Rectangle((0.1, 0.3), 0.8, 0.4, 
                             facecolor=color, edgecolor='#e0e0e0', linewidth=2)
    ax.add_patch(rect)
    
    # Add hex code and name
    ax.text(0.5, 0.75, name, ha='center', va='center', 
           fontsize=11, fontweight='bold', color='#e0e0e0')
    ax.text(0.5, 0.1, color, ha='center', va='center', 
           fontsize=9, color='#aaa', family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\nColor codes for easy copying:")
for name, color in colors.items():
    print(f"  {name:20} {color}")
