import matplotlib.pyplot as plt
import matplotlib.patches as patches

img_height, img_width = 8, 16

feature_map_sizes = [
    (48, 48),
    (24, 24),
    (12, 12),
    (6, 6),
    (3, 3)
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, img_width)
ax.set_ylim(0, img_height)

ax.set_xlabel("Width")
ax.set_ylabel("Height")
plt.title("Spatial Resolutions of Different Feature Maps")

legend_patches = []

# Different line styles for better distinction
line_styles = ['-', ':', ':', ':', ':']
line_widths = [2.0, 1.0, 1.0, 1.0, 1.0]

for idx, (height, width) in enumerate(feature_map_sizes):
    cell_width = img_width / width
    cell_height = img_height / height

    for i in range(width):
        ax.axvline(i * cell_width, linestyle=line_styles[idx], color=f"C{idx}", linewidth=line_widths[idx])
    for j in range(height):
        ax.axhline(j * cell_height, linestyle=line_styles[idx], color=f"C{idx}", linewidth=line_widths[idx])
    
    patch = patches.Patch(color=f"C{idx}", label=f'Layer {idx} - Grid: {height}x{width}, Cell: {cell_height:.2f}x{cell_width:.2f}')
    legend_patches.append(patch)

ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

plt.gca().invert_yaxis()
plt.grid(False)
plt.tight_layout()
plt.savefig("layer_resolutions.png", bbox_inches='tight')
plt.show()
