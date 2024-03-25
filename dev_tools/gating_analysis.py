import torch
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Analyze gating decisions.")
parser.add_argument("--file_path", type=str, required=True, help="Path to the .pt file containing gating decisions.")
args = parser.parse_args()

# Load gating decisions from specified file path
gating_decisions = torch.load(args.file_path)

# Number of layers per group
layers_per_group = 5  # Modify this to change the grouping

# Group layers and analyze
grouped_analysis = []
num_sections = len(gating_decisions)
for i in range(0, num_sections, layers_per_group):
    group = gating_decisions[i:i+layers_per_group]
    # Aggregate analysis for the group
    group_analysis = {'completely_off': 0, 'mostly_off': 0, 'mostly_on': 0, 'always_on': 0, 'blocked': 0}
    for section_gates in group:
        if section_gates[0] is None:
            group_analysis['blocked'] += 1
            continue
        # Convert boolean tensor to float for mean calculation
        section_gates_float = section_gates[0].float()
        # Compute frequencies for the section
        gating_frequency = section_gates_float.mean(dim=0).cpu()
        # Classify filters
        completely_off = (gating_frequency == 0)
        mostly_off = (gating_frequency < 0.3) & ~completely_off
        always_on = (gating_frequency == 1)
        mostly_on = ~(completely_off | mostly_off | always_on)
        # Accumulate the analysis
        group_analysis['completely_off'] += completely_off.float().mean()
        group_analysis['mostly_off'] += mostly_off.float().mean()
        group_analysis['mostly_on'] += mostly_on.float().mean()
        group_analysis['always_on'] += always_on.float().mean()

    # Average the analysis over the group
    for key in group_analysis.keys():
        group_analysis[key] /= len(group)
    grouped_analysis.append(group_analysis)

    # Print the grouped analysis for debugging
    print(f"Layers {i+1} to {min(i+layers_per_group, num_sections)}: {group_analysis}")

# Create labels for each group
num_groups = len(grouped_analysis)
labels = [f"Layers {i*layers_per_group+1} to {min((i+1)*layers_per_group, num_sections)}" for i in range(num_groups)]

# Extract data for plotting with aggregated logic
aggregated_off_blocked = []
for g in grouped_analysis:
    # If any layer in the group is blocked, aggregate completely_off with blocked
    if g['blocked'] > 0:
        aggregated_value = g['completely_off'] + g['blocked']
        # If all layers are blocked, keep it as blocked only
        if g['blocked'] == 1:
            aggregated_value = g['blocked']
        aggregated_off_blocked.append(aggregated_value)
    else:
        aggregated_off_blocked.append(g['completely_off'])

mostly_off = [g['mostly_off'] for g in grouped_analysis]
mostly_on = [g['mostly_on'] for g in grouped_analysis]
on = [g['always_on'] for g in grouped_analysis]

# Plotting
plt.figure(figsize=(20, 8))

# Plot 'Always On' first (at the bottom)
plt.bar(labels, on, color='limegreen', label='Always On')

# Plot 'Mostly On', 'Mostly Off', and 'Completely Off/Blocked' in order
plt.bar(labels, mostly_on, bottom=on, color='gold')
current_bottom = [o + mo for o, mo in zip(on, mostly_on)]
# Determine the color for 'Completely Off/Blocked' based on the group analysis
colors = ['tomato' if g['completely_off'] > 0 or g['always_on'] > 0 else 'gray' for g in grouped_analysis]
plt.bar(labels, aggregated_off_blocked, bottom=current_bottom, color=colors, label='Always Off')

current_bottom = [cb + mo_off for cb, mo_off in zip(current_bottom, mostly_off)]
plt.bar(labels, mostly_off, bottom=current_bottom, color='gray', label="Blocked Layer")


plt.xticks(rotation=45, ha='right', fontsize=18)
plt.xlabel('Grouped Layer Sections', fontsize=18)
plt.ylabel('Proportion of Filters', fontsize=18)
plt.subplots_adjust(bottom=0.30, left=0.04, right=0.99, top=0.99)
plt.legend(fontsize=18)

# Save the figure
plt.savefig('fixed_grouping_gating_analysis.png')
