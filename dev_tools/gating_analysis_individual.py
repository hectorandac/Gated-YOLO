import torch
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

def analyze_section(section_gates):
    # Check if the section is blocked using the third element (boolean)
    if section_gates[2]:
        # If the section is blocked, consider it as completely_off
        return {'completely_off': 1, 'mostly_off': 0, 'mostly_on': 0, 'always_on': 0}

    # Use the gates from the first element of section_gates
    section_gates_float = section_gates[0].float()
    gating_frequency = section_gates_float.mean(dim=0)

    conditions = {
        'completely_off': gating_frequency == 0,
        'mostly_off': (gating_frequency < 0.3) & (gating_frequency > 0),
        'always_on': gating_frequency == 1
    }
    conditions['mostly_on'] = ~(conditions['completely_off'] | conditions['mostly_off'] | conditions['always_on'])

    return {k: v.float().mean().item() for k, v in conditions.items()}  # Use .item() to get a plain number

def plot_individual_analysis(section_analysis):
    labels = [f'Layer {i+1}' for i in range(len(section_analysis))]
    stats = {key: [s[key] for s in section_analysis] for key in section_analysis[0]}
    plt.figure(figsize=(20, 8))
    bottom_stack = [0] * len(section_analysis)

    # Define colors for each key
    colors = {'always_on': 'limegreen', 'completely_off': 'tomato', 'mostly_on': 'orange', 'mostly_off': 'dodgerblue'}
    for key in ['always_on', 'mostly_on', 'mostly_off', 'completely_off']:
        data_to_plot = stats[key]
        plt.bar(labels, data_to_plot, bottom=bottom_stack, color=colors[key], label=key.replace('_', ' ').title())
        bottom_stack = [sum(x) for x in zip(bottom_stack, data_to_plot)]

    plt.xticks(rotation=90)  # Rotate to avoid overlapping labels
    plt.tight_layout()
    plt.xlabel('Layers')
    plt.ylabel('Proportion of Gates')
    plt.legend()
    plt.savefig('individual_layer_gating_analysis.png')

# Load gating decisions and analyze each section
gating_decisions = torch.load('runs/inference/exp240/gates.pt')
section_analysis = [analyze_section((section[0], section[1], section[2])) for section in gating_decisions]

# Plot the analysis for individual layers
plot_individual_analysis(section_analysis)
