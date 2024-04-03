import torch
import matplotlib.pyplot as plt
import numpy as np


# Assuming gating_decisions is loaded from a file as before
gating_decisions = torch.load('runs/inference/exp11/gates.pt')

# New approach to analyze and collect specific gates
specific_gate_analysis = []
for i, section_gates in enumerate(gating_decisions):
    if section_gates[0] is None:
        specific_gate_analysis.append({'layer': i+1, 'status': 'blocked'})
        continue
    
    section_gates_float = section_gates[0].float()
    gating_frequency = section_gates_float.mean(dim=0).cpu()
    
    # Identify specific gates
    completely_off_indices = torch.where(gating_frequency == 0)[0].tolist()
    always_on_indices = torch.where(gating_frequency == 1)[0].tolist()
    
    # Store the specific gate indices
    specific_gate_analysis.append({
        'layer': i+1,
        'completely_off': completely_off_indices,
        'always_on': always_on_indices
    })

# Print specific gates for each layer
for analysis in specific_gate_analysis:
    layer = analysis['layer']
    if 'status' in analysis and analysis['status'] == 'blocked':
        print(f"Layer {layer}: Blocked")
    else:
        off_gates = ', '.join(map(str, analysis['completely_off']))
        on_gates = ', '.join(map(str, analysis['always_on']))
        print(f"Layer {layer}: Completely Off Gates: [{off_gates}], Always On Gates: [{on_gates}]")


# Assuming specific_gate_analysis contains your detailed gate analysis as before
num_layers = max([a['layer'] for a in specific_gate_analysis])
num_gates = max([max(a['always_on'] + a['completely_off'], default=0) for a in specific_gate_analysis if 'status' not in a], default=0) + 1

# Initialize a matrix to represent the gate status across all layers
# We'll use 0 for off, 1 for on, and 0.5 for gates not consistently on or off
gate_status_matrix = np.full((num_layers, num_gates), 0.5)

for analysis in specific_gate_analysis:
    layer_idx = analysis['layer'] - 1  # Adjusting for 0-based indexing
    if 'status' in analysis and analysis['status'] == 'blocked':
        # Optional: Mark blocked layers differently, e.g., with a specific color
        gate_status_matrix[layer_idx, :] = 0.5  # Assuming grey for blocked/not consistent
    else:
        # Mark always on gates
        for gate in analysis['always_on']:
            gate_status_matrix[layer_idx, gate] = 1  # Green for always on
        # Mark always off gates
        for gate in analysis['completely_off']:
            gate_status_matrix[layer_idx, gate] = 0  # Red for always off

# Plotting the gate status matrix
plt.figure(figsize=(12, 8))
plt.imshow(gate_status_matrix, aspect='auto', cmap='RdYlGn', interpolation='none')
plt.colorbar(label='Gate Status', ticks=[0, 0.5, 1], format=plt.FuncFormatter(lambda val, loc: {0: 'Always Off', 0.5: 'Variable', 1: 'Always On'}[val]))
plt.xlabel('Gate Index')
plt.ylabel('Layer Number')
plt.title('Gate Activation Status Across Layers')

# Adjust ticks if necessary based on your number of gates and layers
plt.xticks(range(0, num_gates, max(1, num_gates // 10)))  # Show every 10th gate index if many gates
plt.yticks(range(num_layers))

plt.tight_layout()

# Save the figure before showing it
plt.savefig('gates_status_visualization.png', dpi=300)

plt.show()