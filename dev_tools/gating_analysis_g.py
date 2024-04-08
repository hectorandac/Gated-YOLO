import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Analyze and visualize gating decisions.')
parser.add_argument('path', type=str, help='Path to the .pt file containing gating decisions.')
args = parser.parse_args()

gating_decisions = torch.load(args.path)

specific_gate_analysis = []
for i, section_gates in enumerate(gating_decisions):
    if section_gates[0] is None:
        specific_gate_analysis.append({'layer': i+1, 'status': 'blocked'})
        continue
    
    section_gates_float = section_gates[0].float()
    gating_frequency = section_gates_float.mean(dim=0).cpu()
    
    completely_off_indices = torch.where(gating_frequency == 0)[0].tolist()
    always_on_indices = torch.where(gating_frequency == 1)[0].tolist()
    partially_active_indices = torch.where((gating_frequency > 0) & (gating_frequency < 1))[0].tolist()

    specific_gate_analysis.append({
        'layer': i+1,
        'completely_off': completely_off_indices,
        'always_on': always_on_indices,
        'partially_active': partially_active_indices
    })

for analysis in specific_gate_analysis:
    layer = analysis['layer']
    if 'status' in analysis and analysis['status'] == 'blocked':
        print(f"Layer {layer}:\nBlocked")
    else:
        off_gates = ', '.join(map(str, analysis['completely_off']))
        on_gates = ', '.join(map(str, analysis['always_on']))
        partial_gates = ', '.join(map(str, analysis['partially_active']))
        print(f"Layer {layer}:\nCompletely Off Gates: [{off_gates}]\nAlways On Gates: [{on_gates}]\nPartially active: [{partial_gates}]")


num_layers = max([a['layer'] for a in specific_gate_analysis])
num_gates = max([max(a['always_on'] + a['completely_off'], default=0) for a in specific_gate_analysis if 'status' not in a], default=0) + 1
gate_status_matrix = np.full((num_layers, num_gates), 0.5)

for analysis in specific_gate_analysis:
    layer_idx = analysis['layer'] - 1
    if 'status' in analysis and analysis['status'] == 'blocked':
        gate_status_matrix[layer_idx, :] = 0.5
    else:
        for gate in analysis['always_on']:
            gate_status_matrix[layer_idx, gate] = 1
        
        for gate in analysis['completely_off']:
            gate_status_matrix[layer_idx, gate] = 0

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