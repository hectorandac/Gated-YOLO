import torch
import csv

# Load gating decisions
gating_decisions = torch.load('runs/inference/exp240/gates.pt')

# Prepare data for CSV
csv_data = [['Section', 'Completely Off', 'Mostly Off', 'Mostly On', 'Always On', 'Blocked']]

# Analyze each section and store data in csv_data
for i, section_gates in enumerate(gating_decisions):
    if section_gates[0] is None:
        analysis = {'completely_off': 0, 'mostly_off': 0, 'mostly_on': 0, 'always_on': 0, 'blocked': 1}
    else:
        section_gates_float = section_gates[0].float()
        gating_frequency = section_gates_float.mean(dim=0)
        completely_off = (gating_frequency == 0)
        mostly_off = (gating_frequency < 0.3) & ~completely_off
        always_on = (gating_frequency == 1)
        mostly_on = ~(completely_off | mostly_off | always_on)
        analysis = {
            'completely_off': completely_off.float().mean().item(),
            'mostly_off': mostly_off.float().mean().item(),
            'mostly_on': mostly_on.float().mean().item(),
            'always_on': always_on.float().mean().item(),
            'blocked': 0
        }

    # Append the analysis for the current section to the csv_data list
    csv_data.append([i+1, analysis['completely_off'], analysis['mostly_off'],
                     analysis['mostly_on'], analysis['always_on'], analysis['blocked']])

# Write the data to a CSV file
csv_filename = 'gating_analysis.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Data has been saved to {csv_filename}")
