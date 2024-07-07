import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv

parser = argparse.ArgumentParser(description='Analyze and visualize gating decisions.')
parser.add_argument('paths', type=str, nargs='+', help='Paths to the .pt files containing gating decisions.')
args = parser.parse_args()

def analyze_gating_decisions(path):
    gating_decisions = torch.load(path)
    specific_gate_analysis = []

    for i, section_gates in enumerate(gating_decisions):
        if section_gates[0] is None:
            total_capacity = section_gates[1].numel()
            specific_gate_analysis.append({
                'layer': i+1, 
                'status': 'blocked', 
                'completely_off': 100, 
                'num_gates': 0,
                'off_indices': set(),
                'total_capacity': total_capacity
            })
            continue
        
        section_gates_float = section_gates[0].float()
        gating_frequency = section_gates_float.mean(dim=0).cpu()
        total_capacity = section_gates_float.numel()
        
        completely_off_indices = torch.where(gating_frequency == 0)[0].tolist()
        
        specific_gate_analysis.append({
            'layer': i+1,
            'completely_off': len(completely_off_indices) / total_capacity * 100,
            'num_gates': len(completely_off_indices),
            'off_indices': set(completely_off_indices),
            'total_capacity': total_capacity
        })
    
    return specific_gate_analysis

def calculate_percentages(gate_analysis):
    percentages = {'completely_off': []}
    for analysis in gate_analysis:
        if 'status' in analysis and analysis['status'] == 'blocked':
            percentages['completely_off'].append(100)
        else:
            percentages['completely_off'].append(analysis['completely_off'])
    return percentages

def plot_gate_percentages(percentages_list, paths):
    layers = list(range(1, len(percentages_list[0]['completely_off']) + 1))
    plt.figure(figsize=(10, 6))
    for i, percentages in enumerate(percentages_list):
        dir_name = os.path.basename(os.path.dirname(paths[i]))
        plt.plot(layers, percentages['completely_off'], label=f'{dir_name} - Completely Off')
    
    plt.xlabel('Layer Number')
    plt.ylabel('Percentage of Gates Completely Off')
    plt.title('Percentage of Completely Off Gates Across Layers')
    plt.legend()
    plt.tight_layout()
    plt.savefig('completely_off_gates_percentages.png', dpi=300)
    plt.show()

def plot_gate_differences(analyses, paths):
    if len(analyses) > 1:
        layers = list(range(1, len(analyses[0]) + 1))
        plt.figure(figsize=(10, 6))
        for i in range(1, len(analyses)):
            differences = []
            for layer in range(len(analyses[0])):
                off_indices_i = analyses[i][layer]['off_indices']
                off_indices_i_1 = analyses[i-1][layer]['off_indices']
                diff_gates = off_indices_i.symmetric_difference(off_indices_i_1)
                total_capacity = analyses[i][layer]['total_capacity']
                differences.append(len(diff_gates) / total_capacity * 100)
            dir_name_1 = os.path.basename(os.path.dirname(paths[i-1]))
            dir_name_2 = os.path.basename(os.path.dirname(paths[i]))
            plt.plot(layers, differences, label=f'Differences {dir_name_1} vs {dir_name_2}')
        
        plt.xlabel('Layer Number')
        plt.ylabel('Percentage of Different Closed Gates')
        plt.title('Percentage of Different Closed Gates Between Models')
        plt.legend()
        plt.tight_layout()
        plt.savefig('gate_differences.png', dpi=300)
        plt.show()

def save_percentages_to_csv(percentages_list, paths):
    layers = list(range(1, len(percentages_list[0]['completely_off']) + 1))
    with open('percentages_of_closed_gates.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Layer'] + [os.path.basename(os.path.dirname(path)) for path in paths]
        writer.writerow(header)
        for layer in layers:
            row = [layer] + [percentages['completely_off'][layer-1] for percentages in percentages_list]
            writer.writerow(row)

def save_differences_to_csv(analyses, paths):
    if len(analyses) > 1:
        layers = list(range(1, len(analyses[0]) + 1))
        with open('differences_in_closed_gates.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Layer'] + [f'Diff {os.path.basename(os.path.dirname(paths[i-1]))} vs {os.path.basename(os.path.dirname(paths[i]))}' for i in range(1, len(paths))]
            writer.writerow(header)
            for layer in layers:
                row = [layer]
                for i in range(1, len(analyses)):
                    off_indices_i = analyses[i][layer-1]['off_indices']
                    off_indices_i_1 = analyses[i-1][layer-1]['off_indices']
                    diff_gates = off_indices_i.symmetric_difference(off_indices_i_1)
                    total_capacity = analyses[i][layer-1]['total_capacity']
                    row.append(len(diff_gates) / total_capacity * 100)
                writer.writerow(row)

all_percentages = []
all_analyses = []

for path in args.paths:
    gate_analysis = analyze_gating_decisions(path)
    all_analyses.append(gate_analysis)
    percentages = calculate_percentages(gate_analysis)
    all_percentages.append(percentages)

plot_gate_percentages(all_percentages, args.paths)
plot_gate_differences(all_analyses, args.paths)
save_percentages_to_csv(all_percentages, args.paths)
save_differences_to_csv(all_analyses, args.paths)
