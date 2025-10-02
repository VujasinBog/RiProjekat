import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Putanja do JSON fajla sa rezultatima
results_file = Path("results/results_1759346162.json")
with open(results_file, 'r') as f:
    data = json.load(f)

# --- Priprema podataka ---
bf_data = []  # samo testovi gde postoji BF
greedy_data = []
em_data = []

for res in data:
    tasks = res['tasks']
    nodes = res['nodes']
    n_comb = nodes ** tasks
    
    # Proveri da li postoji BF rezultat
    bf_exists = res.get('bruteforce') is not None
    
    if bf_exists:
        bf_obj = res['bruteforce']['objective']
        bf_time = res['bruteforce']['time']
        
        # Greedy
        if res.get('greedy') and res['greedy'].get('valid'):
            greedy_obj = res['greedy']['objective']
            greedy_time = res['greedy']['time']
            greedy_gap = ((greedy_obj - bf_obj) / bf_obj) * 100
            greedy_speedup = bf_time / greedy_time if greedy_time > 0 else 0
            
            greedy_data.append({
                'combinations': n_comb,
                'gap': greedy_gap,
                'speedup': greedy_speedup,
                'test': res['test']
            })
        
        # EM
        if res.get('em') and res['em'].get('valid'):
            em_obj = res['em']['objective']
            em_time = res['em']['time']
            em_gap = ((em_obj - bf_obj) / bf_obj) * 100
            em_speedup = bf_time / em_time if em_time > 0 else 0
            
            em_data.append({
                'combinations': n_comb,
                'gap': em_gap,
                'speedup': em_speedup,
                'test': res['test']
            })

# Sortiraj po broju kombinacija
greedy_data.sort(key=lambda x: x['combinations'])
em_data.sort(key=lambda x: x['combinations'])

# Ekstraktuj podatke
greedy_combs = [d['combinations'] for d in greedy_data]
greedy_gaps = [d['gap'] for d in greedy_data]
greedy_speedups = [d['speedup'] for d in greedy_data]

em_combs = [d['combinations'] for d in em_data]
em_gaps = [d['gap'] for d in em_data]
em_speedups = [d['speedup'] for d in em_data]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Gap graf
if greedy_combs:
    ax1.plot(greedy_combs, greedy_gaps, 'o-', 
             label='Greedy', color='#2E86AB', linewidth=2.5, markersize=8)

if em_combs:
    ax1.plot(em_combs, em_gaps, 's-', 
             label='EM', color='#A23B72', linewidth=2.5, markersize=8)

ax1.set_xscale('log')
ax1.set_xlabel('Broj kombinacija', fontsize=12, fontweight='bold')
ax1.set_ylabel('Gap od optimalnog (%)', fontsize=12, fontweight='bold')
ax1.set_title('Kvalitet resenja - odstupanje od BF optimuma', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Optimalno')

# Dodaj anotacije za zanimljive tacke
if em_gaps and max(em_gaps) > 0:
    max_gap_idx = em_gaps.index(max(em_gaps))
    ax1.annotate(f'{em_gaps[max_gap_idx]:.2f}%', 
                xy=(em_combs[max_gap_idx], em_gaps[max_gap_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, alpha=0.7)

# 2. Speedup graf
if greedy_combs:
    ax2.plot(greedy_combs, greedy_speedups, 'o-', 
             label='Greedy', color='#2E86AB', linewidth=2.5, markersize=8)

if em_combs:
    ax2.plot(em_combs, em_speedups, 's-', 
             label='EM', color='#A23B72', linewidth=2.5, markersize=8)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Broj kombinacija', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (BF vreme / algoritam vreme)', fontsize=12, fontweight='bold')
ax2.set_title('Brzina izvrsavanja - ubrzanje u odnosu na BF', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, framealpha=0.9)
ax2.grid(True, which="both", ls="--", alpha=0.3)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Jednako brzo')

# Dodaj anotaciju za najbolji speedup
if em_speedups:
    max_speedup_idx = em_speedups.index(max(em_speedups))
    ax2.annotate(f'{em_speedups[max_speedup_idx]:.1f}x', 
                xy=(em_combs[max_speedup_idx], em_speedups[max_speedup_idx]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Ispis statistike ---
print("\n" + "="*60)
print("STATISTIKA POREDJENJA SA BRUTE-FORCE")
print("="*60)

if greedy_data:
    print("\nGREEDY:")
    avg_gap = np.mean(greedy_gaps)
    avg_speedup = np.mean(greedy_speedups)
    print(f"  Prosecan gap: {avg_gap:.2f}%")
    print(f"  Prosecan speedup: {avg_speedup:.2f}x")
    print(f"  Najbolji gap: {min(greedy_gaps):.2f}%")
    print(f"  Najgori gap: {max(greedy_gaps):.2f}%")

if em_data:
    print("\nEM:")
    avg_gap = np.mean(em_gaps)
    avg_speedup = np.mean(em_speedups)
    print(f"  Prosecan gap: {avg_gap:.2f}%")
    print(f"  Prosecan speedup: {avg_speedup:.2f}x")
    print(f"  Najbolji gap: {min(em_gaps):.2f}%")
    print(f"  Najgori gap: {max(em_gaps):.2f}%")

print("\n" + "="*60)
print(f"Ukupno testova sa BF: {len(set(greedy_combs + em_combs))}")
print(f"Testova sa Greedy: {len(greedy_data)}")
print(f"Testova sa EM: {len(em_data)}")
print("="*60)
