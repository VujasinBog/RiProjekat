from typing import List
import matplotlib.pyplot as plt
import numpy as np
from computerNode import ComputeNode



def save_results(results, output_path):
    """Čuva rezultate u JSON"""
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def compare_results(bf_result, em_result):
    """Poredi rezultate"""
    comparison = {}
    if bf_result and em_result:
        if bf_result['valid'] and em_result['valid']:
            bf_obj = bf_result['objective']
            em_obj = em_result['objective']
            gap = ((em_obj - bf_obj) / bf_obj) * 100 if bf_obj > 0 else 0
            comparison['objective_gap_percent'] = gap
    return comparison
    
    
def plot_time_comparison(results):
    """
    Crta grafik zavisnosti vremena izvršavanja od broja kombinacija
    Poredi brute-force i EM algoritam
    """
    # Izvuci podatke samo za testove gde je BF rađen
    bf_data = []
    em_data = []
    test_names = []
    
    for result in results:
        if result.get('bruteforce') is not None:
            n_combinations = result['nodes'] ** result['tasks']
            bf_time = result['bruteforce']['time']
            em_time = result['em']['time']
            
            bf_data.append((n_combinations, bf_time))
            em_data.append((n_combinations, em_time))
            test_names.append(result['test'])
    
    if not bf_data:
        print("Nema podataka za poređenje (svi BF preskočeni)")
        return
    
    # Sortiraj po broju kombinacija
    bf_data.sort()
    em_data.sort()
    
    combinations = [x[0] for x in bf_data]
    bf_times = [x[1] for x in bf_data]
    em_times = [x[1] for x in em_data]
    
    # Kreiraj grafik
    plt.figure(figsize=(12, 6))
    
    # Logaritamska skala za x-osu zbog velikog raspona
    plt.subplot(1, 2, 1)
    plt.plot(combinations, bf_times, 'o-', label='Brute-Force', linewidth=2, markersize=8)
    plt.plot(combinations, em_times, 's-', label='EM Algorithm', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Broj kombinacija')
    plt.ylabel('Vreme izvrsavanja (s)')
    plt.title('Vreme izvrsavanja - logaritamska skala')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speedup faktor
    plt.subplot(1, 2, 2)
    speedups = [bf_times[i] / em_times[i] for i in range(len(bf_times))]
    plt.plot(combinations, speedups, 'o-', color='green', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.axhline(y=1, color='r', linestyle='--', label='Jednako brzo')
    plt.xlabel('Broj kombinacija')
    plt.ylabel('Speedup (BF/EM)')
    plt.title('Koliko je puta EM brzi od BF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/time_comparison.png', dpi=150)
    plt.show()
    
    # Ispisi statistiku
    print("\n" + "="*50)
    print("STATISTIKA VREMENA:")
    print("="*50)
    for i, (comb, bf_t, em_t) in enumerate(zip(combinations, bf_times, em_times)):
        speedup = bf_t / em_t
        print(f"{test_names[i]:15} | Comb: {comb:>10,} | BF: {bf_t:>6.2f}s | EM: {em_t:>6.2f}s | Speedup: {speedup:>6.2f}x")
    
    avg_speedup = np.mean(speedups)
    print(f"\nProsecan speedup: {avg_speedup:.2f}x")
