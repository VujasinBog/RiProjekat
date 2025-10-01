# src/experiment_runner.py
import os
import json
import time
from pathlib import Path
from bruteforce import brute_force_search
from greedy import greedy_schedule
from algorithm import ElectromagnetismAlgorithm
from task import Task
from computerNode import ComputeNode


def load_test(path):
    with open(path, 'r') as f:
        data = json.load(f)
    tasks = [Task(**t) for t in data['tasks']]
    nodes = [ComputeNode(**n) for n in data['nodes']]
    return tasks, nodes


def run_experiment(test_path, em_pop=30, em_iter=100, bf_limit=60):
    print(f"\n{'='*50}")
    print(f"Test: {os.path.basename(test_path)}")
    print('='*50)
    
    tasks, nodes = load_test(test_path)
    print(f"Tasks: {len(tasks)}, Nodes: {len(nodes)}")
    
    results = {
        'test': os.path.basename(test_path),
        'tasks': len(tasks),
        'nodes': len(nodes)
    }
    
    # ---- Brute-force ----
    n_combinations = len(nodes) ** len(tasks)
    print(f"\nBRUTE-FORCE ({n_combinations:,} combinations):")
    
    if n_combinations > 10000000:
        print("  Skipped (too many combinations)")
        results['bruteforce'] = None
    else:
        try:
            bf_assign, bf_obj, bf_valid, bf_time = brute_force_search(
                tasks, nodes, time_limit=bf_limit, prune=True
            )
            results['bruteforce'] = {
                'objective': bf_obj,
                'valid': bf_valid,
                'time': bf_time
            }
            print(f"  Objective: {bf_obj:.2f}")
            print(f"  Valid: {bf_valid}")
            print(f"  Time: {bf_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
            results['bruteforce'] = None
    
    # ---- Greedy ----
    print(f"\nGREEDY (Greedy-Load):")
    try:
        assign, obj, valid, runtime = greedy_schedule(tasks, nodes)
        results['greedy'] = {
            'objective': obj,
            'valid': valid,
            'time': runtime
        }
        print(f"  Objective: {obj:.2f}")
        print(f"  Valid: {valid}")
        print(f"  Time: {runtime:.4f}s")
    except Exception as e:
        print(f"  Greedy Failed: {e}")
        results['greedy'] = None
    
    # ---- EM ----
    print(f"\nEM ALGORITHM:")
    em = ElectromagnetismAlgorithm(tasks, nodes, 
                                   population_size=em_pop, 
                                   max_iterations=em_iter)
    
    t0 = time.time()
    best_particle, best_obj = em.run()
    em_time = time.time() - t0
    
    em_valid = False
    if best_particle:
        best_particle.update_nodes_from_position()
        _, em_valid = best_particle.evaluate()
    
    results['em'] = {
        'objective': best_obj,
        'valid': em_valid,
        'time': em_time
    }
    print(f"  Objective: {best_obj:.2f}")
    print(f"  Valid: {em_valid}")
    print(f"  Time: {em_time:.2f}s")
    
    # ---- Poređenje EM i Greedy vs BF ----
    if results['bruteforce'] and results['bruteforce']['valid']:
        # Standardno poređenje vs brute-force
        bf_obj = results['bruteforce']['objective']
        bf_time = results['bruteforce']['time']
        
        # EM gap i speedup
        if em_valid:
            em_gap = ((best_obj - bf_obj) / bf_obj) * 100
            em_speedup = bf_time / em_time
            print(f"\nCOMPARISON (EM vs BF):")
            print(f"  Gap: {em_gap:.2f}%")
            print(f"  Speedup: {em_speedup:.2f}x")
            results['em']['gap_vs_bf'] = em_gap
            results['em']['speedup_vs_bf'] = em_speedup

        # Greedy gap i speedup
        if results['greedy'] and results['greedy']['valid']:
            greedy_obj = results['greedy']['objective']
            greedy_time = results['greedy']['time']
            greedy_gap = ((greedy_obj - bf_obj) / bf_obj) * 100
            greedy_speedup = bf_time / greedy_time
            print(f"\nCOMPARISON (Greedy vs BF):")
            print(f"  Gap: {greedy_gap:.2f}%")
            print(f"  Speedup: {greedy_speedup:.2f}x")
            results['greedy']['gap_vs_bf'] = greedy_gap
            results['greedy']['speedup_vs_bf'] = greedy_speedup

    else:
        # Ako brute-force nije dostupan, EM je referenca
        if results['greedy'] and results['greedy']['valid'] and em_valid:
            greedy_obj = results['greedy']['objective']
            greedy_time = results['greedy']['time']
            
            greedy_gap_vs_em = ((greedy_obj - best_obj) / best_obj) * 100
            speedup_em_vs_greedy = em_time / greedy_time
            
            print(f"\nCOMPARISON (Greedy vs EM):")
            print(f"  Gap: {greedy_gap_vs_em:.2f}%")
            print(f"  Speedup (EM vs Greedy): {speedup_em_vs_greedy:.2f}x")
            
            results['greedy']['gap_vs_em'] = greedy_gap_vs_em
            results['greedy']['speedup_vs_em'] = speedup_em_vs_greedy
            # EM je referenca
            results['em']['gap_vs_greedy'] = 0.0
            results['em']['speedup_vs_greedy'] = 1.0

    return results


def main():
    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for category in ['easy', 'medium', 'hard']:
        cat_path = data_dir / category
        if not cat_path.exists():
            continue
        
        print(f"\n{'#'*50}")
        print(f"CATEGORY: {category.upper()}")
        print('#'*50)
        
        for test_file in sorted(cat_path.glob('test*.json')):
            result = run_experiment(str(test_file))
            result['category'] = category
            all_results.append(result)
    
    # Sačuvaj rezultate
    output_file = results_dir / f'results_{int(time.time())}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_file}")
    print('='*50)


if __name__ == "__main__":
    main()

