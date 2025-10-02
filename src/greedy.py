# src/greedy_scheduler.py
from typing import List, Tuple
from task import Task
from computerNode import ComputeNode
import numpy as np


def greedy_schedule(tasks: List[Task], nodes: List[ComputeNode]) -> Tuple[List[int], float, bool, float]:
    """
    Greedy algoritam za raspodelu zadataka.
    Strategija: Sortira zadatke po težini i za svaki bira čvor
    sa najmanjim budućim opterećenjem koji može da ga primi.
    
    Returns:
        (assignment, objective, valid, runtime)
    """
    import time
    start_time = time.time()
    
    # Kreiraj kopije čvorova
    node_copies = [ComputeNode(n.id, n.cpu_capacity, n.memory_capacity, n.network_capacity) 
                   for n in nodes]
    
    # Izračunaj "težinu" za svaki zadatak
    task_weights = []
    for i, task in enumerate(tasks):
        weight = task.cpu_req + task.memory_req / 10 + task.network_req / 100
        task_weights.append((i, task, weight))
    
    # Sortiraj po težini (opadajuće)
    task_weights.sort(key=lambda x: x[2], reverse=True)
    
    # Alociraj zadatke
    assignment = [None] * len(tasks)
    for original_idx, task, _ in task_weights:
        best_node = None
        best_score = float('inf')
        
        for node in node_copies:
            if node.can_accommodate(task):
                future_load = calculate_future_load(node, task)
                if future_load < best_score:
                    best_score = future_load
                    best_node = node
        
        if best_node:
            best_node.assign_task(task)
        else:
            # fallback – dodeli najmanje opterećenom iako nema resursa
            best_node = min(node_copies, key=lambda n: n.calculate_load_factor())
            best_node.cpu_used += task.cpu_req
            best_node.memory_used += task.memory_req
            best_node.network_used += task.network_req
            best_node.assigned_tasks.append(task)
        
        assignment[original_idx] = best_node.id
    
    # Evaluacija
    objective, valid = evaluate_greedy_solution(node_copies)
    runtime = time.time() - start_time
    
    return assignment, objective, valid, runtime


def calculate_future_load(node: ComputeNode, task: Task) -> float:
    """ Računa buduće opterećenje čvora ako se doda zadatak. """
    future_cpu = (node.cpu_used + task.cpu_req) / node.cpu_capacity
    future_mem = (node.memory_used + task.memory_req) / node.memory_capacity
    future_net = (node.network_used + task.network_req) / node.network_capacity
    return max(future_cpu, future_mem, future_net)


def evaluate_greedy_solution(nodes: List[ComputeNode]) -> Tuple[float, bool]:
    """ Evaluira kvalitet greedy rešenja. """
    valid = True
    penalty = 0
    
    for node in nodes:
        if (node.cpu_used > node.cpu_capacity or
            node.memory_used > node.memory_capacity or
            node.network_used > node.network_capacity):
            valid = False
            
            cpu_over = max(0, node.cpu_used - node.cpu_capacity)
            mem_over = max(0, node.memory_used - node.memory_capacity)
            net_over = max(0, node.network_used - node.network_capacity)
            
            penalty += 5000 * (cpu_over**2 + mem_over**2 + net_over**2)
            penalty += 2000
    
    total_execution_time = sum(node.calculate_execution_time() for node in nodes)
    load_factors = [node.calculate_load_factor() for node in nodes]
    load_balance = np.std(load_factors) if len(load_factors) > 1 else 0
    
    objective = total_execution_time + 500 * load_balance + penalty
    return objective, valid

