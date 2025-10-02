# src/brute_force.py
from typing import List, Tuple, Optional
import itertools
import math
import time

from task import Task
from computerNode import ComputeNode

def evaluate_solution(assignments: List[int], tasks: List[Task], nodes_template: List[ComputeNode]) -> Tuple[float, bool]:
    # build node copies
    nodes = [ComputeNode(n.id, n.cpu_capacity, n.memory_capacity, n.network_capacity) for n in nodes_template]
    for ti, node_id in enumerate(assignments):
        t = tasks[ti]
        node = next(n for n in nodes if n.id == node_id)
        node.cpu_used += t.cpu_req
        node.memory_used += t.memory_req
        node.network_used += t.network_req
        node.assigned_tasks.append(t)
    # validity
    valid = True
    for node in nodes:
        if node.cpu_used > node.cpu_capacity or node.memory_used > node.memory_capacity or node.network_used > node.network_capacity:
            valid = False
            break
    # total execution time
    total_execution_time = sum(node.calculate_execution_time() for node in nodes)
    load_factors = [node.calculate_load_factor() for node in nodes]
    load_balance = (0 if len(load_factors) <= 1 else float((sum((x - (sum(load_factors)/len(load_factors)))**2 for x in load_factors)/len(load_factors))**0.5))
    penalty = 0.0
    if not valid:
        for node in nodes:
            cpu_overflow = max(0, node.cpu_used - node.cpu_capacity)
            memory_overflow = max(0, node.memory_used - node.memory_capacity)
            network_overflow = max(0, node.network_used - node.network_capacity)
            penalty += 100000 * (cpu_overflow + memory_overflow + network_overflow)
    objective = total_execution_time + 500 * load_balance + penalty
    return objective, valid

def brute_force_search(tasks: List[Task], nodes: List[ComputeNode], time_limit: Optional[float]=None, prune: bool=True) -> Tuple[List[int], float, bool, float]:

    n_tasks = len(tasks)
    n_nodes = len(nodes)
    start = time.time()
    best_obj = float('inf')
    best_assign = None
    best_valid = False

    # generate product of node ids
    # but we will do recursive assignment with pruning
    def rec_assign(idx, partial_assign, nodes_state):
        nonlocal best_obj, best_assign, best_valid, start
        # time limit
        if time_limit is not None and (time.time() - start) > time_limit:
            raise TimeoutError("Brute force time limit reached")
        if idx == n_tasks:
            obj, valid = evaluate_solution(partial_assign, tasks, nodes)
            if obj < best_obj:
                best_obj = obj
                best_assign = partial_assign.copy()
                best_valid = valid
            return
        t = tasks[idx]
        for node in nodes_state:
            node_id = node.id
            # try assign
            node.cpu_used += t.cpu_req
            node.memory_used += t.memory_req
            node.network_used += t.network_req
            node.assigned_tasks.append(t)
            # prune if overflow
            overflow = (node.cpu_used > node.cpu_capacity) or (node.memory_used > node.memory_capacity) or (node.network_used > node.network_capacity)
            if not prune or not overflow:
                # optional optimistic lower bound: minimal additional exec time is 0 -> might skip
                rec_assign(idx + 1, partial_assign + [node_id], nodes_state)
            # undo
            node.cpu_used -= t.cpu_req
            node.memory_used -= t.memory_req
            node.network_used -= t.network_req
            node.assigned_tasks.pop()
            # optional early exit if best possible is already 0 etc.
    # nodes_state copies
    nodes_state = [ComputeNode(n.id, n.cpu_capacity, n.memory_capacity, n.network_capacity) for n in nodes]
    try:
        rec_assign(0, [], nodes_state)
    except TimeoutError:
        pass
    runtime = time.time() - start
    return best_assign, best_obj, best_valid, runtime
