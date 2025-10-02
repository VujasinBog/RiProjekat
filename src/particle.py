import numpy as np
import random
from task import Task
from typing import List,Tuple
from computerNode import ComputeNode

class Particle:

    #Čestica u EM algoritmu koja predstavlja jedno rešenje
    #(raspored zadataka po čvorovima)

    def __init__(self, tasks: List[Task], nodes: List[ComputeNode]):
        self.tasks = tasks
        self.nodes = [ComputeNode(node.id, node.cpu_capacity, node.memory_capacity, node.network_capacity)
                      for node in nodes]  # Pravi kopije čvorova
        self.position = np.zeros(len(tasks), dtype=int)  # Pozicija čestice (raspored zadataka)
        self.charge = 0.0  # Naelektrisanje čestice (kvalitet rešenja)

        # Inicijalno slučajno raspoređujemo zadatke
        self.randomize_allocation()

    def randomize_allocation(self):
        #Slučajno raspoređuje zadatke na čvorove

        # Resetujemo čvorove
        for node in self.nodes:
            node.cpu_used = 0.0
            node.memory_used = 0.0
            node.network_used = 0.0
            node.assigned_tasks = []

        # Slučajno raspoređujemo zadatke
        for i, task in enumerate(self.tasks):
            task.assigned_node = None

            # Pravimo listu čvorova koji mogu da prime zadatak
            valid_nodes = [node for node in self.nodes if node.can_accommodate(task)]

            if valid_nodes:
                # Biramo slučajni čvor iz liste validnih
                selected_node = random.choice(valid_nodes)
                selected_node.assign_task(task)
                self.position[i] = selected_node.id
            else:
                # Ako nema validnih čvorova, biramo slučajan čvor
                # (ovo će biti nevalidno rešenje ali omogućava dalju pretragu)
                self.position[i] = random.randint(0, len(self.nodes) - 1)

    def update_nodes_from_position(self):
        #Ažurira stanje čvorova na osnovu trenutne pozicije

        # Resetujemo čvorove
        for node in self.nodes:
            node.cpu_used = 0.0
            node.memory_used = 0.0
            node.network_used = 0.0
            node.assigned_tasks = []

        # Dodeljujemo zadatke prema poziciji
        for i, task in enumerate(self.tasks):
            node_id = self.position[i]
            task.assigned_node = node_id

            # Pronalazimo odgovarajući čvor
            for node in self.nodes:
                if node.id == node_id:
                    # Dodeljujemo zadatak ali ne proveravamo resurse
                    # jer želimo da izračunamo vrednost funkcije i za nevalidna rešenja
                    node.cpu_used += task.cpu_req
                    node.memory_used += task.memory_req
                    node.network_used += task.network_req
                    node.assigned_tasks.append(task)
                    break

    def evaluate(self) -> Tuple[float, bool]:

        #Evaluira trenutno rešenje i računa naelektrisanje čestice
        #Vraća (vrednost_funkcije, validnost_rešenja)

        self.update_nodes_from_position()

        # Proveravamo da li je rešenje validno (da li su prekoračeni kapaciteti)
        valid_solution = True
        for node in self.nodes:
            if (node.cpu_used > node.cpu_capacity or
                node.memory_used > node.memory_capacity or
                node.network_used > node.network_capacity):
                valid_solution = False
                break

        # Računamo ukupno vreme izvršavanja svih zadataka
        total_execution_time = sum(node.calculate_execution_time() for node in self.nodes)

        # Računamo standardnu devijaciju opterećenja (za balansiranje)
        load_factors = [node.calculate_load_factor() for node in self.nodes]
        load_balance = np.std(load_factors) if len(load_factors) > 1 else 0

        # Računamo penale za prekoračenje resursa ako rešenje nije validno
        penalty = 0
        if not valid_solution:
            for node in self.nodes:
                cpu_overflow = max(0, node.cpu_used - node.cpu_capacity)
                memory_overflow = max(0, node.memory_used - node.memory_capacity)
                network_overflow = max(0, node.network_used - node.network_capacity)

                penalty += 100000 * (cpu_overflow + memory_overflow + network_overflow)

        # Ciljna funkcija: minimizujemo ukupno vreme izvršavanja, disbalans i penale
        objective_value = total_execution_time + 500 * load_balance + penalty

        # Naelektrisanje je obrnuto proporcionalno vrednosti funkcije
        # (veće naelektrisanje za bolja rešenja)
        self.charge = 1.0 / (1.0 + objective_value)

        return objective_value, valid_solution
