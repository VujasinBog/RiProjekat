import numpy as np
import random
from typing import List, Tuple
from task import Task
from computerNode import ComputeNode
from particle import Particle
import matplotlib.pyplot as plt

class ElectromagnetismAlgorithm:

    #Implementacija algoritma elektromagnetizma za problem raspodele resursa

    def __init__(self, tasks: List[Task], nodes: List[ComputeNode],
                 population_size: int = 20, max_iterations: int = 100):
        self.tasks = tasks
        self.nodes = nodes
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.particles = []
        self.best_particle = None
        self.best_objective = float('inf')
        self.history = []

    def initialize(self):
        #Inicijalizuje populaciju čestica
        self.particles = [Particle(self.tasks, self.nodes) for _ in range(self.population_size)]

        # Evaluiramo sve čestice i čuvamo najbolju
        for particle in self.particles:
            objective, valid = particle.evaluate()
            if valid and objective < self.best_objective:
                self.best_objective = objective
                self.best_particle = particle

        # Ako nema validnih čestica, uzimamo najbolju bez obzira na validnost
        if self.best_particle is None and self.particles:
            self.best_particle = min(self.particles, key=lambda p: p.evaluate()[0])
            self.best_objective = self.best_particle.evaluate()[0]

    def calculate_forces(self):
        """Računa elektromagnetne sile između čestica"""
        # Normalizujemo naelektrisanja
        charges = np.array([particle.charge for particle in self.particles])
        total_charge = np.sum(charges)
        if total_charge > 0:
            normalized_charges = charges / total_charge
        else:
            normalized_charges = np.ones(len(charges)) / len(charges)

        # Za svaku česticu računamo rezultantnu silu
        for i, particle_i in enumerate(self.particles):
            force = np.zeros_like(particle_i.position, dtype=float)

            for j, particle_j in enumerate(self.particles):
                if i != j:
                    # Računamo vektor rastojanja
                    distance_vector = particle_j.position - particle_i.position

                    # Računamo euklidsko rastojanje (plus mali epsilon da izbegnemo deljenje sa nulom)
                    distance = np.sqrt(np.sum(distance_vector**2)) + 1e-10

                    # Računamo intenzitet sile prema Kulonovom zakonu
                    magnitude = normalized_charges[i] * normalized_charges[j] / (distance**2)

                    # Određujemo smer sile (privlačenje ili odbijanje)
                    if particle_j.charge > particle_i.charge:
                        # Privlačenje prema boljem rešenju
                        force += magnitude * distance_vector
                    else:
                        # Odbijanje od lošijeg rešenja
                        force -= magnitude * distance_vector

            # Primenjujemo silu za pomeranje čestice
            self.move_particle(particle_i, force)

    def move_particle(self, particle: Particle, force: np.ndarray):
        #Pomera česticu u skladu sa silom koja deluje na nju
        # Za svaki zadatak
        for i in range(len(particle.position)):
            # Računamo verovatnoću promene na osnovu sile
            probability = np.abs(force[i]) / (np.max(np.abs(force)) + 1e-10)

            # Odlučujemo da li ćemo menjati dodelu zadatka
            if random.random() < probability:
                # Biramo novi čvor za zadatak
                current_node_id = particle.position[i]
                available_nodes = [node.id for node in particle.nodes if node.id != current_node_id]

                if available_nodes:
                    # Ako je sila pozitivna, biramo čvor prema usmerenju sile
                    if force[i] > 0:
                        # Biramo sledeći čvor u nizu (kružno)
                        next_indices = [(current_node_id + k) % len(particle.nodes) for k in range(1, len(particle.nodes))]
                        for next_id in next_indices:
                            if next_id in available_nodes:
                                particle.position[i] = next_id
                                break
                    else:
                        # Biramo prethodni čvor u nizu (kružno)
                        prev_indices = [(current_node_id - k) % len(particle.nodes) for k in range(1, len(particle.nodes))]
                        for prev_id in prev_indices:
                            if prev_id in available_nodes:
                                particle.position[i] = prev_id
                                break

        # Evaluiramo novu poziciju
        objective, valid = particle.evaluate()

        # Ažuriramo najbolje rešenje ako je novo rešenje bolje
        if valid and objective < self.best_objective:
            self.best_objective = objective
            self.best_particle = particle

    def local_search(self, particle: Particle, max_attempts: int = 20):
        #Lokalna pretraga za fino podešavanje rešenja
        current_position = particle.position.copy()
        current_objective, current_valid = particle.evaluate()

        for _ in range(max_attempts):
            # Biramo slučajan zadatak
            task_idx = random.randint(0, len(self.tasks) - 1)
            current_node_id = particle.position[task_idx]

            # Biramo drugi slučajan čvor
            available_nodes = [node.id for node in particle.nodes if node.id != current_node_id]
            if not available_nodes:
                continue

            new_node_id = random.choice(available_nodes)

            # Probamo novu dodelu
            particle.position[task_idx] = new_node_id
            new_objective, new_valid = particle.evaluate()

            # Prihvatamo novu dodelu ako je bolja
            if new_valid and (not current_valid or new_objective < current_objective):
                current_objective = new_objective
                current_valid = new_valid

                # Ažuriramo najbolje rešenje ako je potrebno
                if new_valid and new_objective < self.best_objective:
                    self.best_objective = new_objective
                    self.best_particle = particle
            else:
                # Vraćamo staru dodelu
                particle.position[task_idx] = current_node_id
                particle.evaluate()

    def run(self):
        #Pokreće EM algoritam
        self.initialize()

        for iteration in range(self.max_iterations):
            # Računamo sile i pomeramo čestice
            self.calculate_forces()

            # Primenjujemo lokalnu pretragu na najbolju česticu
            if self.best_particle:
                self.local_search(self.best_particle)

            # Pamtimo istoriju najboljih vrednosti za grafik
            self.history.append(self.best_objective)

            # Ispisujemo napredak
            if (iteration + 1) % 10 == 0:
                print(f"Iteracija {iteration + 1}/{self.max_iterations}, "
                      f"Najbolja vrednost: {self.best_objective:.2f}")

        return self.best_particle, self.best_objective

    

    def print_solution(self):
        #Ispisuje detalje najboljeg rešenja
        if self.best_particle:
            self.best_particle.update_nodes_from_position()

            print("\nNajbolje rešenje:")
            print(f"Vrednost ciljne funkcije: {self.best_objective:.2f}")

            print("\nRaspored zadataka po čvorovima:")
            for node in self.best_particle.nodes:
                print(f"\n{node}")
                for task in node.assigned_tasks:
                    print(f"  - {task}")

                load_factor = node.calculate_load_factor()
                exec_time = node.calculate_execution_time()
                print(f"  Opterećenje: {load_factor:.2f}, Vreme izvršavanja: {exec_time:.2f}s")

            # Računamo balans opterećenja
            load_factors = [node.calculate_load_factor() for node in self.best_particle.nodes]
            print(f"\nStandardna devijacija opterećenja: {np.std(load_factors):.4f}")

            # Prikazujemo ukupne resurse
            total_cpu_used = sum(node.cpu_used for node in self.best_particle.nodes)
            total_cpu_capacity = sum(node.cpu_capacity for node in self.best_particle.nodes)

            total_memory_used = sum(node.memory_used for node in self.best_particle.nodes)
            total_memory_capacity = sum(node.memory_capacity for node in self.best_particle.nodes)

            total_network_used = sum(node.network_used for node in self.best_particle.nodes)
            total_network_capacity = sum(node.network_capacity for node in self.best_particle.nodes)

            print(f"\nUkupno iskorišćeno CPU: {total_cpu_used:.2f}/{total_cpu_capacity:.2f} "
                  f"({(total_cpu_used/total_cpu_capacity)*100:.2f}%)")
            print(f"Ukupno iskorišćena memorija: {total_memory_used:.2f}/{total_memory_capacity:.2f}GB "
                  f"({(total_memory_used/total_memory_capacity)*100:.2f}%)")
            print(f"Ukupno iskorišćen mrežni saobraćaj: {total_network_used:.2f}/{total_network_capacity:.2f}Mbps "
                  f"({(total_network_used/total_network_capacity)*100:.2f}%)")

