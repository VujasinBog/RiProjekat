from task import Task

class ComputeNode:

    #Klasa koja predstavlja računarski čvor sa dostupnim resursima

    def __init__(self, id: int, cpu_capacity: float, memory_capacity: float, network_capacity: float):
        self.id = id
        self.cpu_capacity = cpu_capacity          # Ukupan kapacitet CPU (broj jezgara)
        self.memory_capacity = memory_capacity    # Ukupan kapacitet memorije (u GB)
        self.network_capacity = network_capacity  # Ukupan mrežni kapacitet (u Mbps)

        # Trenutno zauzeti resursi
        self.cpu_used = 0.0
        self.memory_used = 0.0
        self.network_used = 0.0

        # Lista zadataka dodeljenih ovom čvoru
        self.assigned_tasks = []

    def get_remaining_cpu(self) -> float:
        return self.cpu_capacity - self.cpu_used

    def get_remaining_memory(self) -> float:
        return self.memory_capacity - self.memory_used

    def get_remaining_network(self) -> float:
        return self.network_capacity - self.network_used

    def can_accommodate(self, task: Task) -> bool:
        """Proverava da li čvor može da primi zadatak"""
        return (self.get_remaining_cpu() >= task.cpu_req and
                self.get_remaining_memory() >= task.memory_req and
                self.get_remaining_network() >= task.network_req)

    def assign_task(self, task: Task) -> bool:
        #Dodeljuje zadatak čvoru ako ima dovoljno resursa
        if self.can_accommodate(task):
            self.cpu_used += task.cpu_req
            self.memory_used += task.memory_req
            self.network_used += task.network_req
            self.assigned_tasks.append(task)
            task.assigned_node = self.id
            return True
        return False

    def remove_task(self, task: Task) -> bool:
        """Uklanja zadatak sa čvora"""
        if task in self.assigned_tasks:
            self.cpu_used -= task.cpu_req
            self.memory_used -= task.memory_req
            self.network_used -= task.network_req
            self.assigned_tasks.remove(task)
            task.assigned_node = None
            return True
        return False

    def calculate_load_factor(self) -> float:
        """Računa faktor opterećenja čvora (0-1)"""
        if len(self.assigned_tasks) == 0:
            return 0.0

        cpu_factor = self.cpu_used / self.cpu_capacity
        memory_factor = self.memory_used / self.memory_capacity
        network_factor = self.network_used / self.network_capacity

        # Vraća maksimum od tri faktora kao indikator uskog grla
        return max(cpu_factor, memory_factor, network_factor)

    def calculate_execution_time(self) -> float:

        #Računa ukupno vreme izvršavanja zadataka na čvoru,
        #uzimajući u obzir usporenje zbog zauzetosti resursa

        if len(self.assigned_tasks) == 0:
            return 0.0

        load_factor = self.calculate_load_factor()

        # Penalizacija za visoko opterećenje (nelinearno povećanje vremena izvršavanja)
        slowdown_factor = 1.0 + 2.0 * (load_factor ** 2)

        # Računa ukupno vreme izvršavanja sa usporenjem
        return sum(task.execution_time * slowdown_factor for task in self.assigned_tasks)

    def __str__(self):
        return (f"Node {self.id}: CPU={self.cpu_used}/{self.cpu_capacity}, "
                f"Mem={self.memory_used}/{self.memory_capacity}GB, "
                f"Net={self.network_used}/{self.network_capacity}Mbps, "
                f"Tasks={len(self.assigned_tasks)}")
