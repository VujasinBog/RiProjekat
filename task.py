
class Task:

    #Klasa koja predstavlja zadatak sa zahtevima za resursima

    def __init__(self, id: int, cpu_req: float, memory_req: float, network_req: float, execution_time: float):
        self.id = id
        self.cpu_req = cpu_req          # CPU zahtev (0-1, gde 1 predstavlja 100% jezgra)
        self.memory_req = memory_req    # Memorijski zahtev (u GB)
        self.network_req = network_req  # Mrežni zahtev (u Mbps)
        self.execution_time = execution_time  # Osnovno vreme izvršavanja (u sec)
        self.assigned_node = None       # Čvor kome je zadatak dodeljen

    def __str__(self):
        return f"Task {self.id}: CPU={self.cpu_req}, Mem={self.memory_req}GB, Net={self.network_req}Mbps, Time={self.execution_time}s"
