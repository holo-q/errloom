from dataclasses import dataclass

@dataclass
class VastOffer:
    """
    A vast.ai offer data class.
    An offer is a machine that can be rented by the user.
    """
    index: int
    id: int
    cuda: str
    num: int
    model: str
    pcie: str
    cpu_ghz: float
    vcpus: int
    ram: float
    disk: float
    price: float
    dlp: float
    dlp_per_dollar: float
    score: float
    nv_driver: str
    net_up: float
    net_down: float
    r: float
    max_days: float
    mach_id: str
    status: str
    ports: str
    country: str

    def __str__(self):
        return self.tostring(self.index)

    def tostring(self, i):
        return f'[{i + 1:02}] - {self.model} - {self.num} - {self.dlp:.2f} - {self.net_down:.2f} Mbps - {self.price:.2f} $/hr - {self.dlp_per_dollar:.2f} DLP/HR'
