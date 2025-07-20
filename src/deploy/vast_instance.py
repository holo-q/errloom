from typing import Dict, List, Optional
from msgspec import Struct, field
from dataclasses import dataclass
from typing import Optional

class VastInstanceStats(Struct):
    """Statistics for a vast.ai instance including GPU, CPU, and system metrics."""
    gpu_name: str
    gpu_ram: int  # MB
    gpu_util: float
    gpu_temp: float
    cpu_name: str
    cpu_ram: float  # MB
    cpu_cores: int
    cpu_util: float
    disk_name: str
    disk_space: float  # GB
    disk_util: float
    disk_bw: float
    inet_up: float  # Mbps
    inet_down: float  # Mbps
    driver_version: str
    compute_cap: int
    dlperf: float
    reliability: float
    gpu_power: Optional[float] = None
    gpu_display_active: bool = False
    gpu_frac: float = 1.0
    gpu_lanes: int = 16
    gpu_mem_bw: float = 0.0
    cpu_arch: str = 'amd64'

class VastInstanceCosts(Struct):
    """Cost-related information for a vast.ai instance."""
    dph_total: float  # Total dollars per hour
    dph_base: float  # Base dollars per hour
    storage_cost: float
    inet_up_cost: float
    inet_down_cost: float
    credit_discount: Optional[float] = None
    credit_discount_max: float = 0.0
    credit_balance: Optional[float] = None

class VastInstance(Struct):
    """
    A vast.ai instance representing a rented machine.
    Contains comprehensive information about the instance including
    its configuration, status, and associated costs.
    """
    # Core Instance Information
    id: int
    host_id: int
    machine_id: int
    actual_status: str
    intended_status: str
    bundle_id: int
    stats: VastInstanceStats
    costs: VastInstanceCosts
    ssh_host: str
    ssh_port: int
    public_ipaddr: str
    local_ipaddrs: List[str]
    ports: Dict[str, List[Dict[str, str]]]
    image_uuid: str
    image_runtype: str
    extra_env: List[List[str]]
    start_date: float
    end_date: float
    duration: float
    client_run_time: float
    host_run_time: float

    # Optional Fields
    jupyter_token: Optional[str] = None
    label: Optional[str] = None
    webpage: Optional[str] = None
    onstart: Optional[str] = None

    def __post_init__(self):
        """Convert string IPs to list and ensure types are correct."""
        if isinstance(self.local_ipaddrs, str):
            self.local_ipaddrs = [ip.strip() for ip in self.local_ipaddrs.split() if ip.strip()]

    @property
    def status(self) -> str:
        """Current status of the instance."""
        return self.actual_status

    @property
    def is_running(self) -> bool:
        """Check if the instance is currently running."""
        return self.actual_status == 'running'

    @property
    def uptime(self) -> float:
        """Get instance uptime in seconds."""
        return self.client_run_time

    @property
    def hourly_cost(self) -> float:
        """Get total hourly cost."""
        return self.costs.dph_total

    def get_ssh_command(self) -> str:
        """Generate SSH command for connecting to the instance."""
        return f"ssh -p {self.ssh_port} root@{self.ssh_host}"

    def __str__(self):
        """String representation of the instance."""
        return (f"VastInstance(id={self.id}, "
                f"status={self.status}, "
                f"gpu={self.stats.gpu_name}, "
                f"cost=${self.hourly_cost:.3f}/hr)")

    @classmethod
    def from_api_response(cls, data: Dict) -> 'VastInstance':
        """Create a VastInstance from raw API response data."""
        stats = VastInstanceStats(
            gpu_name=data['gpu_name'],
            gpu_ram=data['gpu_ram'],
            gpu_util=data['gpu_util'],
            gpu_temp=data['gpu_temp'],
            gpu_display_active=data['gpu_display_active'],
            gpu_frac=data['gpu_frac'],
            gpu_lanes=data['gpu_lanes'],
            gpu_mem_bw=data['gpu_mem_bw'],
            cpu_name=data['cpu_name'],
            cpu_ram=float(data['cpu_ram']),
            cpu_cores=data['cpu_cores'],
            cpu_util=data['cpu_util'],
            cpu_arch=data['cpu_arch'],
            disk_name=data['disk_name'],
            disk_space=data['disk_space'],
            disk_util=data['disk_util'],
            disk_bw=data['disk_bw'],
            inet_up=data['inet_up'],
            inet_down=data['inet_down'],
            driver_version=data['driver_version'],
            compute_cap=data['compute_cap'],
            dlperf=data['dlperf'],
            reliability=data['reliability2']
        )

        costs = VastInstanceCosts(
            dph_total=data['dph_total'],
            dph_base=data['dph_base'],
            storage_cost=data['storage_cost'],
            inet_up_cost=data['inet_up_cost'],
            inet_down_cost=data['inet_down_cost'],
            credit_discount=data['credit_discount'],
            credit_discount_max=data['credit_discount_max'],
            credit_balance=data['credit_balance']
        )

        return cls(
            id=data['id'],
            host_id=data['host_id'],
            machine_id=data['machine_id'],
            actual_status=data['actual_status'],
            intended_status=data['intended_status'],
            bundle_id=data['bundle_id'],
            stats=stats,
            costs=costs,
            ssh_host=data['ssh_host'],
            ssh_port=data['ssh_port'],
            public_ipaddr=data['public_ipaddr'],
            local_ipaddrs=data['local_ipaddrs'],
            ports=data.get('ports', {}),
            image_uuid=data['image_uuid'],
            image_runtype=data['image_runtype'],
            extra_env=data['extra_env'],
            jupyter_token=data['jupyter_token'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            duration=data['duration'],
            client_run_time=data['client_run_time'],
            host_run_time=data['host_run_time'],
            label=data['label'],
            webpage=data['webpage'],
            onstart=data['onstart']
        )



@dataclass
class VastInstanceView:
    """
    A simplified view of VastInstance for display in tables/lists
    Contains the most relevant information for users monitoring instances
    """
    id: int
    status: str  # actual_status
    gpu: str  # Combines stats.cuda and stats.gpu_name
    price: float  # costs.dph_total
    uptime: str  # duration formatted
    ip: Optional[str]  # public_ipaddr
    _instance: Optional['VastInstance'] = None
    
    @classmethod
    def from_instance(cls, instance: 'VastInstance'):
        gpu_str = f"{instance.stats.gpu_name} ({instance.stats.driver_version})"
        return cls(
            id=instance.id,
            status=instance.actual_status,
            gpu=gpu_str,
            price=instance.costs.dph_total,
            uptime=str(instance.duration),
            ip=instance.public_ipaddr,
            _instance=instance
        )


