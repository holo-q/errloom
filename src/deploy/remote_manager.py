import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import aiohttp

from src.deploy import vast_manager
from src.deploy.remote import LoomRemote
from src.deploy.vast_instance import VastInstance

log = logging.getLogger('rman')


class InstanceManager:
    """
Manages multiple RemoteInstance objects.

This class orchestrates operations across multiple instance instances,
handling deployment, job management, and balance monitoring.

Attributes:
    instances (list): List of RemoteInstance objects.
    vast (vast_manager): Manager for Vast.ai operations.

Methods:
    deploy_all(): Deploys Discore to all managed instance instances.
    start_all_jobs(): Starts jobs on all managed instance instances.
    stop_all_jobs(): Stops jobs on all managed instance instances.
    wait_for_all_jobs(jobs): Waits for all specified jobs to complete.
    monitor_balance(): Continuously monitors the Vast.ai account balance.
    add(instance): Adds a new RemoteInstance to be managed.
    """

    def __init__(self):
        self.instances: Dict[int, LoomRemote] = {}

    async def get_instance(self, iid_or_info, session: Optional[aiohttp.ClientSession] = None) -> LoomRemote:
        match iid_or_info:
            case LoomRemote():
                return iid_or_info
            case VastInstance() as info:
                self.instances[info.id] = LoomRemote(info, None)
                return self.instances[info.id]
            case int() as iid if iid in self.instances:
                return self.instances[iid]
            case int() as iid:
                infos = await vast_manager.instance.fetch_instances()  # TODO maybe we can pass in existing?
                if info := next((i for i in infos if i.id == iid), None):
                    s = LoomRemote(info, session)
                    self.instances[iid] = s
                    return s
                raise ValueError(f"Instance {iid} not found")
            case _:
                raise TypeError(f"Unexpected type for iid_or_info: {type(iid_or_info)}")

    def deploy_all(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(instance.deploy) for instance in self.instances.values()]
            for future in futures:
                future.result()

    def start_all_jobs(self):
        all_jobs = []
        for instance in self.instances.values():
            all_jobs.extend(instance.start_jobs())
        return all_jobs

    def stop_all_jobs(self):
        for instance in self.instances.values():
            instance.stop_jobs()

    def wait_for_all_jobs(self, jobs):
        for job in jobs:
            job.join()

    def monitor_balance(self):
        while any(instance.continue_work for instance in self.instances.values()):
            balance = vast.fetch_balance()
            log.info(f"Current balance: ${balance:.2f}")
            time.sleep(60)  # Check balance every minute

    def add(self, instance):
        self.instances[instance.id] = instance


instance = InstanceManager()
