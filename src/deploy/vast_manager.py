import ast
import asyncio
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp

import src.deploy.deploy_constants as const
import userconf
from src import paths
from src.deploy.remote import LoomRemote
from src.deploy.vast_instance import VastInstance
from src.deploy.vast_offer import VastOffer

log = logging.getLogger('vast')

class VastAIManager:
    def __init__(self):
        self.vastpath = paths.root / 'vast'
        self.remotes = dict()
        self.executor = ThreadPoolExecutor()
        self.destroyed_instances = set()

    async def download_vast_script(self):
        if not self.vastpath.is_file():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py") as response:
                    content = await response.text()
                    async with aiofiles.open(self.vastpath, mode='w') as f:
                        await f.write(content)
            if not sys.platform.startswith('win'):
                await asyncio.to_thread(self.vastpath.chmod, 0o755)

    async def run_command(self, command):
        command = [str(c) for c in command]
        # log.info(f"Running command: vast {' '.join(command)}")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            self.vastpath.as_posix(),
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode('utf-8')

    async def fetch_balance(self)->float:
        out = await self.run_command(['show', 'invoices'])
        ret = json.loads(out.splitlines()[-1].replace('Current:  ', '').replace("'", '"'))['credit']
        return ret

    async def fetch_offers(self, search=None, diskspace=None) -> List[VastOffer]:
        search = search or args.vastai_search or userconf.vastai_default_search
        diskspace = diskspace or const.VASTAI_DISK_SPACE

        search = f"{search} disk_space>{diskspace} verified=true"
        search_query = f"{search} disk_space>{diskspace} verified=true"
        out = await self.run_command(['search', 'offers', search_query])
        return [
            VastOffer(
                index=int(index),
                id=int(fields[0]),
                cuda=fields[1],
                num=int(fields[2].rstrip('x')),
                model=fields[3].replace('_', ' '),
                pcie=fields[4],
                cpu_ghz=float(fields[5]) if fields[5] != '-' else 0.0,
                vcpus=float(fields[6]),
                ram=float(fields[7]),
                disk=float(fields[8]),
                price=float(fields[9]),
                dlp=float(fields[10]),
                dlp_per_dollar=float(fields[11]),
                score=float(fields[12]),
                nv_driver=fields[13],
                net_up=float(fields[14]),
                net_down=float(fields[15]),
                r=float(fields[16]),
                max_days=float(fields[17]),
                mach_id=fields[18],
                status=fields[19],
                ports=fields[20],
                country=fields[21].replace('_', ' ')
            )
            for index, line in enumerate(out.splitlines()[1:], start=0)
            if len(fields := line.split()) >= 22
        ]

    async def fetch_instances(self) -> List[VastInstance]:
        out = await self.run_command(['show', 'instances', '--raw'])

        # Handle empty response
        if not out or out.isspace():
            log.error("Empty response from vast.ai")
            return []

        try:
            # Try to parse JSON directly
            instances_data = json.loads(out)
            ret = [
                VastInstance.from_api_response(instance_dataobj)
                for index, instance_dataobj in enumerate(instances_data, start=1)
            ]

            # Remove destroyed instances (the server is not updated immediately)
            ret = [instance for instance in ret if instance.id not in self.destroyed_instances]
            return ret
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse instances JSON: {e}")
            log.error(f"Raw output: {out}")
            return []
        except Exception as e:
            log.error(f"Unexpected error fetching instances: {e}")
            return []

    async def create_instance(self, offer_or_id)->Dict[str, Any]:
        """
        Return a dict from the request response, which as of writing looks like
        { new_contract:str, id:int }
        @param offer_or_id:
        @return:
        """
        if isinstance(offer_or_id, int):
            offer_id = offer_or_id
        else:
            offer_id = offer_or_id.id

        result = await self.run_command(['create', 'instance', offer_id, '--image', const.VASTAI_DOCKER_IMAGE, '--disk', const.VASTAI_DISK_SPACE, '--env', '-p 8188:8188', '--ssh'])
        if result.startswith('failed'):
            return None

        # remove til the first { and remove all after the last }
        # they literally put a json inside a string bruhhhhhhhhhhh cmon

        result = result[result.find('{'):result.rfind('}')+1]
        return ast.literal_eval(result)

    async def create_and_wait_remote(self, offer_or_id) -> Optional[LoomRemote]:
        async def wait_for_instance_creation(new_id):
            retry_delay = 3
            max_retry_delay = 60

            while True:
                try:
                    log.info(f"Fetching instance info for {new_id}")
                    infos = await self.fetch_instances()
                    if any(info.id == new_id for info in infos):
                        break
                except Exception as e:
                    log.error(f"Fetch data not ready yet, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(math.ceil(retry_delay * 1.3), max_retry_delay)
                else:
                    await asyncio.sleep(1)

        log.info(f"Creating instance with of: {offer_or_id}")

        result = await self.create_instance(offer_or_id)
        if result:
            from . import remote_manager

            new_id = result['new_contract']
            await wait_for_instance_creation(new_id)
            return await remote_manager.instance.get_instance(new_id)
        else:
            log.error("Failed to create instance. Retrying...")
            await asyncio.sleep(1.5)

    async def wait_for_ready(self, instance:VastInstance, timeout: Optional[float] = None) -> bool:
        if instance.status == "running":
            return True

        start_time = asyncio.get_running_loop().time()

        while True:
            try:
                instances = await self.fetch_instances()
                info = next((i for i in instances if i.id == instance.id), None)

                if info:
                    instance = info
                    if instance.status == "running":
                        log.info(f"Instance {instance.id} is ready")
                        return True

                log.info(
                    f"Waiting for instance {instance.id} to be ready (status: {instance.status if info else 'unknown'})"
                )

                if timeout and (asyncio.get_running_loop().time() - start_time > timeout):
                    log.warning(f"Timeout waiting for instance {instance.id} to be ready")
                    return False

                await asyncio.sleep(3)

            except Exception as e:
                log.error(f"Error while waiting for instance to be ready: {str(e)}")
                return False

    async def destroy_instance(self, instance_id):
        self.remotes.pop(instance_id, None)
        self.destroyed_instances.add(instance_id)
        return await self.run_command(['destroy', 'instance', str(instance_id)])

    async def reboot_instance(self, instance_id):
        return await self.run_command(['reboot', 'instance', str(instance_id)])

    async def stop_instance(self, instance_id):
        return await self.run_command(['stop', str(instance_id)])

    # Synchronous versions of the methods for compatibility
    def download_vast_script_sync(self):
        asyncio.run(self.download_vast_script())

    def run_command_sync(self, command):
        return asyncio.run(self.run_command(command))

    def fetch_balance_sync(self):
        return asyncio.run(self.fetch_balance())

    def fetch_offers_sync(self) -> List[VastOffer]:
        return asyncio.run(self.fetch_offers())

    def fetch_instances_sync(self) -> List[VastInstance]:
        return asyncio.run(self.fetch_instances())

    def create_instance_sync(self, offer_id):
        return asyncio.run(self.create_instance(offer_id))

    def destroy_instance_sync(self, instance_id):
        return asyncio.run(self.destroy_instance(instance_id))

    def reboot_instance_sync(self, instance_id):
        return asyncio.run(self.reboot_instance(instance_id))

    def stop_instance_sync(self, instance_id):
        return asyncio.run(self.stop_instance(instance_id))


instance = VastAIManager()

