import typing
from dataclasses import dataclass

from errloom.deploy.execution_state import ExecutionState

if typing.TYPE_CHECKING:
    from errloom.ssh import SSHState
    from errloom.deploy.remote import LoomRemote


@dataclass
class RemoteView:
    """
    Provides a UI view for the DiscoInstance class with only
    the necessary information for the UI.
    """
    id: int
    ip: str
    port: int
    machine: str
    ssh: 'SSHState'
    mounted: bool
    discore: 'ExecutionState'
    comfy: 'ExecutionState'
    _remote: 'LoomRemote' = None

    @property
    def info(self):
        return self._remote.instance

    @classmethod
    async def from_remote(cls, remote: 'LoomRemote') -> 'RemoteView':
        """

        @rtype: object
        """
        status = await remote.probe_deployment_status()

        # TODO verify what we're getting back here
        return cls(
            id=int(remote.id),
            ip=remote.ip,
            port=int(remote.port),
            machine=remote.instance.status,
            ssh=remote.connection_state,
            mounted=await remote.ssh.is_mounted(),
            discore=ExecutionState.from_bool(await remote.is_discore_running()),
            comfy=ExecutionState.from_bool(await remote.is_comfy_running()),
            _remote=remote
        )