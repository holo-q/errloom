# from src.deploy.DeployUI import UIState, change_state, change_state_dialog, log
import asyncio
import logging
from typing import Optional, Union

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Dimension, Float, HSplit, Layout, VSplit
from prompt_toolkit.layout.containers import ConditionalContainer, FloatContainer
from prompt_toolkit.widgets import Box, Frame, Label, TextArea

from errloom.deploy.deploy_utils import forget, invalidate, make_header_text
from errloom.deploy.remote import LoomRemote
from errloom.ssh import SSHState
from errloom.tui.app_button import AppButton
from errloom.tui.log_display import LogDisplay
from errloom.tui.spacer import Spacer
from errloom.tui.ui_vars import app, kb, rman, vast, Window
from errloom.interop.vast_instance import VastInstance

log = logging.getLogger("TUI")

class ManageLayout(Layout):
    """
    ManageLayout class manages the UI layout for a remote instance.

    Flow and API:
    1. Initialization:
       - Creates text areas for logs, session, and toolbar
       - Initializes with an empty window
       - Calls update_window() to set up the initial layout

    2. Content Updates:
       - update_content() is called to refresh the displayed information
       - Updates balance, session, logs, and toolbar text
       - Automatically scrolls log to bottom if not manually scrolled

    3. Window Updates:
       - update_window() is called to recreate the entire layout
       - This happens on initialization and when significant changes occur

    4. Refresh Cycle:
       - _refresh() method runs periodically to update instance status
       - Fetches new balance and deployment status
       - Triggers update_content() to reflect changes

    5. Remote Management:
       - set_remote() method associates a DiscoInstance instance
       - Starts the refresh cycle when a remote is set

    6. User Interactions:
       - Various async methods (_disconnect, _deploy, etc.) handle user actions
       - These methods update the remote instance and trigger UI updates

    Key Properties:
    - info: VastInfo data associated with the current remote
    - id: ID of the current VastInfo
    """

    # Refresh interval in seconds
    REFRESH_INTERVAL = 5

    def __init__(self):
        super().__init__(Window())

        self.txt_logs = LogDisplay()
        self.txt_session = None
        self.txt_toolbar = None
        self.remote: Optional[LoomRemote] = None
        self.window = None
        self._last_status = None
        self._last_balance = None
        self._refresh_task = None

        self.txt_toolbar = TextArea(multiline=False,
            height=1,
            style="bg:#CCCCCC #000000",
            read_only=True)
        self.txt_session = TextArea(multiline=False, height=1)

        # self.on_back = lambda: app.change_state(UIState.MAIN)
        self.on_back = lambda: None

        self.update_window()

    @property
    def info(self):
        return self.remote.instance if self.remote else None

    @property
    def id(self):
        return self.info.id if self.info else None

    def update_content(self):
        log.debug("Updating content in DeployTUI")

        if self._last_balance is not None:
            self.txt_toolbar.text = f'Balance: {self._last_balance:.02f}$'
        else:
            self.txt_toolbar.text = "Balance: Loading..."

        if self.remote:
            self.txt_session.text = self._last_status.work_session
            if self.remote.ssh:
                new = self.remote.ssh.logs.text
                self.txt_logs.update_text(new)

        if self._last_status:
            toolbar_text = self.txt_toolbar.text
            toolbar_text += f" | ComfyUI: {'Running' if self._last_status.is_comfy_running else 'Stopped'}"
            toolbar_text += f" | Discore: {'Running' if self._last_status.is_discore_running else 'Stopped'}"
            self.txt_toolbar.text = toolbar_text

        invalidate()

    def _update(self):
        forget(self.update())

    async def update(self):
        log.debug("ManageLayout.update")
        log.info("1")
        self.remote.refresh_instance_status()
        log.info("2")
        self.update_content()
        log.info("3")

        status = await self.remote.probe_deployment_status()
        self._last_status = status

        text = f""
        for k, v in status.__dict__.items():
            text += f"{k}: {v}\n"

        self.update_window()

    def update_window(self):
        @Condition
        def is_connected():
            return self.remote and self.remote.connection_state == SSHState.CONNECTED

        @Condition
        def can_deploy():
            remote = self.remote
            return self.remote and remote.instance and remote.instance.status == "running" and self.txt_session.text is not None

        @Condition
        def is_running():
            return self.remote and self._last_status and self.info.status == "running"

        @Condition
        def is_comfy_running():
            return self.remote and self._last_status and self._last_status.is_comfy_running

        @Condition
        def is_discore_running():
            return self.remote and self._last_status and self._last_status.is_discore_running

        log.debug(f"Updating window for instance {self.id} (balance: {self._last_balance}, status: {self._last_status}")
        self.update_content()

        left_col = [
            Label("Commands"),
            # AppButton("Refresh1", self._refresh, key='r'),
            Spacer(2),
            AppButton("Refresh", self._update, key='r'),
            Spacer(2),
            ConditionalContainer(AppButton("Start", self._start, key='s'), ~is_running),
            ConditionalContainer(AppButton("Stop", self._stop, key='t'), is_running),
            ConditionalContainer(AppButton("Connect", self._connect, key='c'), ~is_connected),
            ConditionalContainer(AppButton("Disconnect", self._disconnect, key='C'), is_connected),
            ConditionalContainer(HSplit([
                ConditionalContainer(AppButton("Shell", self._shell, key='h'), is_running),
                ConditionalContainer(AppButton("Mount", self._mount, key='m'), is_running),
            ]), is_running),
            Spacer(1),
            ConditionalContainer(HSplit([
                self.txt_session,
                AppButton("Deploy", self._deploy, key='d'),
                AppButton("1 Send fast", self._send_fast, key='1'),
                AppButton("2 Send slow", self._send_slow, key='2'),
                AppButton("3 Pip Upgrades", self._pipupgrades, key='3'),
                ConditionalContainer(AppButton("4 Start Comfy", self._start_comfy, key='4'), ~is_comfy_running),
                ConditionalContainer(AppButton("4 Stop Comfy", self._start_comfy, key='4'), is_comfy_running),
                ConditionalContainer(AppButton("5 Start Discore", self._start_discore, key='5'), ~is_discore_running),
                ConditionalContainer(AppButton("5 Stop Discore", self._stop_discore, key='5'), is_discore_running),
                Spacer(),
            ]), is_connected),
            AppButton("Destroy", self._destroy, key='D'),
            Spacer(1),
            AppButton("Back", handler=lambda: self.on_back),
        ]

        main_container = Box(
            padding_left=4,
            padding_right=4,
            padding_top=2,
            padding_bottom=2,
            body=HSplit([
                self.txt_toolbar,
                Spacer(),
                VSplit([
                    HSplit(left_col, padding=0),
                    Frame(self.txt_logs, title="Logs", height=Dimension(preferred=9999999999999)),
                ], padding=4),
            ]),
            key_bindings=kb
        )

        # app.key_bindings.add
        self.window = Layout(FloatContainer(
            content=Window(),
            floats=[
                Float(
                    left=2, right=2, top=2, bottom=2,
                    content=Frame(
                        title="Manage Instance",
                        body=main_container),
                )
            ]
        ))

        invalidate()

    async def change(self, info_or_instance: Union[VastInstance, LoomRemote, int, None]):
        if self.remote is not None:
            self.remote.ssh.logs.handlers.remove(self._on_ssh_log_line)

        match info_or_instance:
            case id if isinstance(id, int) and not self.id == id:
                self.remote = await rman.get_remote(id)
            case info if isinstance(info, VastInstance) and not self.id == info.id:
                self.remote = await rman.get_remote(info)
            case remote if isinstance(remote, LoomRemote):
                self.remote = remote
            case None:
                self.remote = None

        self.update_window()
        invalidate()

    async def _connect(self):
        await self.remote.connect()
        app.change_state_dialog.info("Connecting ...", f"Connecting to root@{self.remote.ip}:{self.remote.port} (instance #{self.remote.instance.id})...")

        log.info(f"Connected to {self.remote.ip}:{self.remote.port}")

        if self.remote.connection_state == SSHState.HOST_KEY_NOT_VERIFIABLE:
            app.change_state_dialog.info("Connecting", "Host key not verifiable, opening SSH in terminal to ask ...")
            await self.remote.shell()
            await self._connect()
            return

        app.change_state(UIState.MANAGE_INSTANCE)
        self.update_window()

    async def refresh_balance(self):
        if self.remote:
            self._last_balance = await vast.fetch_balance()
            log.info(f"Retrieved balance: {self._last_balance}")
            self.update_window()

    def start_refresh_task(self):
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_balance_loop())

    def stop_refresh_task(self):
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

    async def _refresh_balance_loop(self):
        while True:
            log.debug("Refreshing balance from _refresh_balance_loop...")
            await self.refresh_balance()
            await asyncio.sleep(30)


    async def _start(self):
        forget(vast.start_instance(self.info.id))

    async def _stop(self):
        forget(vast.stop_instance(self.info.id))

    async def _reboot(self):
        forget(vast.reboot_instance(self.info.id))
        await self.update()

    async def _destroy(self):
        forget(vast.destroy_instance(self.info.id))
        app.change_state(UIState.MAIN)
        # TODO show a blocking popup for like 1 second (if you refresh too soon after destroying, the instance will still be there)

    async def _disconnect(self):
        self.remote.log_info(make_header_text("Disconnecting ..."))
        await self.remote.disconnect()
        self.update_window()

    async def _deploy(self):
        await self.remote.deploy(self.txt_session.text)
        self.remote.log_info("== DONE ==")

    async def _mount(self):
        await self.remote.mount()

    async def _shell(self):
        await self.remote.shell()

    async def _send_fast(self):
        self.remote.log_info(make_header_text("Sending fast uploads ..."))
        await self.remote.send_fast_uploads()
        self.remote.log_info("== DONE ==")

    async def _send_slow(self):
        self.remote.log_info(make_header_text("Sending slow uploads ..."))
        await self.remote.send_slow_uploads()
        self.remote.log_info("== DONE ==")

    async def _pipupgrades(self):
        self.remote.log_info(make_header_text("Running pip upgrade ..."))
        await self.remote.pip_upgrade()
        self.remote.log_info("== DONE ==")

    async def _start_comfy(self):
        self.remote.log_info(make_header_text("Starting Comfy ..."))
        await self.remote.start_comfy()
        await self.update()
        self.remote.log_info("== DONE ==")

    async def _stop_comfy(self):
        self.remote.log_info(make_header_text("Stopping Comfy ..."))
        await self.remote.stop_comfy()
        await self.update()
        self.remote.log_info("== DONE ==")

    async def _start_discore(self):
        self.remote.log_info(make_header_text("Starting Discore ..."))
        await self.remote.run_discore()
        await self.update()
        self.remote.log_info("== DONE ==")

    async def _stop_discore(self):
        self.remote.log_info(make_header_text("Stopping Discore ..."))
        await self.remote.stop_discore()
        await self.update()
        self.remote.log_info("== DONE ==")

    def _on_ssh_log_line(self, _):
        self.update_content()

    def __pt_container__(self):
        return self.window
