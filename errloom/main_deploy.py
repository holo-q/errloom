import asyncio
import logging
from typing import Optional

from errloom.deploy import ErrloomRemote_VLLMMan
from errloom.deploy import VastOffer
from errloom import argp
from src import storage
from errloom.interop import vast_manager
from errloom.deploy import make_header_text
from errloom.deploy import DeploymentStep
from errloom.deploy import ErrloomRemote
from errloom.deploy import VastInstanceView
from src.gui.utils import async_thread
from src.lib import loglib
from errloom.session import Session

logger = logging.getLogger("errloom_remote")

vastman = vast_manager.instance


async def start_errloom(remote: ErrloomRemote, session: Session):
    pass


async def _stop_errloom(remote: ErrloomRemote):
    remote.log_info(make_header_text("Stopping Errloom ..."))
    await remote.stop_errloom()


def launch_vllm_webui(instance: ErrloomRemote):
    vllm_url = ErrloomRemote_VLLMMan.get_server_url(instance.instance)
    if vllm_url:
        import webbrowser

        logger.info(f"(errloom:main) Opening VLLM interface in browser: {vllm_url}")
        webbrowser.open(vllm_url)
    else:
        if not hasattr(instance.instance, "public_ipaddr"):
            logger.error(
                "(errloom:main) Instance does not have a 'publicip' attribute. Unable to open browser interface."
            )
        elif not instance.instance.ports.get("8000/tcp"):
            logger.error(
                "(errloom:main) VLLM port (8000) not mapped. Unable to open browser interface."
            )
        else:
            logger.error(
                "(errloom:main) Unable to open browser interface due to missing information."
            )


async def create_remote(session):
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.application import Application
    from errloom.deploy import forget
    from prompt_toolkit.key_binding import KeyBindings
    from errloom.deploy import OfferList

    logger.info("(errloom:main) cmd_create_instance: entry")

    class Result:
        def __init__(self):
            self.remote: Optional[ErrloomRemote] = None
            self.confirmed = False

    result = Result()

    async def on_confirm(offer: VastOffer):
        logger.info(f"(errloom:main) OfferList.confirm_offer: {offer}")
        remote = await vastman.create_and_wait_remote(offer)
        if remote:
            await vastman.wait_for_ready(remote.instance)
            logger.info(
                f"(errloom:main) Instance {remote.id} has been successfully created and is ready for use."
            )
            result.remote = remote
            result.confirmed = True
            app.exit()
        else:
            logger.error("(errloom:main) Failed to create instance.")

    # OfferList for selecting an instance type
    offer_list = OfferList(on_confirm)
    forget(offer_list.fetch_offers())
    layout = Layout(HSplit([offer_list]))

    kb = KeyBindings()

    @kb.add("q")
    def _(event):
        app.exit()

    app = Application(
        layout=layout,
        full_screen=True,
        mouse_support=True,
        style=offer_list.style,
        key_bindings=kb,
    )

    await app.run_async()

    if session and result.confirmed and result.remote:
        await result.remote.connect()
        await result.remote.send_session(session)

    return result.remote


async def choose_instance(session):
    """
    Displays a list of available instances and allows user to select one.
    Uses InstanceList for display and selection, similar to create_instance's OfferList.
    """
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from errloom.deploy import forget
    from errloom.deploy import InstanceList

    logger.info("(errloom:main) choose_instance: entry")

    ret: Optional[VastInstanceView] = None

    def on_confirm(remote):
        nonlocal ret
        ret = remote
        app.exit()

    # Create instance list view
    list = InstanceList()
    list.handler = on_confirm
    layout = Layout(HSplit([list]))
    kb = KeyBindings()

    @kb.add("q")
    def _(event):
        app.exit()

    app = Application(
        layout=layout, full_screen=True, mouse_support=True, key_bindings=kb
    )

    # Fetch and display instances
    forget(list.fetch_instances())
    await app.run_async()

    return ret._instance if ret else None


async def main():
    # deploy_logging.setup_logging(cli=True)

    async_thread.setup()

    args = argp.errlargs
    session = argp.get_errloom_session()
    if session is None:
        logger.error("(errloom:main) A session is required to deploy on a remote instance.")
        return

    # TODO we have to decide if we want to choose an instance or use the one stored in the session.
    # I'm prolly not gonna do multiple instances for a while, so it makes no difference for now.

    instances = await vastman.fetch_instances()
    if not instances:
        # Create instance
        # ----------------------------------------
        remote = await create_remote(session)
        if not remote:
            logger.error("(errloom:main) Failed to create instance.")
            return
    else:
        instance = storage.application.vast_mappings.get(session.name, None)
        if instance is not None:
            # Load up session instance
            # ----------------------------------------
            if any(i.id == instance.id for i in instances):
                remote = await ErrloomRemote.ConnectNew(instance, session)
                status = await remote.probe_deployment_status()
                if status.is_errloom_running:  # errloom is allowed to continue running on the remote, without errloom active
                    await remote.syncman.start()
            else:
                storage.application.vast_mappings.pop(session.name, None)
                logger.info(
                    "(errloom:main) Removing expired remote instance from errloom. (not existing on vast)"
                )

        # Assign existing instance
        # ----------------------------------------
        if len(instances) == 1:
            instance = instances[0]
        else:
            instance = await choose_instance(session)
            if not instance:
                logger.error("(errloom:main) No instance selected.")
                return

        remote = await ErrloomRemote.ConnectNew(instance, session)

    if args.vastai_destroy:
        logger.info("(errloom:main) Destroying VastAI instance in 3 seconds...")
        await asyncio.sleep(3)
        await vastman.destroy_instance(instance.id)
        logger.info("(errloom:main) VastAI instance destroyed.")
        return

    # Kill processes to prevent duplicates
    # ----------------------------------------
    if args.vastai_vllm:
        await remote.vllmman.kill()
    else:
        await remote.stop_errloom()

    # Auto-deploy/start on first connect (so we can step away from the computer and come back to a render in progress)
    # ----------------------------------------
    if not await remote.probe_apt_packages_installed():
        await remote.apt_install()

    just_deployed = False
    status = await remote.probe_deployment_status()
    if status.installation_step.value < DeploymentStep.DONE.value or args.vastai_redeploy:
        await remote.deploy(redeploy=args.vastai_redeploy)
        just_deployed = True
    else:
        # These will all be done by deploy, hence the arguments are ignored if we are just deploying
        if args.vastai_upgrade:
            await remote.pip_upgrade()
        if args.vastai_copy:
            await remote.send_fast_uploads()
        if args.vastai_install:
            await remote.apt_install()
        if args.vastai_upgrade:
            await remote.pip_upgrade()

    if args.vastai_mount:
        logger.info("(errloom:main) Mounting remote instance ...")
        path = await remote.mount()
        logger.info(f"(errloom:main) Mounted at {path}")

    await remote.send_session()

    if args.dry:
        return

    if session:
        storage.application.vast_mappings[session.name] = remote.instance
        storage.application.write()

    # Set API tokens
    # ----------------------------------------
    import userconf

    hf_token = getattr(userconf, 'hf_token', '')
    wandb_token = getattr(userconf, 'wandb_token', '')

    # Setup Hugging Face token
    await remote.ssh.run(f"mkdir -p ~/.huggingface")
    await remote.ssh.run(f"echo '{hf_token}' > {remote.dst_hf_token.as_posix()}")

    # Setup W&B token
    netrc_entry = f"machine api.wandb.ai\n  login user\n  password {wandb_token}\n"
    await remote.ssh.run(f"echo '{netrc_entry}' > {remote.dst_wandb_token.as_posix()}")
    await remote.ssh.run(f"chmod 600 {remote.dst_wandb_token.as_posix()}")

    if args.vastai_vllm:
        # clear the vllm output folder
        logger.info("(errloom:main) Clearing VLLM output folder ...")
        await remote.ssh.run(
            f"sudo rm  -rf {(remote.dst_errloom / 'output').as_posix()}/*"
        )

        logger.info("(errloom:main) Starting VLLM ...")
        while True:
            await remote.vllmman.start(force_restart=True)
            if just_deployed:
                launch_vllm_webui(remote)
            await asyncio.sleep(9999999999999999)
    else:
        # Check if vllm is running or not
        if not await remote.vllmman.is_running():
            logger.info("(errloom:main) VLLM is not running, starting it ...")
            await remote.vllmman.start()


        logger.info("(errloom:main) Starting Errloom process loop ...")
        logger.info("(errloom:main) If errloom crashes, it will be restarted automatically.")

        # Auto-restart loop
        while True:
            logger.info("(errloom:main) Starting Errloom ...")
            await remote.run_errloom()
