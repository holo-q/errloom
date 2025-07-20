from src import paths

# Constants
VAST_PYTHON_BIN = '/opt/conda/bin/python3'
VAST_PIP_BIN = '/opt/conda/bin/pip3'
VASTAI_DOCKER_IMAGE = 'pytorch/pytorch'
VASTAI_DISK_SPACE = '50'

# These packages will be installed when first connecting to the instance
APT_PACKAGES = [
    'python3-venv', 'libgl1', 'zip', 'ffmpeg', 'gcc', 'tree', 'psmisc', 'git'
]

# Files to upload every time we connect
FAST_UPLOADS = [
    ('requirements-vastai.txt', 'requirements.txt'),
    'requirements-gpu.txt',
    'setup.py',
    'errloom.py',
    'jargs.py',
    paths.userconf,
    paths.user_holowares,
    # paths.src_plugins_name,
    paths.root_src,
]

# Files to upload the first time we install the instance or when specified
SLOW_UPLOADS = [
    # paths.plug_res_name
]

DEPLOY_UPLOAD_BLACKLIST_PATHS = [
    "*.mp4", "video__*.mp4", "*.jpg", "__pycache__", "tmp", "*.pth", "*.safetensors"
]

SYNCMAN_UPLOAD_INCLUSIONS = [
    paths.root_scripts,
    paths.root_src,
    "{SESSION}/script.py",
    "{SESSION}/workflow.json",
]

SYNCMAN_DOWNLOAD_EXCLUSIONS = [
    "video.mp4",
    "video__*.mp4",
    "script.py",
    "*.npy",
    "__pycache__/*",
    "tmp/*",
    "workflow.json",
    "htdemucs_6s",
    *[str(x).replace("{SESSION}/", "") for x in SYNCMAN_UPLOAD_INCLUSIONS]
]

MODEL_URLS = [
    'https://huggingface.co/Qwen/Qwen3-4B'  # Example model, can be customized per session
]

enable_auto_connect = True
ERRLOOM_MAIN_PY = "errloom.py" # main.py -> errloom.py so we can target the process
