# This class allows a persistent storage of data in a json file
# We do not store anything in memory, instead we read and write
# to the file every time we need to access the data
# We use pathlib as much as possible
import logging
import traceback
from pathlib import Path

import dill
import msgspec

from errloom import paths
from errloom.interop.vast_instance import VastInstance
from errloom.paths import root


log = logging.getLogger("Storage")


class Storage(msgspec.Struct):
	_path: str = ""

	@property
	def path(self) -> Path:
		return Path(self._path)

	@path.setter
	def path(self, value: Path | str):
		self._path = str(value) if isinstance(value, Path) else value

	def write(self):
		match self.path.suffix:
			case ".json":
				self.write_json()
			case ".dill":
				self.write_dill()

	def read(self, path=None):
		path = path or self._path
		if not path:
			log.error("Cannot read storage, no path provided")
			return

		path = Path(path)

		match self.path.suffix:
			case ".json":
				self.read_json()
			case ".dill":
				self.read_dill()

	def write_json(self):
		serializable_data = {name: getattr(self, name) for name in self.__struct_fields__ if not name.startswith("_")}
		encoder = msgspec.json.Encoder()
		with self.path.open("wb") as f:
			encoded = encoder.encode(serializable_data)
			formatted = msgspec.json.format(encoded, indent=2)
			f.write(formatted)
		log.debug(f"write_json -> {self._path}")

	def read_json(self):
		log.debug(f"read_json -> {self._path}")
		try:
			if self.path.exists():
				with self.path.open("r") as f:
					loaded_data = msgspec.json.decode(f.read(), type=type(self))
					for name in self.__struct_fields__:
						if not name.startswith("_") and hasattr(loaded_data, name):
							setattr(self, name, getattr(loaded_data, name))
		except Exception as e:
			log.error(f"Invalid json, cannot load {self._path}: {e}")

	def _reconstruct_objects(self, data):
		if isinstance(data, dict):
			if "__path__" in data:
				return Path(data["__path__"])
			if "__dataclass__" in data:
				dataclass_name = data["__dataclass__"]
				dataclass_module = data["__module__"]

				module = __import__(dataclass_module, fromlist=[dataclass_name])
				dataclass = getattr(module, dataclass_name)
				data = {k: v for k, v in data.items() if k not in ("__dataclass__", "__module__")}
				for key, value in data.items():
					data[key] = self._reconstruct_objects(value)

				return dataclass(**data)
			else:
				return {k: self._reconstruct_objects(v) for k, v in data.items()}
		elif isinstance(data, list):
			return [self._reconstruct_objects(item) for item in data]
		else:
			return data

	def write_dill(self):
		serializable_data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
		with self.path.with_suffix(".dill").open("wb") as f:
			dill.dump(serializable_data, f)
		log.debug(f"write_dill -> {self._path}")

	def read_dill(self):
		log.debug(f"read_dill -> {self._path}")
		try:
			dill_path = self.path.with_suffix(".dill")
			if dill_path.exists():
				with dill_path.open("rb") as f:
					loaded_data = dill.load(f)
				for key, value in loaded_data.items():
					if not key.startswith("_"):
						setattr(self, key, value)
		except Exception as e:
			log.error(f"Invalid dill, cannot load {self._path}")
			log.error(traceback.format_exc())


class AppStorage(Storage):
	recent_sessions: list = []
	open_sessions: list = []
	cached_session_paths_detected: list = []
	auto_layout: bool = True
	vast_instances: list[VastInstance] = []
	vast_mappings: dict[str, VastInstance] = {}
	last_args: list[str] = []

	def get_recent_sessions(self):
		return [Path(path) for path in self.recent_sessions if Path(path).exists()]


application = AppStorage((paths.root / "storage.json").as_posix())
try:
	application.read_json()
except Exception as e:
	print("Invalid storage.json!")


def store_last_session_name(name):
	"""
	Write the session name to root/last_session.txt
	"""
	with open(root / "last_session.txt", "w") as f:
		f.write(name)


def fetch_last_session_name():
	"""
	Read the session name from root/last_session.txt
	"""
	try:
		with open(root / "last_session.txt", "r") as f:
			return f.read()
	except FileNotFoundError:
		return None


def has_last_session():
	return Path(root / "last_session.txt").exists()

# ---- CLI last-args helpers ---------------------------------------------------
def save_last_args(args: list[str]) -> None:
	"""
	Persist the latest CLI args used (argv slice, excluding program name).
	"""
	try:
		application.last_args = list(args) if args else []
		application.write_json()
		log.debug(f"save_last_args -> {application.last_args}")
	except Exception:
		log.error("Failed to save last_args")
		log.error(traceback.format_exc())


def load_last_args() -> list[str] | None:
	"""
	Load the last saved CLI args, or None if not available.
	"""
	try:
		# Ensure freshest read in case of external edits
		application.read_json()
		if getattr(application, "last_args", None):
			return list(application.last_args)
		return None
	except Exception:
		log.error("Failed to load last_args")
		log.error(traceback.format_exc())
		return None


def has_last_args() -> bool:
	la = load_last_args()
	return bool(la and isinstance(la, list) and len(la) > 0)
