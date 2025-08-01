from abc import ABC
from pathlib import Path

from src.gui.base_window import BaseWindow
from src.party.ModuleRecorder import ModuleRecorder

class ModuleWindow(BaseWindow, ABC):
	"""
	Base class for a window that can be used to edit a PyTorch module (nn.Module)
	"""
	def __init__(self, module, title, prop, torch_to_numpy):
		super().__init__(title)
		self._model = module
		self._torch_to_numpy = torch_to_numpy
		self.ckpt_name = "base"
		self.recorder_prop = 'grid'
		self.recorder = ModuleRecorder(module, self.recorder_prop, self._torch_to_numpy)
		self.recorder.on_stopped = self._on_recorder_stopped
		self.torch_to_numpy = torch_to_numpy
		self.last_video_path = None

	def _on_recorder_stopped(self, path:Path):
		self.last_video_path = path

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, value):
		self._model = value
		self.recorder = ModuleRecorder(value, self.recorder_prop, self.torch_to_numpy)
		self.recorder.on_stopped = self._on_recorder_stopped
