import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional, Type


logger = logging.getLogger(__name__)

_class_registry: Dict[str, Type[Any]] = {}
_crawled_packages: set[str] = set()


def crawl_package(
    package_name: str,
    base_classes: Optional[List[Type[Any]]] = None,
    check_has_attr: Optional[List[str]] = None,
):
    """
    Crawls a given package to find and register classes based on specified criteria.

    The function uses importlib and pkgutil to recursively walk through all modules in the specified package.
    For each module, it inspects all classes and registers those that match the criteria:
    - It is a subclass of any class in `base_classes`.
    - It has any attribute listed in `check_has_attr`.

    If a class with the same name is already registered, a warning is logged.
    The function also keeps track of which packages have already been crawled to avoid redundant work.
    """
    if package_name in _crawled_packages:
        return

    from errloom import Attractor, CommModel, Loom
    DEFAULT_CLASSES = [Loom, CommModel, Attractor]  # TODO HoloFunca

    base_classes = base_classes or []
    base_classes.extend(DEFAULT_CLASSES)

    logger.debug(f"Crawling package '{package_name}' for classes...")

    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        logger.error(f"Could not import package {package_name}: {e}")
        return

    if not hasattr(package, "__file__") or package.__file__ is None:
        logger.debug(f"Package '{package_name}' has no __file__, cannot crawl. Likely a namespace package.")
        return

    def visit_module(module_info):
        full_module_name = f"{package_name}.{module_info.name}"
        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    should_register = False
                    if base_classes:
                        for base in base_classes:
                            if issubclass(obj, base) and obj is not base:
                                should_register = True
                                break

                    if not should_register and check_has_attr:
                        for attr in check_has_attr:
                            if hasattr(obj, attr):
                                should_register = True
                                break

                    if should_register:
                        _class_registry[obj.__name__] = obj
                        logger.debug(f"- {obj.__module__}.{obj.__name__}") # TODO color the parts

        except Exception as e:
            logger.debug(f"Could not import or inspect module {full_module_name}: {e}")

    package_path = os.path.dirname(package.__file__)
    for module_info in pkgutil.walk_packages([package_path]):
        # print(module_info.name)
        visit_module(module_info)

    _crawled_packages.add(package_name)


def get_class(name: str) -> Type[Any] | None:
    """
    Retrieves a class from the registry by its name.
    """
    return _class_registry.get(name)


def get_all_classes() -> Dict[str, Type[Any]]:
    """
    Returns the entire class registry.
    """
    return _class_registry.copy()
