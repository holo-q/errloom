import importlib
import inspect
import logging
import os
import pkgutil
import sys
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
        logger.debug(f"📋 [dim]Package '{package_name}' already crawled[/dim]")
        return

    from errloom.loom import Loom
    from errloom.comm import CommModel
    from errloom.attractor import Attractor
    DEFAULT_CLASSES = [Loom, CommModel, Attractor]  # TODO HoloFunca

    base_classes = base_classes or []
    base_classes.extend(DEFAULT_CLASSES)

    logger.debug(f"🔍 [bold cyan]CRAWLING[/bold cyan] {package_name}")
    
    # Log search criteria on one line
    criteria = []
    if base_classes:
        criteria.append(f"bases: {[cls.__name__ for cls in base_classes]}")
    if check_has_attr:
        criteria.append(f"attrs: {check_has_attr}")
    if criteria:
        logger.debug(f"   🎯 {' | '.join(criteria)}")

    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        logger.debug(f"   ❌ [red]Import failed: {e}[/red]")
        return

    if not hasattr(package, "__file__") or package.__file__ is None:
        logger.debug(f"   ⚠️  [yellow]No __file__, skipping namespace package[/yellow]")
        return

    package_path = os.path.dirname(package.__file__)
    registered_count = 0
    module_count = 0
    
    def visit_module(module_info):
        nonlocal registered_count, module_count
        module_count += 1
        full_module_name = module_info.name
        
        try:
            module = importlib.import_module(full_module_name)
            
            registered_in_module = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    should_register = False
                    match_reason = None
                    
                    if base_classes:
                        for base in base_classes:
                            if issubclass(obj, base) and obj is not base:
                                should_register = True
                                match_reason = f"→{base.__name__}"
                                break

                    if not should_register and check_has_attr:
                        for attr in check_has_attr:
                            if hasattr(obj, attr):
                                should_register = True
                                match_reason = f"@{attr}"
                                break

                    if should_register:
                        _class_registry[obj.__name__] = obj
                        registered_count += 1
                        registered_in_module.append(f"{name}{match_reason}")
            
            # Only log modules that registered classes
            if registered_in_module:
                module_short = full_module_name.replace(f"{package_name}.", "")
                logger.debug(f"   📦 [green]{module_short}[/green]: {', '.join(registered_in_module)}")

        except Exception as e:
            logger.debug(f"   💥 [red]{module_info.name.split('.')[-1]}: {str(e)[:50]}[/red]")

    for module_info in pkgutil.walk_packages([package_path], prefix=f"{package_name}."):
        visit_module(module_info)

    _crawled_packages.add(package_name)
    
    logger.debug(f"   ✅ [bold green]COMPLETE[/bold green] {registered_count}/{module_count} classes/modules → {len(_class_registry)} total")


def get_class(name: str) -> Type[Any] | None:
    """
    Retrieves a class from the registry by its name.
    """
    result = _class_registry.get(name)
    if logger.isEnabledFor(logging.DEBUG):
        status = "✅" if result else "❌"
        logger.debug(f"🔍 {name} {status}")
    return result


def get_all_classes() -> Dict[str, Type[Any]]:
    """
    Returns the entire class registry.
    """
    logger.debug(f"📚 [cyan]Registry dump[/cyan]: {len(_class_registry)} classes")
    return _class_registry.copy()

_holo_classes_cache = {}
_cache_valid = False

def find_holo_classes():
    global _holo_classes_cache, _cache_valid

    if _cache_valid:
        logger.debug(f"🎯 [green]Holo cache hit[/green]: {len(_holo_classes_cache)} classes")
        return _holo_classes_cache

    logger.debug(f"🔍 [bold cyan]SCANNING HOLO CLASSES[/bold cyan]")
    
    holo_classes = {}
    modules = dict(sys.modules)
    found_holo_classes = 0
    
    for module_name, module in modules.items():
        if module is None:
            continue

        try:
            module_holo_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, '__holo__') and callable(getattr(obj, '__holo__')):
                    holo_classes[name] = obj
                    found_holo_classes += 1
                    module_holo_classes.append(name)
            
            # Only log modules with holo classes
            if module_holo_classes:
                module_short = module_name.split('.')[-1] if '.' in module_name else module_name
                logger.debug(f"   🎭 [green]{module_short}[/green]: {', '.join(module_holo_classes)}")
                
        except Exception:
            continue

    _holo_classes_cache = holo_classes
    _cache_valid = True
    
    logger.debug(f"   ✅ [bold green]COMPLETE[/bold green] {found_holo_classes} holo classes found")
    
    return holo_classes

def resolve_holo_class(class_name):
    holo_classes = find_holo_classes()
    result = holo_classes.get(class_name)
    
    if logger.isEnabledFor(logging.DEBUG):
        status = "✅" if result else "❌"
        logger.debug(f"🎯 {class_name} {status}")
    
    return result

def invalidate_holo_cache():
    global _cache_valid
    logger.debug(f"🗑️  [yellow]Holo cache invalidated[/yellow]")
    _cache_valid = False
