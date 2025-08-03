"""
Generic class discovery module with lazy loading.

This module provides flexible class discovery capabilities for Python packages.
It can find classes based on inheritance, attributes, or method signatures without
eagerly importing all modules.

Architecture:
- _class_index: Maps class names to module names (built during crawling)
- _class_registry: Maps class names to actual class objects (populated on-demand)

During crawling, we scan files for class definitions using text analysis.
Classes are only imported when requested via get_class() or get_all_classes().

Main Functions:
- crawl_package_fast(): Lazy indexing with configurable criteria
- crawl_package(): Legacy eager loading
- get_class(): Retrieve class with lazy loading
- find_special_classes(): Find classes with specific methods
- resolve_special_class(): Resolve class by name and method criteria

This module is framework-agnostic and can be used with any Python project.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set

logger = logging.getLogger(__name__)

_class_registry: Dict[str, Type[Any]] = {}  # Actual imported classes (lazy cache)
_class_index: Dict[str, str] = {}  # class_name -> module_name mapping
_crawled_packages: set[str] = set()
_text_scan_cache: Dict[str, Dict[str, Set[str]]] = {}  # module_path -> {pattern_type -> class_names}


def scan_file_for_classes(file_path: Path, base_class_names: Set[str], method_patterns: Optional[List[str]] = None) -> Dict[str, Set[str]]:
    """
    Scan a Python file for class definitions without importing it.
    
    Args:
        file_path: Path to the Python file to scan
        base_class_names: Set of base class names to look for in inheritance
        method_patterns: List of method name patterns to search for (e.g., ['__holo__', '__call__'])
    
    Returns:
        Dict with keys 'inheritance' and 'special_methods' mapping to sets of class names
    """
    cache_key = str(file_path)
    if cache_key in _text_scan_cache:
        return _text_scan_cache[cache_key]
    
    method_patterns = method_patterns or []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        # Skip files that can't be read as UTF-8 or don't exist
        return {'inheritance': set(), 'special_methods': set()}
    
    inheritance_classes = set()
    special_method_classes = set()
    
    # Pattern for class definitions (with or without inheritance)
    # Matches: class SomeName: or class SomeName(BaseClass): or class SomeName(Base1, Base2):
    class_pattern = r'^class\s+(\w+)(?:\s*\(\s*([^)]*)\s*\))?\s*:'
    
    # Build patterns for special methods
    method_regex_patterns = []
    for method_name in method_patterns:
        # Escape regex special characters and create pattern
        escaped_method = re.escape(method_name)
        pattern = rf'^\s+(async\s+)?def\s+{escaped_method}\s*\('
        method_regex_patterns.append(pattern)
    
    lines = content.split('\n')
    current_class = None
    
    for line in lines:
        # Check for class definitions
        class_match = re.match(class_pattern, line)
        if class_match:
            class_name = class_match.group(1)
            inheritance_list = class_match.group(2)  # May be None if no inheritance
            current_class = class_name
            
            # Check if any of the base classes match our target base classes
            if inheritance_list and base_class_names:
                for base_name in base_class_names:
                    if base_name in inheritance_list:
                        inheritance_classes.add(class_name)
                        break
        
        # Check for special methods within classes
        elif current_class and method_regex_patterns:
            for pattern in method_regex_patterns:
                if re.match(pattern, line):
                    special_method_classes.add(current_class)
                    break
        
        # Reset current_class when we exit the class (crude but effective)
        elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            if not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                current_class = None
    
    result = {'inheritance': inheritance_classes, 'special_methods': special_method_classes}
    _text_scan_cache[cache_key] = result
    return result


def crawl_package_fast(
    package_name: str,
    base_classes: Optional[List[Type[Any]]] = None,
    check_has_attr: Optional[List[str]] = None,
    method_patterns: Optional[List[str]] = None,
    skip_patterns: Optional[List[str]] = None,
):
    """
    Fast text-based package crawling that scans files without importing them.
    Classes are indexed by location but only imported when requested.
    
    Args:
        package_name: Name of the package to crawl
        base_classes: List of base classes to match against
        check_has_attr: List of attribute names to check for  
        method_patterns: List of method names to search for (e.g., ['__holo__', '__call__'])
        skip_patterns: List of module name patterns to skip during crawling
    """
    if package_name in _crawled_packages:
        logger.debug(f"📋 [dim]Package '{package_name}' already crawled[/dim]")
        return

    # Get base class names for pattern matching
    base_class_names = set()
    if base_classes:
        base_class_names.update(cls.__name__ for cls in base_classes)
    
    logger.debug(f"🔍 [bold cyan]FAST CRAWLING[/bold cyan] {package_name}")
    
    # Build criteria description
    criteria = []
    if base_class_names:
        criteria.append(f"bases: {sorted(base_class_names)}")
    if check_has_attr:
        criteria.append(f"attrs: {check_has_attr}")
    if method_patterns:
        criteria.append(f"methods: {method_patterns}")
    
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

    package_path = Path(package.__file__).parent
    candidate_modules = []
    scanned_files = 0
    
    # Scan all .py files in the package
    for py_file in package_path.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        scanned_files += 1
        relative_path = py_file.relative_to(package_path.parent)
        module_name = str(relative_path.with_suffix("")).replace(os.sep, ".")
        
        # Apply skip patterns
        if skip_patterns:
            should_skip = any(pattern in module_name for pattern in skip_patterns)
            if should_skip:
                logger.debug(f"   ⏭️  [dim]Skipping {py_file.name}[/dim]")
                continue
        
        # Scan file for classes
        scan_result = scan_file_for_classes(py_file, base_class_names, method_patterns)
        
        if scan_result['inheritance'] or scan_result['special_methods']:
            candidate_modules.append((module_name, scan_result))
    
    # Index classes without importing modules
    indexed_count = 0
    for module_name, scan_result in candidate_modules:
        indexed_in_module = []
        
        # Index inheritance-based classes
        if scan_result['inheritance']:
            for class_name in scan_result['inheritance']:
                _class_index[class_name] = module_name
                indexed_count += 1
                indexed_in_module.append(f"{class_name}→inheritance")
        
        # Index special method classes
        if scan_result['special_methods']:
            for class_name in scan_result['special_methods']:
                _class_index[class_name] = module_name
                indexed_count += 1
                method_indicator = f"@{method_patterns[0] if method_patterns else 'method'}"
                indexed_in_module.append(f"{class_name}{method_indicator}")
        
        if indexed_in_module:
            module_short = module_name.replace(f"{package_name}.", "")
            logger.debug(f"   📦 [green]{module_short}[/green]: {', '.join(indexed_in_module)}")

    _crawled_packages.add(package_name)
    logger.debug(f"   ✅ [bold green]COMPLETE[/bold green] {indexed_count} classes indexed from {len(candidate_modules)}/{scanned_files} modules → {len(_class_index)} total")


def crawl_package(
    package_name: str,
    base_classes: Optional[List[Type[Any]]] = None,
    check_has_attr: Optional[List[str]] = None,
    skip_patterns: Optional[List[str]] = None,
):
    """
    LEGACY: Eager loading package crawler. Use crawl_package_fast() for lazy loading.
    
    Crawls a given package to find and register classes based on specified criteria.
    This function imports all modules immediately and stores class objects in the registry.

    The function uses importlib and pkgutil to recursively walk through all modules in the specified package.
    For each module, it inspects all classes and registers those that match the criteria:
    - It is a subclass of any class in `base_classes`.
    - It has any attribute listed in `check_has_attr`.

    Args:
        package_name: Name of the package to crawl
        base_classes: List of base classes to match against
        check_has_attr: List of attribute names to check for
        skip_patterns: List of module name patterns to skip during crawling

    If a class with the same name is already registered, a warning is logged.
    The function also keeps track of which packages have already been crawled to avoid redundant work.
    """
    if package_name in _crawled_packages:
        logger.debug(f"📋 [dim]Package '{package_name}' already crawled[/dim]")
        return

    base_classes = base_classes or []

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
        # Skip modules matching skip patterns
        if skip_patterns:
            should_skip = False
            for pattern in skip_patterns:
                if pattern in module_info.name:
                    should_skip = True
                    break
            if should_skip:
                logger.debug(f"   ⏭️  [dim]Skipping {module_info.name.split('.')[-1]}[/dim]")
                continue
        
        visit_module(module_info)

    _crawled_packages.add(package_name)
    
    logger.debug(f"   ✅ [bold green]COMPLETE[/bold green] {registered_count}/{module_count} classes/modules → {len(_class_registry)} total")


def get_class(name: str) -> Type[Any] | None:
    """
    Retrieves a class by its name, lazily importing if needed.
    """
    # Check if already loaded
    if name in _class_registry:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 {name} ✅ [cached]")
        return _class_registry[name]
    
    # Check if we know where to find it
    if name not in _class_index:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 {name} ❌ [not indexed]")
        return None
    
    # Lazy import the module and find the class
    module_name = _class_index[name]
    try:
        module = importlib.import_module(module_name)
        
        # Find the class in the module
        for class_name, obj in inspect.getmembers(module, inspect.isclass):
            if class_name == name:
                _class_registry[name] = obj
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"🔍 {name} ✅ [lazy loaded from {module_name}]")
                return obj
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 {name} ❌ [not found in {module_name}]")
        return None
        
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 {name} ❌ [import failed: {str(e)[:30]}]")
        return None


def get_all_classes() -> Dict[str, Type[Any]]:
    """
    Returns all classes, lazy-loading any that aren't already loaded.
    """
    # Lazy-load any classes that are indexed but not yet loaded
    for class_name in _class_index:
        if class_name not in _class_registry:
            get_class(class_name)  # This will lazy-load it
    
    logger.debug(f"📚 [cyan]Registry dump[/cyan]: {len(_class_registry)} classes (from {len(_class_index)} indexed)")
    return _class_registry.copy()

_special_classes_cache = {}
_cache_valid = False

def find_special_classes(method_names: Optional[List[str]] = None):
    """
    Find classes in loaded modules that have specific methods.
    
    Args:
        method_names: List of method names to look for (defaults to ['__holo__'] for backwards compatibility)
    """
    global _special_classes_cache, _cache_valid
    
    method_names = method_names or ['__holo__']
    cache_key = tuple(sorted(method_names))

    if _cache_valid and cache_key in _special_classes_cache:
        cached_classes = _special_classes_cache[cache_key]
        logger.debug(f"🎯 [green]Special class cache hit[/green]: {len(cached_classes)} classes")
        return cached_classes

    logger.debug(f"🔍 [bold cyan]SCANNING SPECIAL CLASSES[/bold cyan]: {method_names}")
    
    special_classes = {}
    modules = dict(sys.modules)
    found_classes = 0
    
    for module_name, module in modules.items():
        if module is None:
            continue

        try:
            module_special_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if class has any of the specified methods
                for method_name in method_names:
                    # Check if the method is defined in the class itself, not inherited
                    if method_name in obj.__dict__ and callable(getattr(obj, method_name)):
                        special_classes[name] = obj
                        found_classes += 1
                        module_special_classes.append(f"{name}@{method_name}")
                        break  # Found one method, no need to check others
            
            # Only log modules with special classes
            if module_special_classes:
                module_short = module_name.split('.')[-1] if '.' in module_name else module_name
                logger.debug(f"   🎭 [green]{module_short}[/green]: {', '.join(module_special_classes)}")
                
        except Exception:
            continue

    _special_classes_cache[cache_key] = special_classes
    _cache_valid = True
    
    logger.debug(f"   ✅ [bold green]COMPLETE[/bold green] {found_classes} special classes found")
    
    return special_classes

def resolve_special_class(class_name: str, method_names: Optional[List[str]] = None):
    """
    Resolve a class by name that has specific methods, using lazy loading first, then fallback to module scanning.
    
    Args:
        class_name: Name of the class to find
        method_names: List of method names the class should have (defaults to ['__holo__'])
    """
    method_names = method_names or ['__holo__']
    
    # First try lazy loading from index
    result = get_class(class_name)
    if result:
        # Check if it has any of the required methods
        for method_name in method_names:
            if hasattr(result, method_name):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"🎯 {class_name} ✅ [lazy]")
                return result
    
    # Fallback to scanning loaded modules
    special_classes = find_special_classes(method_names)
    result = special_classes.get(class_name)
    
    if logger.isEnabledFor(logging.DEBUG):
        status = "✅ [scanned]" if result else "❌"
        logger.debug(f"🎯 {class_name} {status}")
    
    return result


def resolve_holo_class(class_name: str):
    """
    DEPRECATED: Use resolve_special_class() instead.
    Resolve a holo class by name (backwards compatibility).
    """
    return resolve_special_class(class_name, ['__holo__'])


def find_holo_classes():
    """
    DEPRECATED: Use find_special_classes() instead. 
    Find holo classes (backwards compatibility).
    """
    return find_special_classes(['__holo__'])


def get_class_index() -> Dict[str, str]:
    """
    Returns the class name -> module name index for debugging.
    """
    logger.debug(f"📋 [cyan]Index dump[/cyan]: {len(_class_index)} classes indexed")
    return _class_index.copy()

def invalidate_special_cache():
    """Invalidate the special classes cache."""
    global _cache_valid
    logger.debug(f"🗑️  [yellow]Special class cache invalidated[/yellow]")
    _cache_valid = False


def invalidate_holo_cache():
    """DEPRECATED: Use invalidate_special_cache() instead."""
    return invalidate_special_cache()


def crawl_on_demand(
    module_patterns: List[str], 
    base_classes: Optional[List[Type[Any]]] = None, 
    check_has_attr: Optional[List[str]] = None,
    method_patterns: Optional[List[str]] = None,
    skip_patterns: Optional[List[str]] = None,
):
    """
    Crawl specific module patterns that were previously skipped during initial package crawling.
    Uses lazy indexing - classes are catalogued but not imported until requested.
    
    Args:
        module_patterns: List of module patterns to crawl (e.g., ['errloom.training', 'errloom.gui'])
        base_classes: List of base classes to match against  
        check_has_attr: List of attribute names to check for
        method_patterns: List of method names to search for
        skip_patterns: List of module patterns to skip
    """
    logger.debug(f"🔄 [bold cyan]ON-DEMAND CRAWL[/bold cyan]: {module_patterns}")
    
    for pattern in module_patterns:
        try:
            # Use lazy crawling - index classes without importing
            crawl_package_fast(
                pattern, 
                base_classes=base_classes, 
                check_has_attr=check_has_attr,
                method_patterns=method_patterns,
                skip_patterns=skip_patterns
            )
        except Exception as e:
            logger.debug(f"   ❌ [red]Failed to crawl {pattern}: {e}[/red]")
    
    # Invalidate special class cache since we may have found new classes
    invalidate_special_cache()
