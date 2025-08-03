import tempfile
import textwrap
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


from errloom.lib.discovery import (
    scan_file_for_classes,
    crawl_package_fast,
    get_class,
    get_all_classes,
    get_class_index,
    find_special_classes,
    resolve_special_class,
    # Backwards compatibility functions
    find_holo_classes,
    resolve_holo_class,
    invalidate_special_cache,
    invalidate_holo_cache,
    # Internal state for testing
    _class_registry,
    _class_index,
    _crawled_packages,
    _text_scan_cache,
)
from tests.base import ErrloomTest


class MockTestClass:
    """Mock class for testing discovery."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __holo__(self, holophore, span):
        return "mock holo result"


class MockBaseClass:
    """Mock base class for inheritance testing."""
    pass


class MockDerivedClass(MockBaseClass):
    """Mock derived class for inheritance testing."""
    def __call__(self):
        return "callable"


class ScanFileForClassesTest(ErrloomTest):
    """Test text-based file scanning without imports."""

    def setUp(self):
        super().setUp()
        # Clear cache for each test
        _text_scan_cache.clear()

    def test_scan_file_empty(self):
        """Test scanning empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            result = scan_file_for_classes(Path(f.name), {'BaseClass'})
            self.assertEqual(result, {'inheritance': set(), 'special_methods': set()})

    def test_scan_file_inheritance_detection(self):
        """Test detection of class inheritance."""
        code = textwrap.dedent("""
            class RegularClass:
                pass
            
            class DerivedClass(BaseClass):
                def method(self):
                    pass
            
            class MultipleInheritance(BaseClass, OtherBase):
                pass
            
            class NotMatching(SomeOtherBase):
                pass
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = scan_file_for_classes(Path(f.name), {'BaseClass'})
            self.assertEqual(result['inheritance'], {'DerivedClass', 'MultipleInheritance'})
            self.assertEqual(result['special_methods'], set())

    def test_scan_file_method_detection(self):
        """Test detection of special methods."""
        code = textwrap.dedent("""
            class HoloClass:
                def __holo__(self, holophore, span):
                    return "result"
            
            class AsyncHoloClass:
                async def __holo__(self, holophore, span):
                    return "async result"
            
            class CallableClass:
                def __call__(self):
                    pass
            
            class RegularClass:
                def regular_method(self):
                    pass
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = scan_file_for_classes(Path(f.name), set(), ['__holo__', '__call__'])
            self.assertEqual(result['inheritance'], set())
            self.assertEqual(result['special_methods'], {'HoloClass', 'AsyncHoloClass', 'CallableClass'})

    def test_scan_file_combined_criteria(self):
        """Test scanning with both inheritance and method criteria."""
        code = textwrap.dedent("""
            class BaseClass:
                pass
            
            class DerivedHolo(BaseClass):
                def __holo__(self, holophore, span):
                    pass
            
            class JustDerived(BaseClass):
                pass
            
            class JustHolo:
                def __holo__(self, holophore, span):
                    pass
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = scan_file_for_classes(Path(f.name), {'BaseClass'}, ['__holo__'])
            self.assertEqual(result['inheritance'], {'DerivedHolo', 'JustDerived'})
            self.assertEqual(result['special_methods'], {'DerivedHolo', 'JustHolo'})

    def test_scan_file_caching(self):
        """Test that file scanning results are cached."""
        code = "class TestClass(BaseClass): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

            # First scan
            result1 = scan_file_for_classes(path, {'BaseClass'})
            self.assertEqual(len(_text_scan_cache), 1)

            # Second scan should use cache
            result2 = scan_file_for_classes(path, {'BaseClass'})
            self.assertEqual(result1, result2)
            self.assertEqual(len(_text_scan_cache), 1)  # Cache size unchanged

    def test_scan_file_unicode_error(self):
        """Test handling of files that can't be read as UTF-8."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            f.write(b'\xff\xfe')  # Invalid UTF-8
            f.flush()

            result = scan_file_for_classes(Path(f.name), {'BaseClass'})
            self.assertEqual(result, {'inheritance': set(), 'special_methods': set()})


class LazyLoadingTest(ErrloomTest):
    """Test lazy loading functionality."""

    def setUp(self):
        super().setUp()
        # Clear registries for each test
        _class_registry.clear()
        _class_index.clear()
        _crawled_packages.clear()

    def test_get_class_not_indexed(self):
        """Test getting a class that isn't indexed."""
        result = get_class('NonExistentClass')
        self.assertIsNone(result)

    def test_get_class_already_cached(self):
        """Test getting a class that's already in the registry."""
        _class_registry['TestClass'] = MockTestClass

        result = get_class('TestClass')
        self.assertEqual(result, MockTestClass)

    @patch('errloom.lib.discovery.importlib.import_module')
    @patch('errloom.lib.discovery.inspect.getmembers')
    def test_get_class_lazy_import(self, mock_getmembers, mock_import):
        """Test lazy importing of a class."""
        # Setup index
        _class_index['LazyClass'] = 'test.module'

        # Mock the module and class
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        mock_getmembers.return_value = [('LazyClass', MockTestClass), ('OtherClass', str)]

        result = get_class('LazyClass')

        mock_import.assert_called_once_with('test.module')
        mock_getmembers.assert_called_once_with(mock_module, inspect.isclass)
        self.assertEqual(result, MockTestClass)
        self.assertEqual(_class_registry['LazyClass'], MockTestClass)

    @patch('errloom.lib.discovery.importlib.import_module')
    def test_get_class_import_failure(self, mock_import):
        """Test handling of import failures during lazy loading."""
        _class_index['FailClass'] = 'nonexistent.module'
        mock_import.side_effect = ImportError("Module not found")

        result = get_class('FailClass')
        self.assertIsNone(result)

    def test_get_all_classes_lazy_load(self):
        """Test that get_all_classes lazy-loads indexed classes."""
        # Setup some indexed classes
        _class_index['Class1'] = 'module1'
        _class_index['Class2'] = 'module2'
        _class_registry['Class1'] = MockTestClass  # Already loaded

        # Mock get_class to simulate lazy loading
        with patch('errloom.lib.discovery.get_class') as mock_get_class:
            mock_get_class.side_effect = lambda name: MockDerivedClass if name == 'Class2' else _class_registry.get(name)

            result = get_all_classes()

            # Should attempt to lazy-load Class2
            mock_get_class.assert_called_with('Class2')

    def test_get_class_index(self):
        """Test getting the class index."""
        _class_index['TestClass'] = 'test.module'
        _class_index['OtherClass'] = 'other.module'

        result = get_class_index()
        self.assertEqual(result, {'TestClass': 'test.module', 'OtherClass': 'other.module'})
        # Should return a copy, not the original
        self.assertIsNot(result, _class_index)


class SpecialClassDiscoveryTest(ErrloomTest):
    """Test special method-based class discovery."""

    def setUp(self):
        super().setUp()
        # Add mock classes to sys.modules for testing
        mock_module = MagicMock()
        mock_module.__name__ = 'test_module'
        sys.modules['test_module'] = mock_module

        # Mock inspect.getmembers to return our test classes
        self.original_getmembers = inspect.getmembers

    def tearDown(self):
        super().tearDown()
        # Clean up mock module
        if 'test_module' in sys.modules:
            del sys.modules['test_module']

    @patch('errloom.lib.discovery.inspect.getmembers')
    def test_find_special_classes_with_methods(self, mock_getmembers):
        """Test finding classes with specific methods."""
        # Mock classes with different methods
        class_with_holo = type('HoloClass', (), {'__holo__': lambda self, h, s: None})
        class_with_call = type('CallableClass', (), {'__call__': lambda self: None})
        regular_class = type('RegularClass', (), {})

        mock_getmembers.return_value = [
            ('HoloClass', class_with_holo),
            ('CallableClass', class_with_call),
            ('RegularClass', regular_class),
        ]

        result = find_special_classes(['__holo__', '__call__'])

        # Should find classes with either method
        self.assertIn('HoloClass', result)
        self.assertIn('CallableClass', result)
        self.assertNotIn('RegularClass', result)

    def test_find_special_classes_default_holo(self):
        """Test that find_special_classes defaults to __holo__ method."""
        with patch('errloom.lib.discovery.sys.modules', {'test': MagicMock()}):
            with patch('errloom.lib.discovery.inspect.getmembers') as mock_getmembers:
                holo_class = type('HoloClass', (), {'__holo__': lambda self, h, s: None})
                mock_getmembers.return_value = [('HoloClass', holo_class)]

                result = find_special_classes()  # No method_names parameter
                self.assertIn('HoloClass', result)

    @patch('errloom.lib.discovery.get_class')
    def test_resolve_special_class_lazy_first(self, mock_get_class):
        """Test that resolve_special_class tries lazy loading first."""
        mock_class = type('TestClass', (), {'__holo__': lambda self, h, s: None})
        mock_get_class.return_value = mock_class

        result = resolve_special_class('TestClass', ['__holo__'])

        mock_get_class.assert_called_once_with('TestClass')
        self.assertEqual(result, mock_class)

    def test_resolve_special_class_fallback_scan(self):
        """Test resolve_special_class falls back to module scanning."""
        with patch('errloom.lib.discovery.get_class', return_value=None):
            with patch('errloom.lib.discovery.find_special_classes') as mock_find:
                mock_class = MockTestClass
                mock_find.return_value = {'TestClass': mock_class}

                result = resolve_special_class('TestClass', ['__holo__'])

                mock_find.assert_called_once_with(['__holo__'])
                self.assertEqual(result, mock_class)


class BackwardsCompatibilityTest(ErrloomTest):
    """Test backwards compatibility functions."""

    def test_find_holo_classes_compatibility(self):
        """Test that find_holo_classes calls find_special_classes."""
        with patch('errloom.lib.discovery.find_special_classes') as mock_find:
            mock_find.return_value = {'HoloClass': MockTestClass}

            result = find_holo_classes()

            mock_find.assert_called_once_with(['__holo__'])
            self.assertEqual(result, {'HoloClass': MockTestClass})

    def test_resolve_holo_class_compatibility(self):
        """Test that resolve_holo_class calls resolve_special_class."""
        with patch('errloom.lib.discovery.resolve_special_class') as mock_resolve:
            mock_resolve.return_value = MockTestClass

            result = resolve_holo_class('HoloClass')

            mock_resolve.assert_called_once_with('HoloClass', ['__holo__'])
            self.assertEqual(result, MockTestClass)

    def test_invalidate_holo_cache_compatibility(self):
        """Test that invalidate_holo_cache calls invalidate_special_cache."""
        with patch('errloom.lib.discovery.invalidate_special_cache') as mock_invalidate:
            invalidate_holo_cache()
            mock_invalidate.assert_called_once()


class CrawlPackageFastTest(ErrloomTest):
    """Test fast package crawling with lazy indexing."""

    def setUp(self):
        super().setUp()
        _class_index.clear()
        _crawled_packages.clear()

    def test_crawl_package_already_crawled(self):
        """Test that already crawled packages are skipped."""
        _crawled_packages.add('test.package')

        with patch('errloom.lib.discovery.importlib.import_module') as mock_import:
            crawl_package_fast('test.package')
            mock_import.assert_not_called()

    @patch('errloom.lib.discovery.importlib.import_module')
    @patch('errloom.lib.discovery.Path.rglob')
    @patch('errloom.lib.discovery.scan_file_for_classes')
    def test_crawl_package_indexes_classes(self, mock_scan, mock_rglob, mock_import):
        """Test that crawl_package_fast indexes discovered classes."""
        # Mock package import
        mock_package = MagicMock()
        mock_package.__file__ = '/path/to/package/__init__.py'
        mock_import.return_value = mock_package

        # Mock file discovery
        mock_file = MagicMock()
        mock_file.name = 'module.py'
        mock_file.relative_to.return_value = Path('package/module.py')
        mock_rglob.return_value = [mock_file]

        # Mock file scanning
        mock_scan.return_value = {
            'inheritance': {'InheritedClass'},
            'special_methods': {'HoloClass'}
        }

        crawl_package_fast('test.package',
                         base_classes=[MockBaseClass],
                         method_patterns=['__holo__'])

        # Should index both types of classes
        self.assertIn('InheritedClass', _class_index)
        self.assertIn('HoloClass', _class_index)
        self.assertEqual(_class_index['InheritedClass'], 'package.module')
        self.assertEqual(_class_index['HoloClass'], 'package.module')
        self.assertIn('test.package', _crawled_packages)

    @patch('errloom.lib.discovery.importlib.import_module')
    def test_crawl_package_import_error(self, mock_import):
        """Test handling of package import errors."""
        mock_import.side_effect = ImportError("Package not found")

        # Should not raise, just log and return
        crawl_package_fast('nonexistent.package')

        self.assertNotIn('nonexistent.package', _crawled_packages)

    @patch('errloom.lib.discovery.importlib.import_module')
    def test_crawl_package_namespace_package(self, mock_import):
        """Test handling of namespace packages (no __file__)."""
        mock_package = MagicMock()
        mock_package.__file__ = None
        mock_import.return_value = mock_package

        crawl_package_fast('namespace.package')

        self.assertNotIn('namespace.package', _crawled_packages)


class IntegrationTest(ErrloomTest):
    """Integration tests for the complete discovery workflow."""

    def setUp(self):
        super().setUp()
        _class_registry.clear()
        _class_index.clear()
        _crawled_packages.clear()
        invalidate_special_cache()

    def test_end_to_end_lazy_loading(self):
        """Test complete workflow from indexing to lazy loading."""
        # Simulate indexing a class
        _class_index['MockTestClass'] = 'tests.test_discovery'

        # Mock the import to return this module (which has MockTestClass)
        with patch('errloom.lib.discovery.importlib.import_module') as mock_import:
            mock_import.return_value = sys.modules[__name__]

            # Get the class - should trigger lazy loading
            result = get_class('MockTestClass')  # Use a real class from this module

            self.assertIsNotNone(result)
            self.assertEqual(result, MockTestClass)
            self.assertIn('MockTestClass', _class_registry)

    def test_class_discovery_and_resolution(self):
        """Test discovering and resolving special classes."""
        # Add a mock module with our test class
        mock_module = MagicMock()
        mock_module.__name__ = 'test_module'
        sys.modules['test_module'] = mock_module

        try:
            with patch('errloom.lib.discovery.inspect.getmembers') as mock_getmembers:
                # Make MockTestClass appear in the module
                mock_getmembers.return_value = [('MockTestClass', MockTestClass)]

                # Find classes with __holo__ method
                special_classes = find_special_classes(['__holo__'])

                # Should find our mock class
                self.assertIn('MockTestClass', special_classes)

                # Resolve the class by name
                resolved = resolve_special_class('MockTestClass', ['__holo__'])
                self.assertEqual(resolved, MockTestClass)

        finally:
            # Clean up
            if 'test_module' in sys.modules:
                del sys.modules['test_module']