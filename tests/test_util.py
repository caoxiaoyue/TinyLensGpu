"""
Unit tests for utility functions.
"""
import pytest
import numpy as np
import os
from TinyLensGpu.util import auto_mkdir_path


@pytest.mark.unit
class TestAutoMkdir:
    """Test automatic directory creation utility."""
    
    def test_auto_mkdir_creates_directory(self, tmp_path):
        """Test that auto_mkdir_path creates a new directory."""
        test_dir = tmp_path / "test_directory"
        
        # Directory should not exist initially
        assert not test_dir.exists()
        
        # Create directory
        auto_mkdir_path(str(test_dir))
        
        # Directory should now exist
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_auto_mkdir_nested_directories(self, tmp_path):
        """Test that auto_mkdir_path creates nested directories."""
        test_dir = tmp_path / "level1" / "level2" / "level3"
        
        # Directory should not exist initially
        assert not test_dir.exists()
        
        # Create nested directories
        auto_mkdir_path(str(test_dir))
        
        # All directories should now exist
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_auto_mkdir_existing_directory(self, tmp_path):
        """Test that auto_mkdir_path handles existing directory gracefully."""
        test_dir = tmp_path / "existing_dir"
        
        # Create directory
        test_dir.mkdir()
        assert test_dir.exists()
        
        # Call auto_mkdir_path on existing directory (should not raise error)
        auto_mkdir_path(str(test_dir))
        
        # Directory should still exist
        assert test_dir.exists()
    
    def test_auto_mkdir_relative_path(self, tmp_path):
        """Test auto_mkdir_path with relative path."""
        # Change to tmp_path temporarily
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            test_dir = "relative_test_dir"
            
            # Directory should not exist initially
            assert not os.path.exists(test_dir)
            
            # Create directory with relative path
            auto_mkdir_path(test_dir)
            
            # Directory should now exist
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)
        finally:
            # Restore original directory
            os.chdir(original_dir)

