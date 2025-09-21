#!/usr/bin/env python3
"""
Script to modify detectron2 imports to use local detectron2 folder instead of system installation.
This bypasses compilation issues while maintaining functionality.
"""

import os
import re
import sys
from pathlib import Path

def modify_detectron2_imports(file_path):
    """Modify detectron2 imports in a Python file to use local path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Add sys.path modification at the top of the file (after existing imports)
        detectron2_path_insert = '''
# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))
'''
        
        # Check if the path modification is already present
        if "Add local detectron2 to path" not in content:
            # Find the first import statement and insert our path modification before it
            import_pattern = r'^(import |from )'
            lines = content.split('\n')
            insert_pos = 0
            
            # Skip initial comments and docstrings
            in_docstring = False
            docstring_delim = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Handle docstrings
                if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                    docstring_delim = stripped[:3]
                    if stripped.count(docstring_delim) >= 2 and len(stripped) > 3:
                        # Single line docstring
                        continue
                    else:
                        in_docstring = True
                        continue
                elif in_docstring and docstring_delim in stripped:
                    in_docstring = False
                    continue
                elif in_docstring:
                    continue
                
                # Skip comments and empty lines
                if stripped.startswith('#') or stripped == '':
                    continue
                
                # Found first import
                if re.match(import_pattern, stripped):
                    insert_pos = i
                    break
            
            # Insert the path modification
            lines.insert(insert_pos, detectron2_path_insert.strip())
            content = '\n'.join(lines)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Modified: {file_path}")
            return True
        else:
            print(f"⚬ Already modified: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def find_python_files_with_detectron2(root_dir):
    """Find all Python files that import detectron2."""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip the detectron2 directory itself
        if 'detectron2' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'from detectron2' in content or 'import detectron2' in content:
                            python_files.append(file_path)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    return python_files

def main():
    """Main function to modify all detectron2 imports."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🔍 Finding Python files with detectron2 imports...")
    files_to_modify = find_python_files_with_detectron2(script_dir)
    
    if not files_to_modify:
        print("No Python files with detectron2 imports found.")
        return
    
    print(f"📝 Found {len(files_to_modify)} files to modify:")
    for file_path in files_to_modify:
        print(f"  - {os.path.relpath(file_path, script_dir)}")
    
    print("\n🔧 Modifying imports...")
    modified_count = 0
    
    for file_path in files_to_modify:
        if modify_detectron2_imports(file_path):
            modified_count += 1
    
    print(f"\n✅ Complete! Modified {modified_count} files to use local detectron2.")
    print("\n💡 Now you can run training without installing detectron2 system-wide:")
    print("   python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 1")

if __name__ == "__main__":
    main()