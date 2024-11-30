import os
import re
import json
from pathlib import Path
from nbconvert import MarkdownExporter
import nbformat

# Directories to ignore
IGNORE_DIRS = {
    '_templates',
    'changes',
    'versions',
    'example_data',
    '__pycache__',
    'node_modules'
}

# Order of main directories for logical concatenation
DIRECTORY_ORDER = [
    'introduction.mdx',  # Start with the introduction
    'concepts',         # Then basic concepts
    'how_to',          # How-to guides
    'tutorials',        # Tutorials
    'integrations',    # Integrations
    'additional_resources',  # Additional resources
    'contributing',    # Contributing guides
    'troubleshooting'  # Troubleshooting at the end
]

def convert_notebook_to_md(notebook_path):
    """Convert Jupyter notebook to markdown."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
        
        # Create a notebook object
        nb = nbformat.reads(json.dumps(notebook_content), as_version=4)
        
        # Configure and create the markdown exporter
        md_exporter = MarkdownExporter()
        
        # Convert notebook to markdown
        markdown, _ = md_exporter.from_notebook_node(nb)
        return markdown
    except Exception as e:
        print(f"Error converting notebook {notebook_path}: {str(e)}")
        return f"Error converting notebook {notebook_path}\n\n"

def convert_mdx_to_md(content):
    """Convert MDX content to Markdown by removing MDX-specific syntax."""
    # Remove import statements
    content = re.sub(r'^import.*$', '', content, flags=re.MULTILINE)
    # Remove export statements
    content = re.sub(r'^export.*$', '', content, flags=re.MULTILINE)
    # Remove JSX components (basic implementation)
    content = re.sub(r'<[^>]+>', '', content)
    return content.strip()

def process_file(file_path):
    """Process a single file and return its markdown content."""
    try:
        # Handle Jupyter notebooks
        if file_path.suffix == '.ipynb':
            content = convert_notebook_to_md(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix == '.mdx':
                content = convert_mdx_to_md(content)
        
        # Add file header
        relative_path = file_path.relative_to(docs_root)
        header = f"\n\n# {relative_path}\n\n"
        return header + content
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return f"Error processing file {file_path}\n\n"

def should_process_directory(dir_path):
    """Check if directory should be processed."""
    dir_name = dir_path.name
    return (
        not dir_name.startswith('.') and
        not dir_name.startswith('_') and
        dir_name not in IGNORE_DIRS
    )

def get_files_in_order(root_dir):
    """Get all markdown files in a specified order."""
    ordered_files = []
    
    # First add the introduction file if it exists
    intro_file = root_dir / 'introduction.mdx'
    if intro_file.exists():
        ordered_files.append(intro_file)
    
    # Then process each directory in the specified order
    for dir_name in DIRECTORY_ORDER:
        dir_path = root_dir / dir_name
        if isinstance(dir_name, str) and dir_path.is_dir():
            for file_path in sorted(dir_path.rglob('*')):
                if file_path.suffix in ['.md', '.mdx', '.ipynb'] and should_process_directory(file_path.parent):
                    ordered_files.append(file_path)
    
    # Add any remaining markdown files in the root directory
    for file_path in root_dir.glob('*'):
        if (file_path.suffix in ['.md', '.mdx', '.ipynb'] and 
            file_path not in ordered_files and 
            file_path.name != 'introduction.mdx'):
            ordered_files.append(file_path)
    
    return ordered_files

if __name__ == '__main__':
    # Set the root directory
    docs_root = Path('/home/nauman/repos/langchain/docs/docs')
    output_file = docs_root / 'llms-full.txt'
    
    # Process all files
    with open(output_file, 'w', encoding='utf-8') as out_f:
        files = get_files_in_order(docs_root)
        for file_path in files:
            print(f"Processing: {file_path}")
            content = process_file(file_path)
            out_f.write(content + '\n\n')
    
    print(f"\nDocumentation has been generated at: {output_file}")
