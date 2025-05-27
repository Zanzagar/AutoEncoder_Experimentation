import json
import re

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get basic info
    cell_count = len(notebook['cells'])
    code_cell_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
    markdown_cell_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    
    print(f"Total cells: {cell_count}")
    print(f"Code cells: {code_cell_count}")
    print(f"Markdown cells: {markdown_cell_count}")
    
    # Extract classes
    classes = set()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            for match in re.finditer(r'^class\s+(\w+)', source, re.MULTILINE):
                classes.add(match.group(1))
    
    print("\nClasses:")
    for cls in sorted(classes):
        print(f" - {cls}")
    
    # Extract imports
    imports = set()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            for line in source.split('\n'):
                if line.strip().startswith('import ') or ' import ' in line:
                    imports.add(line.strip())
    
    print("\nImports:")
    for imp in sorted(imports):
        print(f" - {imp}")

    # Extract functions
    functions = set()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            for match in re.finditer(r'^def\s+(\w+)', source, re.MULTILINE):
                functions.add(match.group(1))
    
    print("\nFunctions:")
    for func in sorted(functions):
        print(f" - {func}")

if __name__ == "__main__":
    analyze_notebook("AutoEncoderWrapper.ipynb") 