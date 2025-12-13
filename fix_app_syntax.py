
import os

def fix_app():
    path = "app.py"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    found = False
    
    for i, line in enumerate(lines):
        # Check for the zombie block start
        if "pred_rf2 =" in line and "self.rf2.predict" in line:
            print(f"Found zombie line at {i+1}: {line.strip()}")
            skip = True
            found = True
        
        # Heuristic to stop skipping: if we hit a def/class or specific known next line
        if skip:
            if "def predict_proba" in line: # Also zombie
                continue
            if line.strip() == "": # Keep empty lines?
                pass
            if "# =====================================================================" in line: # Next section
                skip = False
            elif "def " in line and "predict_proba" not in line: # New method
                skip = False
        
        if not skip:
            new_lines.append(line)
        else:
            print(f"Removing: {line.strip()}")

    if found:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("Fixed app.py")
    else:
        print("No zombie lines found in app.py via python script.")
        # Print lines around 188 to debug
        start = max(0, 180)
        end = min(len(lines), 200)
        print(f"Lines {start}-{end}:")
        for j in range(start, end):
            print(f"{j+1}: {lines[j].rstrip()}")

if __name__ == "__main__":
    fix_app()
