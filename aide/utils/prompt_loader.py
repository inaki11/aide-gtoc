def load_prompts(filepath="prompts.txt"):
    prompts = {}
    current_key = None
    current_lines = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.startswith("[") and line.endswith("]\n"):
                if current_key:
                    prompts[current_key] = "".join(current_lines).strip()
                current_key = line.strip()[1:-1]
                current_lines = []
            else:
                current_lines.append(line)
        if current_key:
            prompts[current_key] = "".join(current_lines).strip()
    return prompts