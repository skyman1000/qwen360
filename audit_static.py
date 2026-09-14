"""Read-only source audit. No torch import, model execution, or data generation."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="Optional JSON audit report")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    upstream = json.loads((root / "upstream_sources.json").read_text())
    checks = []
    sources = list(root.glob("*.py")) + list((root / "tests").glob("*.py")) + list((root / "vendor").rglob("*.py"))
    for path in sorted(sources):
        tree = ast.parse(path.read_text(), filename=str(path))
        # Catch statements accidentally indented after an unconditional exit.
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if isinstance(body, list):
                    for i, statement in enumerate(body[:-1]):
                        if isinstance(statement, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
                            raise ValueError(f"Unreachable statements in {path}:{statement.lineno}")
        checks.append(dict(check="python_syntax_and_unreachable_tail", path=str(path.relative_to(root)), status="pass"))
    for relative, item in upstream["files"].items():
        local = root.parent / "DiT360" / relative
        if digest(local) != item["sha256"]:
            raise ValueError(f"Local DiT360 source drifted from pinned upstream: {relative}")
        checks.append(dict(check="local_source_equals_downloaded_official_sha256", path=relative, status="pass"))
    for filename in ("cube_map.py", "yaw_rotate.py", "LICENSE"):
        key = filename if filename == "LICENSE" else "src/" + filename
        if digest(root / "vendor/dit360" / filename) != upstream["files"][key]["sha256"]:
            raise ValueError(f"Vendored upstream file modified: {filename}")
        checks.append(dict(check="vendor_byte_identity", path=filename, status="pass"))
    tree = ast.parse((root / "profiles.py").read_text())
    recipes = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("REPO_PANORAMA", "REPO_MIX"):
            if isinstance(node.value, ast.Call):
                recipe = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            else:
                recipe = {}
                for key, value in zip(node.value.keys, node.value.values):
                    if key is None:
                        recipe.update(recipes[value.id])
                    else:
                        recipe[ast.literal_eval(key)] = ast.literal_eval(value)
            recipes[node.targets[0].id] = recipe
    mapping = dict(height="resolution", epochs="max_epochs", batch_size="train_batch_size",
                   accumulation_steps="accumulate_grad_batches", learning_rate="learning_rate",
                   rank="rank", lora_alpha="lora_alpha", lora_dropout="lora_drop_out",
                   padding_columns="padding_n", lambda_cube="lambda_cube", lambda_yaw="lambda_yaw", seed="seed")
    for name, script in (("REPO_PANORAMA", "train.sh"), ("REPO_MIX", "train_mix_staged_lora_dynamic.sh")):
        flags = dict(re.findall(r"--(\w+)=([^\s\\]+)", (root.parent / "DiT360" / script).read_text()))
        for ours, theirs in mapping.items():
            if recipes[name][ours] != float(flags[theirs]):
                raise ValueError(f"Profile differs from official launch script: {name}.{ours}")
        checks.append(dict(check="numerical_recipe_equals_official_shell", profile=name, status="pass"))
    official_workers = int(re.search(r"--dataloader_num_workers=(\d+)",
                                     (root.parent / "DiT360/train.sh").read_text()).group(1))
    launch_workers = int(re.search(r"--workers\s+(\d+)",
                                   (root / "scripts/train_pano.sh").read_text()).group(1))
    if launch_workers != official_workers:
        raise ValueError("Panorama launcher worker count differs from official train.sh")
    checks.append(dict(check="panorama_launch_workers_equal_official_shell", workers=official_workers, status="pass"))
    scripts = sorted(list((root / "scripts").glob("*.sh")) + list((root / "scripts").glob("*.sbatch")))
    for script in scripts:
        if re.search(r"(?:source|bash)\s+[^\n]*scripts/", script.read_text()):
            raise ValueError(f"Batch job must not call another project shell script: {script}")
        # -n parses the shell program without executing its commands.
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)
        checks.append(dict(check="bash_syntax_only", path=str(script.relative_to(root)), status="pass"))
    reviewed = sources + scripts + [root / name for name in
                                    ("README.md", "RUN_FIRST.md", "AUDIT.md", "requirements.txt", "upstream_sources.json")]
    report = dict(upstream_commit=upstream["commit"], checks=checks,
                  reviewed_file_sha256={str(path.relative_to(root)): digest(path) for path in sorted(reviewed)},
                  result="static_checks_passed", model_execution=False, unit_tests_executed=False,
                  limitations=["No dependency imports or runtime/gradient/checkpoint validation.",
                               "No author perspective corpus or complete paper training configuration was available.",
                               "No claim of numerical FLUX/Qwen parity or reproduced paper metrics."])
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(result=report["result"], checks=len(checks), upstream_commit=upstream["commit"])))


if __name__ == "__main__":
    main()
