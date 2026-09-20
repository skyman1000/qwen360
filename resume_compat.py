"""Accept only the recorded, training-neutral LoRA-scale interface migration.

Never ignore arbitrary source edits or training/inference parameter changes.
Existing checkpoint files and generation metadata are not rewritten here.
"""
import json
from pathlib import Path


def compatible_config(previous, current, *, inference=False):
    previous = dict(previous)
    if inference:
        previous.setdefault("lora_scale", 1.0 if previous.get("lora_sha256") else None)
    if previous == current:
        return True
    old_sources = previous.get("source_hashes")
    new_sources = current.get("source_hashes")
    migration_path = Path(__file__).with_name("lora_scale_source_migration.json")
    if not migration_path.is_file():
        return False
    migration = json.loads(migration_path.read_text())
    if old_sources != migration["before"] or new_sources != migration["after"]:
        return False
    previous["source_hashes"] = new_sources
    return previous == current
