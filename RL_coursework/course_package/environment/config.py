"""Environment-config helpers for transferable policies."""

import json
import re
from copy import deepcopy
from pathlib import Path

from course_package.core.params import HOUSE_SPECS

EXCLUDED_FILES = ("server", "template")

REQUIRED_SPECS = frozenset(HOUSE_SPECS)


def select_prosumers(env_config: dict, prosumer_ids: list[str]) -> dict:
    """Return a config restricted to prosumers, preserving fitted bounds."""
    available = list(env_config["prosumers"])
    unknown = sorted(set(prosumer_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown prosumers: {unknown}")
    if not prosumer_ids:
        raise ValueError("prosumer_ids cannot be empty")

    config = deepcopy(env_config)
    config["prosumers"] = {
        prosumer_id: deepcopy(env_config["prosumers"][prosumer_id])
        for prosumer_id in prosumer_ids
    }
    return config


def build_prosumer_config(setup_dir: str) -> tuple[dict, dict]:
    """
    Read JSON files from external directory, build a prosumer config dictionary
    and identify the reference observation and action space schemas.
    """
    if not setup_dir:
        raise ValueError("Prosumer config directory is not specified")
    if not Path(setup_dir).exists():
        raise FileNotFoundError(
            f"Prosumer config directory does not exist: {setup_dir}"
        )

    prosumers = {}
    prosumer_files = [
        p
        for p in Path(setup_dir).glob("*.json")
        if not any(term in p.name for term in EXCLUDED_FILES)
    ]
    if not prosumer_files:
        raise FileNotFoundError(f"No prosumer config files found in {setup_dir}")
    prosumer_files = sorted(prosumer_files, key=lambda f: extract_number(f.name))

    reference_schemas = None
    obs_schemas = {}
    action_schemas = {}
    for file in prosumer_files:
        if any(term in file.name for term in ("server", "template")):
            continue

        with open(file, "r") as f:
            setup = json.load(f)

        id = setup["global-properties"]["label"]
        prosumers[id] = {}
        prosumers[id]["data_file"] = id + ".csv"
        prosumers[id]["specs"] = {}
        obs_schemas[id] = {}
        action_schemas[id] = {}

        for device in setup["devices"]:
            if device["label"] == "RLInfo":
                continue

            props = device["properties"]
            prosumers[id]["specs"].update(device["properties"].get("specs", {}))

            signals = props.get("sub-labels", {})
            if signals:
                actions = [a for a in signals if a.startswith("action")]
                measurement = [m for m in signals if m not in actions]
                if measurement:
                    obs_schemas[id].update({props["kind"]: measurement})
                if actions:
                    action_schemas[id].update({props["kind"]: actions})

        missing_specs = REQUIRED_SPECS - set(prosumers[id]["specs"])
        if missing_specs:
            raise ValueError(
                f"Prosumer config file {file} is missing required agent-context "
                f"spec(s): {', '.join(missing_specs)}."
            )

        if reference_schemas is None:
            reference_schemas = {
                "obs": deepcopy(obs_schemas[id]),
                "action": deepcopy(action_schemas[id]),
            }
        else:
            mismatched = ()
            if obs_schemas[id] != reference_schemas["obs"]:
                mismatched += ("observation",)
            if action_schemas[id] != reference_schemas["action"]:
                mismatched += ("action",)
            if mismatched:
                raise ValueError(
                    f"Prosumer config file {file} has a mismatch in {' and '.join(mismatched)} schemas."
                )
    return prosumers, reference_schemas


def load_config(env_config_path: str) -> dict:
    """Load environment and prosumer configs from JSON files."""
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    prosumer_config, schemas = build_prosumer_config(env_config["data"]["prosumers"])
    env_config["prosumers"] = prosumer_config
    env_config["measurements_schema"] = schemas["obs"]
    env_config["action_space_schema"] = schemas["action"]
    return env_config


def extract_number(filename):
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(filename).name)
    if not match:
        raise ValueError(f"No trailing number found in filename: {filename}")
    return int(match.group(1))


def extract_agent_number(agent_id):
    match = re.search(r"(\d+)$", agent_id)
    if not match:
        raise ValueError(f"No trailing number found in agent ID: {agent_id}")
    return int(match.group(1))
