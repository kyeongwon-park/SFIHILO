"""Paper-facing method names and internal acquisition aliases."""

METHODS = {
    "random": {
        "label": "Random",
        "query_policy": "random",
        "active_mode": None,
    },
    "mutual_information": {
        "label": "Mutual Information",
        "query_policy": "active",
        "active_mode": "mutual_information",
    },
    "boundary_lookahead": {
        "label": "Boundary Lookahead",
        "query_policy": "active",
        "active_mode": "boundary_lookahead",
    },
}


def _normalize_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def method_spec(spec: str | dict) -> dict:
    if isinstance(spec, dict):
        base_name = spec.get("base", spec.get("method", spec.get("name", spec.get("key", ""))))
        base_key = _normalize_key(base_name)
        if base_key not in METHODS:
            raise KeyError(f"Unknown method base '{base_name}'. Valid methods: {', '.join(METHODS)}")

        out = dict(METHODS[base_key])
        out["key"] = _normalize_key(spec.get("key", base_key))
        out["label"] = str(spec.get("label", out["label"]))
        if "query_policy" in spec:
            out["query_policy"] = spec["query_policy"]
        if "active_mode" in spec:
            out["active_mode"] = spec["active_mode"]
        if "forward_weight_mode" in spec:
            out["forward_weight_mode"] = spec["forward_weight_mode"]
        if "use_oracle_phi_for_acquisition" in spec:
            out["use_oracle_phi_for_acquisition"] = bool(spec["use_oracle_phi_for_acquisition"])
        if "learner_reward_model" in spec:
            out["learner_reward_model"] = spec["learner_reward_model"]
        return out

    key = _normalize_key(spec)
    if key not in METHODS:
        raise KeyError(f"Unknown method '{spec}'. Valid methods: {', '.join(METHODS)}")
    return dict(METHODS[key], key=key)
