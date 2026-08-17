import torch

from CSDNet.model.lightning_module import PROP_NAMES, PROP_SCALES


def scale_property_values(values):
    return torch.tensor(
        [
            float(values["qed"]) / PROP_SCALES["qed"],
            float(values["logp"]) / PROP_SCALES["logp"],
            float(values["sa"]) / PROP_SCALES["sa"],
            float(values["tpsa"]) / PROP_SCALES["tpsa"],
            float(values["mw"]) / PROP_SCALES["mw"],
        ],
        dtype=torch.float,
    )


def make_condition_tensor(values, active_props, cond_dim):
    active = set(active_props)
    if cond_dim == 0:
        raise SystemExit("This checkpoint is unconditional and cannot use property control.")

    if cond_dim == len(PROP_NAMES):
        if active != set(PROP_NAMES):
            raise SystemExit(
                "This is a legacy 5D full-condition checkpoint; provide all five properties."
            )
        return scale_property_values(values)

    if cond_dim == len(PROP_NAMES) * 2:
        scaled = scale_property_values(values)
        mask = torch.tensor([1.0 if p in active else 0.0 for p in PROP_NAMES], dtype=torch.float)
        return torch.cat([scaled * mask, mask], dim=0)

    raise SystemExit(f"Unsupported cond_dim={cond_dim}.")


def build_condition_from_args(args, cond_dim):
    individual = {
        "qed": args.qed,
        "logp": args.logp,
        "sa": args.sa,
        "tpsa": args.tpsa,
        "mw": args.mw,
    }
    has_individual = any(v is not None for v in individual.values())
    if args.cond is not None and has_individual:
        raise SystemExit("Do not mix --cond with individual property arguments.")

    if args.cond is None and not has_individual:
        return None, "de novo generation", None

    if args.cond is not None:
        values = dict(zip(PROP_NAMES, args.cond))
        active_props = list(PROP_NAMES)
    else:
        values = {p: 0.0 for p in PROP_NAMES}
        active_props = []
        for prop, value in individual.items():
            if value is not None:
                values[prop] = value
                active_props.append(prop)

    cond_tensor = make_condition_tensor(values, active_props, cond_dim)
    pretty = ", ".join(f"{p.upper()}={values[p]}" for p in active_props)
    return cond_tensor, f"property subset control ({pretty})", (values, active_props)
