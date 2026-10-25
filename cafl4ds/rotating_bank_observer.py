"""A read-only rotating reconstruction assay with explicit composition accounting."""

import copy

import numpy as np
import torch

from cafl4ds.single_image_forecast import fixed_reconstruction


class RotatingBankObserver:
    def __init__(self, bank, stream, cfg, start, state=None):
        self.cfg, self.stream, self.start = cfg, stream, start
        if state is None:
            ids = np.linspace(0, len(bank) - 1, cfg["size"], dtype=int)
            self.state = dict(members=[dict(source="fixed", index=int(i), born=0) for i in ids], rows=[], swaps=[])
        else:
            self.state = copy.deepcopy(state)
        self.images = torch.stack([
            bank[m["index"]] if m["source"] == "fixed" else stream[m["index"]]
            for m in self.state["members"]
        ])

    def losses(self, model):
        # A stable mask seed per slot: inserting an image cannot change any other
        # member's mask. These functions restore learner RNG and module modes.
        return np.array([
            float(fixed_reconstruction(model, self.images[j:j + 1], self.cfg["mask_seed"] + j)[0])
            for j in range(len(self.images))
        ])

    def read(self, model, offered, accepted):
        before = self.losses(model)
        previous = self.state.get("last_losses")
        learning_change = float(before.mean() - np.mean(previous)) if previous is not None else 0.0
        for m, loss in zip(self.state["members"], before, strict=True):
            m.setdefault("entry_loss", float(loss))
        members_before = copy.deepcopy(self.state["members"])
        after = before.copy()
        if offered >= self.cfg["rotate_after"] and offered > 0:
            # Independent per-block generator: no training RNG consumption and
            # no dependence on predictor score, acceptance or reconstruction.
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg["selection_seed"], offered]))
            local = int(rng.integers(max(0, offered - self.cfg["cadence"]), offered))
            slot = len(self.state["swaps"]) % len(self.images)
            self.images[slot] = self.stream[self.start + local]
            loss = float(fixed_reconstruction(model, self.images[slot:slot + 1], self.cfg["mask_seed"] + slot)[0])
            member = dict(source="stream", index=self.start + local, born=offered,
                          entry_loss=loss, trained_on=bool(accepted[local]))
            self.state["members"][slot] = member
            after[slot] = loss
            self.state["swaps"].append(dict(offered=offered, slot=slot, removed=members_before[slot], admitted=member))
        composition_change = float(after.mean() - before.mean())
        assert np.isclose(after.mean() - (np.mean(previous) if previous is not None else before.mean()),
                          learning_change + composition_change)
        self.state["rows"].append(dict(
            offered=offered, mean_before_swap=float(before.mean()), mean_after_swap=float(after.mean()),
            learning_change=learning_change, composition_change=composition_change,
            losses_before=before.tolist(), members_before=members_before,
        ))
        self.state["last_losses"] = after.tolist()
