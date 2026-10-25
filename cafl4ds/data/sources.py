"""Data sources: produce ``(images, labels)`` tensors for the streams to order.

A :class:`DataSource` is the *raw material* a stream orders into eras — it decouples the
stream/ordering logic from where the pixels come from. Two sources exist in Phase 0:

* :class:`STL10Source` — the real STL-10 labeled split (Coates et al. 2011), resized tiny for
  CPU. Labels are carried only so the stream can build class-blocked ordering and held-out
  eval sets; they never reach the SSL update.
* :class:`CIFAR100Source` — the real CIFAR-100 labeled split (Krizhevsky 2009), the canonical
  catastrophic-forgetting benchmark; same label-only contract.
* :class:`SyntheticSource` — class-structured Gaussian blobs, network-free, for fast unit
  tests and the fastest smoke runs.

Both return images as ``float32`` ``[N, C, H, W]`` in ``[0, 1]`` and integer labels ``[N]``.
Adding BDD100K/ZOD later means adding a new source, not touching the stream.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from loguru import logger
from torchvision import transforms
from torchvision.datasets import CIFAR100, DTD, STL10, EuroSAT, ImageFolder


class DataSource(ABC):
    """Produces ``(images, labels)`` tensors for a stream to order into eras."""

    @abstractmethod
    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the full dataset into memory.

        Returns:
            A tuple ``(images, labels)`` with ``images`` of shape ``[N, C, H, W]``
            (``float32`` in ``[0, 1]``) and integer ``labels`` of shape ``[N]``.
        """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of distinct classes in the source."""


class STL10Source(DataSource):
    """The real STL-10 labeled split, resized to a tiny CPU-friendly size."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 32,
        max_per_class: int | None = None,
    ) -> None:
        """Configure the STL-10 source.

        Args:
            root: Directory holding the downloaded ``stl10_binary`` (torchvision layout).
            split: Which labeled split to load (``"train"`` or ``"test"``).
            img_size: Side length to bilinearly resize the 96px images to.
            max_per_class: If set, keep at most this many images per class (tiny runs).
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_per_class = max_per_class

    @property
    def num_classes(self) -> int:
        """STL-10 has 10 classes."""
        return 10

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and (optionally) per-class-subsample the STL-10 split.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]``.

        Raises:
            FileNotFoundError: If the STL-10 binaries are not present under ``root``.
        """
        if not (Path(self.root) / "stl10_binary").is_dir():
            raise FileNotFoundError(
                f"STL-10 binaries not found under {self.root}. Download once with "
                "torchvision.datasets.STL10(root=..., split=..., download=True)."
            )
        ds = STL10(root=self.root, split=self.split, download=False)
        labels = torch.from_numpy(ds.labels).long()
        if self.max_per_class is not None:
            # Select while the source is still uint8. This matters for STL-10's 100k-image
            # unlabeled split: converting the entire array to float before applying a small
            # experiment cap needlessly allocates more than 10 GiB.
            keep = _subsample_indices(labels, self.max_per_class)
            images = torch.from_numpy(ds.data[keep.numpy()]).float() / 255.0
            labels = labels[keep]
        else:
            images = torch.from_numpy(ds.data).float() / 255.0  # [N, 3, 96, 96]
        images = F.interpolate(images, size=self.img_size, mode="bilinear", align_corners=False, antialias=True)
        logger.info(f"STL10Source: loaded {images.shape[0]} images ({self.split}) at {self.img_size}px")
        return images, labels


class CIFAR100Source(DataSource):
    """The real CIFAR-100 labeled split (Krizhevsky 2009), resized for the HPU.

    CIFAR-100 is the canonical catastrophic-forgetting benchmark: 100 fine object classes with
    genuinely distinct, learnable structure, so disjoint class-group tasks respecialize (and
    erode) far more strongly than STL-10's transferable-feature split. Same interface as
    :class:`STL10Source`.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 32,
        max_per_class: int | None = None,
        transform: str | None = None,
        transform_seed: int = 0,
    ) -> None:
        """Configure the CIFAR-100 source.

        Args:
            root: Directory holding the downloaded ``cifar-100-python`` (torchvision layout).
            split: Which labeled split to load (``"train"`` or ``"test"``).
            img_size: Side length to bilinearly resize the native 32px images to.
            max_per_class: If set, keep at most this many images per class (tiny runs).
            transform: Optional pixel transform applied to the loaded phase-B images before resize
                (P0.3.8 distribution-shift vehicles): ``"grayscale"`` (cosmetic chroma drop) or
                ``"phase_scramble"`` (structure-destroying Fourier phase randomization). ``None``
                keeps the native color images.
            transform_seed: RNG seed for the (stochastic) ``phase_scramble`` transform, so the
                shifted phase-B distribution is reproducible across the PC and healthy arms.
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_per_class = max_per_class
        self.transform = transform
        self.transform_seed = transform_seed

    @property
    def num_classes(self) -> int:
        """CIFAR-100 has 100 fine classes."""
        return 100

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and (optionally) per-class-subsample the CIFAR-100 split.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]``.

        Raises:
            FileNotFoundError: If the CIFAR-100 data is not present under ``root``.
        """
        if not (Path(self.root) / "cifar-100-python").is_dir():
            raise FileNotFoundError(
                f"CIFAR-100 data not found under {self.root}. Download once with "
                "torchvision.datasets.CIFAR100(root=..., train=..., download=True)."
            )
        ds = CIFAR100(root=self.root, train=(self.split == "train"), download=False)
        images = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0  # [N, 3, 32, 32] (HWC->CHW)
        labels = torch.tensor(ds.targets, dtype=torch.long)
        if self.max_per_class is not None:
            images, labels = _subsample_per_class(images, labels, self.max_per_class)
        if self.transform is not None:
            images = _apply_pixel_transform(images, self.transform, self.transform_seed)
        images = F.interpolate(images, size=self.img_size, mode="bilinear", align_corners=False, antialias=True)
        suffix = f" [{self.transform}]" if self.transform is not None else ""
        logger.info(f"CIFAR100Source: loaded {images.shape[0]} images ({self.split}) at {self.img_size}px{suffix}")
        return images, labels


class ImagenetteSource(DataSource):
    """Imagenette — a 10-class ImageNet subset (fast.ai), the MAE ViT-B *pretraining-distribution* proxy.

    P0.3.6's forgetting fire needs the well's own source distribution as a *replayable* task A: the
    MAE ViT-B task-A probe (~0.9+) is sourced from ImageNet pretraining, so a replay control can only
    protect it by revisiting ImageNet-distribution data — not a proxy from another corpus (that would
    only pull the representation *further* from the ImageNet manifold). Imagenette is exactly that
    slice: 10 full ImageNet classes of native photos, letting us keep the standard FB MAE checkpoint
    while making the replay control well-posed. Same label-only contract as the other sources (labels
    feed the probe / task split, never the SSL loss). Download once from fast.ai
    (``imagenette2-320.tgz``) and extract under ``root``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 160,
        max_per_class: int | None = None,
    ) -> None:
        """Configure the Imagenette source.

        Args:
            root: Directory holding the extracted ``imagenette2-320`` (``train/<wnid>/*.JPEG`` layout).
            split: Which split to load (``"train"`` or ``"val"``).
            img_size: Side length to resize the native images to (upsampled to 224 by the ViT-B
                encoder). Kept independent of the harness's global ``img_size`` so task A can stay at
                a faithful ImageNet-native resolution while task B (e.g. CIFAR) stays at 32.
            max_per_class: If set, keep at most this many images per class (bounds memory / run time).
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_per_class = max_per_class

    @property
    def num_classes(self) -> int:
        """Imagenette has 10 classes."""
        return 10

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and (optionally) per-class-subsample the Imagenette split.

        Only the *selected* images are decoded (paths are indexed first, then capped per class), so a
        small ``max_per_class`` keeps the load cheap despite the full-resolution source.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]`` and labels
            ``0..9`` (sorted WordNet-id order).

        Raises:
            FileNotFoundError: If the Imagenette split directory is not present under ``root``.
        """
        base = Path(self.root) / ("train" if self.split == "train" else "val")
        if not base.is_dir():
            raise FileNotFoundError(
                f"Imagenette split not found under {base}. Download imagenette2-320.tgz from "
                "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and extract under root."
            )
        folder = ImageFolder(str(base))  # indexes (path, label); decode is deferred to loader() below
        resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        by_cls: dict[int, list[str]] = {}
        for path, label in folder.samples:
            by_cls.setdefault(label, []).append(path)
        imgs, labels = [], []
        for label in sorted(by_cls):
            paths = by_cls[label][: self.max_per_class] if self.max_per_class is not None else by_cls[label]
            for path in paths:
                imgs.append(resize(folder.loader(path).convert("RGB")))
                labels.append(label)
        images = torch.stack(imgs)
        logger.info(f"ImagenetteSource: loaded {images.shape[0]} images ({self.split}) at {self.img_size}px")
        return images, torch.tensor(labels, dtype=torch.long)


class EuroSATSource(DataSource):
    """EuroSAT — Sentinel-2 satellite RGB, a phase-B domain deliberately *far* from ImageNet.

    A far-from-pretraining phase-B vehicle for the JE forgetting study ([P0.6](../../docs/experiments/
    phase0/P0.6.0.md)): where CIFAR shares ImageNet's natural-object-photo statistics (so continued SSL
    training on it largely *reinforces* the task-A features rather than overwriting them), aerial
    land-cover imagery has low-level and semantic statistics unlike ImageNet objects — so phase-B
    training on it should respecialize the encoder *harder*, a stronger test of whether a
    joint-embedding backbone forgets than the same-domain CIFAR shift. 10 land-cover classes, ~27k
    native 64px images. Same label-only contract as the other sources (labels feed the probe / task
    split, never the SSL loss). Download once with ``torchvision.datasets.EuroSAT(root=..., download=
    True)`` and point ``root`` at it.
    """

    def __init__(self, root: str, img_size: int = 64, max_per_class: int | None = None) -> None:
        """Configure the EuroSAT source.

        Args:
            root: Directory holding the downloaded ``eurosat`` (torchvision layout).
            img_size: Side length to resize the native 64px images to (upsampled to 224 by the ViT-B
                encoder). Kept independent of the harness's global ``img_size`` so task A (Imagenette)
                can stay at its own resolution.
            max_per_class: If set, keep at most this many images per class (bounds memory / run time).
        """
        self.root = root
        self.img_size = img_size
        self.max_per_class = max_per_class

    @property
    def num_classes(self) -> int:
        """EuroSAT has 10 land-cover classes."""
        return 10

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and (optionally) per-class-subsample EuroSAT.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]`` and labels
            ``0..9`` (sorted class order).

        Raises:
            FileNotFoundError: If the EuroSAT data is not present under ``root``.
        """
        if not (Path(self.root) / "eurosat").is_dir():
            raise FileNotFoundError(
                f"EuroSAT data not found under {self.root}. Download once with "
                "torchvision.datasets.EuroSAT(root=..., download=True)."
            )
        ds = EuroSAT(root=self.root, download=False)  # ImageFolder-backed: exposes (path, label) samples
        resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        by_cls: dict[int, list[str]] = {}
        for path, label in ds.samples:
            by_cls.setdefault(label, []).append(path)
        imgs, labels = [], []
        for label in sorted(by_cls):
            paths = by_cls[label][: self.max_per_class] if self.max_per_class is not None else by_cls[label]
            for path in paths:
                imgs.append(resize(ds.loader(path).convert("RGB")))
                labels.append(label)
        images = torch.stack(imgs)
        logger.info(f"EuroSATSource: loaded {images.shape[0]} images at {self.img_size}px")
        return images, torch.tensor(labels, dtype=torch.long)


class DTDSource(DataSource):
    """DTD (Describable Textures) — a *far* phase-B domain matched for inpainting difficulty (P0.3.10).

    The far-domain MAE forgetting test needs a phase B that is both (a) genuinely far from the ImageNet
    object manifold and (b) as *hard to inpaint* as task A, because in MAE the reconstruction-loss
    magnitude sets the gradient magnitude, hence the encoder-update pressure a phase-B domain applies
    (the P0.4.1 finding: recon loss reads inpaintability, not novelty). EuroSAT is far but *easy* to
    inpaint (blurred 64px tiles), so it under-powers the shift. DTD textures are far (texture, not
    objects) and, at native high resolution, ~1.26x *harder* to inpaint than Imagenette at the
    phase-B-start condition — a conservative choice (it applies more update pressure than continued
    task-A training, so a resistance verdict cannot be dismissed as under-powered).

    DTD is small — 47 classes x 120 images (pooled over the train/val/test splits of partition 1). To
    supply enough phase-B training data at the harness's 60/60 eval sizing, textures (crop/flip/rotation-
    invariant by construction) are **augmented** up to ``max_per_class`` with random-resized crops, flips
    and 90-degree rotations. The augmentation draws from a *fixed* internal seed and restores the global
    RNG, so the far-domain dataset is identical across model seeds (only the model varies) and training
    numerics are untouched. Same label-only contract as the other sources. Any crop-leakage between the
    augmented train and eval pools touches only the *secondary* task-B probe; the forgetting ground truth
    is the clean, un-augmented task-A (Imagenette) probe.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        max_per_class: int | None = None,
        partition: int = 1,
        augment_seed: int = 0,
        classes: list[int] | None = None,
    ) -> None:
        """Configure the DTD source.

        Args:
            root: Directory holding the downloaded ``dtd`` (torchvision layout).
            img_size: Side length to resize / crop the native high-res textures to.
            max_per_class: Target images per class. When it exceeds the ~120 pooled base images, the
                remainder is filled with texture-preserving augmented crops (``None`` = base only).
            partition: DTD split partition (1..10); the three splits of one partition are pooled.
            augment_seed: Fixed seed for the fill augmentation, so the far-domain data is seed-stable.
            classes: If set, load only these class ids (the phase-B split uses ≤ 10 of DTD's 47 classes,
                so restricting the load avoids decoding + augmenting the ~37 unused classes).
        """
        self.root = root
        self.img_size = img_size
        self.max_per_class = max_per_class
        self.partition = partition
        self.augment_seed = augment_seed
        self.classes = classes

    @property
    def num_classes(self) -> int:
        """DTD has 47 texture classes."""
        return 47

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the pooled DTD splits, resize, and augment each class up to ``max_per_class``.

        Returns:
            ``(images, labels)`` with images ``[N, 3, img_size, img_size]`` in ``[0, 1]`` and labels
            ``0..46`` (sorted class order); base images first, then augmented crops.

        Raises:
            FileNotFoundError: If the DTD data is not present under ``root``.
        """
        if not (Path(self.root) / "dtd").is_dir():
            raise FileNotFoundError(
                f"DTD data not found under {self.root}. Download once with "
                "torchvision.datasets.DTD(root=..., split='train', download=True)."
            )
        keep = set(self.classes) if self.classes is not None else None
        by_cls: dict[int, list[str]] = {}
        for split in ("train", "val", "test"):
            ds = DTD(root=self.root, split=split, partition=self.partition, download=False)
            for path, label in zip(ds._image_files, ds._labels, strict=True):
                if keep is None or int(label) in keep:
                    by_cls.setdefault(int(label), []).append(str(path))
        loader = DTD(root=self.root, split="train", partition=self.partition, download=False).loader

        base_resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.img_size, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice([transforms.RandomRotation((r, r)) for r in (0, 90, 180, 270)]),
                transforms.ToTensor(),
            ]
        )
        # Deterministic, RNG-neutral augmentation: seed both RNGs the transforms draw from (torch +
        # Python `random`), then restore global state on exit so the far-domain data is seed-stable.
        torch_state, py_state = torch.get_rng_state(), random.getstate()
        torch.manual_seed(self.augment_seed)
        random.seed(self.augment_seed)
        try:
            imgs: list[torch.Tensor] = []
            labels: list[int] = []
            for label in sorted(by_cls):
                paths = by_cls[label]
                pil = [loader(p).convert("RGB") for p in paths]
                target = self.max_per_class if self.max_per_class is not None else len(pil)
                base = pil[:target]
                imgs.extend(base_resize(im) for im in base)
                labels.extend([label] * len(base))
                for i in range(len(base), target):  # fill the remainder with texture-preserving crops
                    imgs.append(aug(pil[i % len(pil)]))
                    labels.append(label)
        finally:
            torch.set_rng_state(torch_state)
            random.setstate(py_state)
        images = torch.stack(imgs)
        logger.info(
            f"DTDSource: loaded {images.shape[0]} images at {self.img_size}px (target {self.max_per_class}/class)"
        )
        return images, torch.tensor(labels, dtype=torch.long)


class SyntheticSource(DataSource):
    """Class-structured Gaussian images — network-free, for tests and fast smoke runs."""

    def __init__(
        self,
        num_classes: int = 4,
        per_class: int = 64,
        img_size: int = 16,
        channels: int = 3,
        noise: float = 0.3,
        seed: int = 0,
    ) -> None:
        """Configure the synthetic source.

        Args:
            num_classes: Number of classes to generate.
            per_class: Images per class.
            img_size: Image side length.
            channels: Number of channels.
            noise: Standard deviation of the per-pixel Gaussian noise around each class mean.
            seed: RNG seed for reproducibility.
        """
        self._num_classes = num_classes
        self.per_class = per_class
        self.img_size = img_size
        self.channels = channels
        self.noise = noise
        self.seed = seed

    @property
    def num_classes(self) -> int:
        """Return the configured class count."""
        return self._num_classes

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate class-structured images (a distinct random mean pattern per class).

        Returns:
            ``(images, labels)`` with images ``[N, C, img_size, img_size]`` in ``[0, 1]``,
            where a class's images cluster around its own random pattern (so probes are
            learnable and the effective rank is meaningful).
        """
        g = torch.Generator().manual_seed(self.seed)
        shape = (self.channels, self.img_size, self.img_size)
        images, labels = [], []
        for c in range(self._num_classes):
            mean = torch.rand(shape, generator=g)
            block = mean.unsqueeze(0) + self.noise * torch.randn(self.per_class, *shape, generator=g)
            images.append(block.clamp_(0.0, 1.0))
            labels.append(torch.full((self.per_class,), c, dtype=torch.long))
        return torch.cat(images), torch.cat(labels)


def _apply_pixel_transform(images: torch.Tensor, kind: str, seed: int) -> torch.Tensor:
    """Apply a P0.3.8 distribution-shift transform to ``[N, C, H, W]`` images in ``[0, 1]``.

    Args:
        images: Loaded images ``[N, 3, H, W]`` in ``[0, 1]``.
        kind: ``"grayscale"`` (cosmetic) or ``"phase_scramble"`` (structure-destroying).
        seed: RNG seed for the stochastic ``phase_scramble`` (ignored by ``grayscale``).

    Returns:
        The transformed images ``[N, 3, H, W]`` in ``[0, 1]``.

    Raises:
        ValueError: If ``kind`` is not a known transform.
    """
    if kind == "grayscale":
        return _to_grayscale(images)
    if kind == "phase_scramble":
        return _phase_scramble(images, seed)
    raise ValueError(f"Unknown pixel transform {kind!r} (expected 'grayscale' or 'phase_scramble').")


def _to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Drop chroma via Rec.601 luminance broadcast back to 3 channels — a *cosmetic* shift.

    Leaves spatial/object structure intact (grayscale is a sub-projection of color).

    Args:
        images: Images ``[N, 3, H, W]`` in ``[0, 1]``.

    Returns:
        The de-chromatized images ``[N, 3, H, W]`` (all channels equal).
    """
    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luminance = (images * weights).sum(dim=1, keepdim=True)  # [N, 1, H, W]
    return luminance.expand(-1, 3, -1, -1).contiguous()


def _phase_scramble(images: torch.Tensor, seed: int) -> torch.Tensor:
    """Destroy spatial structure while *preserving the per-channel power spectrum*.

    Keep each image's Fourier amplitude, swap in the phase of a white-noise field (one field per
    image, shared across channels → a coherent texture with a real inverse), inverse-FFT, and
    per-image min-max renormalize to ``[0, 1]``. Objects dissolve into texture; the amplitude
    statistics task A relies on survive — the "hard-disjoint" phase-B vehicle.

    Args:
        images: Images ``[N, 3, H, W]`` in ``[0, 1]``.
        seed: RNG seed for the random phase field (reproducible across arms).

    Returns:
        The phase-scrambled images ``[N, 3, H, W]`` in ``[0, 1]``.
    """
    n, c, h, w = images.shape
    generator = torch.Generator().manual_seed(seed)
    amplitude = torch.fft.fft2(images).abs()  # [N, C, H, W], real
    noise = torch.rand(n, 1, h, w, generator=generator)  # one phase field per image
    phase = torch.angle(torch.fft.fft2(noise))  # conjugate-symmetric → real ifft
    scrambled = torch.fft.ifft2(amplitude * torch.exp(1j * phase)).real
    flat = scrambled.reshape(n, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).reshape(n, c, h, w)


def _subsample_per_class(
    images: torch.Tensor, labels: torch.Tensor, max_per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most ``max_per_class`` images of each class (first-occurring, order-preserving).

    Args:
        images: All images ``[N, C, H, W]``.
        labels: All labels ``[N]``.
        max_per_class: Cap on images retained per class.

    Returns:
        The subsampled ``(images, labels)``.
    """
    idx = _subsample_indices(labels, max_per_class)
    return images[idx], labels[idx]


def _subsample_indices(labels: torch.Tensor, max_per_class: int) -> torch.Tensor:
    """Return order-preserving indices capped to ``max_per_class`` for each label."""
    keep: list[int] = []
    counts: dict[int, int] = {}
    for i, y in enumerate(labels.tolist()):
        if counts.get(y, 0) < max_per_class:
            keep.append(i)
            counts[y] = counts.get(y, 0) + 1
    return torch.tensor(keep, dtype=torch.long)
