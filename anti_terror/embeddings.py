from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from loguru import logger
from torchvision import models, transforms
import cv2
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from .config import EmbeddingConfig, select_device


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def compute_blur_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as blur metric. Higher = sharper."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


@dataclass
class FaceQuality:
    """Quality metrics for a detected face."""
    blur_score: float = 0.0
    face_size: int = 0
    detection_score: float = 0.0
    yaw: float = 0.0  # head rotation left/right
    pitch: float = 0.0  # head rotation up/down

    @property
    def overall_quality(self) -> float:
        """Compute overall quality score [0, 1]."""
        # Normalize blur score (typical range 0-1000+)
        blur_norm = min(self.blur_score / 500.0, 1.0)
        # Normalize face size (min 50px, ideal 150px+)
        size_norm = min(max(self.face_size - 50, 0) / 100.0, 1.0)
        # Penalize extreme head poses (threshold ~30 degrees)
        pose_penalty = 1.0 - min(abs(self.yaw) / 45.0, 1.0) * 0.3 - min(abs(self.pitch) / 45.0, 1.0) * 0.3

        return (blur_norm * 0.3 + size_norm * 0.3 + self.detection_score * 0.2 + pose_penalty * 0.2)


@dataclass
class EmbeddingSample:
    vectors: list[torch.Tensor]  # normalized embeddings history
    label: str
    image_path: str | None = None
    centroid: torch.Tensor | None = None  # cached centroid for fast matching
    quality_scores: list[float] = field(default_factory=list)  # quality for each vector


class BagEmbedder:
    """Bag embedding using DINOv2 for instance-level discrimination.

    DINOv2 is a self-supervised Vision Transformer trained for instance
    discrimination — it naturally separates individual objects, unlike
    ResNet50 which groups objects by category.

    DINOv2-small: 384-dim, fast, excellent discrimination.
    """

    def __init__(self, cfg: EmbeddingConfig):
        device = select_device(cfg.device)
        model_name = getattr(cfg, 'bag_model_name', 'dinov2_vits14')

        if model_name.startswith("dinov2"):
            # DINOv2 via torch.hub
            logger.info(f"Loading DINOv2 ({model_name}) for bag embeddings...")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", model_name, verbose=False
            )
            self.model.eval().to(device)
            self.embed_dim = self.model.embed_dim  # 384 for vits14
            self._is_dino = True
        else:
            # Fallback to ResNet50
            if model_name.lower() == "resnet50":
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
            backbone.eval().to(device)
            self.model = backbone
            self.embed_dim = 2048 if "50" in model_name else 512
            self._is_dino = False

        self.device = device
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"Bag embedder ready on {device} "
                     f"({model_name}, {self.embed_dim}-dim)")

    @torch.inference_mode()
    def __call__(self, crop: np.ndarray) -> torch.Tensor:
        """Extract embedding from bag crop.

        Returns normalized embedding (384-dim for DINOv2-small).
        """
        tensor = self.transform(crop).unsqueeze(0).to(self.device)

        if self._is_dino:
            # DINOv2 cls_token output
            emb = self.model(tensor)  # (1, embed_dim)
        else:
            # ResNet: global average pool
            emb = self.model(tensor).flatten(1)  # (1, 2048)

        emb = F.normalize(emb, dim=1).cpu().squeeze(0)
        return emb


class FaceEmbedder:
    """Face detection and embedding extraction with quality assessment.

    Uses an InsightFace ArcFace pack when available, with FaceNet fallback.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.provider = cfg.face_provider.lower()
        if self.provider == "insightface" and FaceAnalysis is not None:
            # Select providers based on platform
            # - CUDA for NVIDIA GPUs (Linux/Windows)
            # - CoreML for Apple Silicon (macOS M1/M2/M3/M4)
            # - CPU as fallback
            device = select_device(cfg.device)
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device == "mps":
                # Apple Silicon: CoreML provides GPU acceleration
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            selected_model = self._resolve_insightface_model_name(cfg.face_model_name)
            fallback_model = getattr(cfg, "face_fallback_model_name", cfg.face_model_name)
            self.device = device

            try:
                self.model = self._load_insightface_model(selected_model, providers, device)
                self.active_model_name = selected_model
                logger.info(f"Face embedder ready (InsightFace {self.active_model_name})")
                return
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize InsightFace model '{selected_model}': {exc}"
                )

            if fallback_model != selected_model:
                try:
                    self.model = self._load_insightface_model(fallback_model, providers, device)
                    self.active_model_name = fallback_model
                    logger.warning(
                        f"Falling back to InsightFace model '{fallback_model}'"
                    )
                    logger.info(f"Face embedder ready (InsightFace {self.active_model_name})")
                    return
                except Exception as exc:
                    logger.warning(
                        f"Failed to initialize fallback InsightFace model '{fallback_model}': {exc}"
                    )

            logger.warning("Switching face embedder to FaceNet fallback")

        device = select_device(cfg.device)
        self.mtcnn = MTCNN(
            image_size=160,
            device=device,
            thresholds=[0.7, 0.8, 0.9],  # stricter thresholds
            keep_all=True
        )
        self.model = InceptionResnetV1(pretrained=cfg.face_model).eval().to(device)
        self.device = device
        self.provider = "facenet"
        self.active_model_name = cfg.face_model
        logger.info(f"Face embedder ready (FaceNet {self.active_model_name}) on {device}")

    def _resolve_insightface_model_name(self, requested_name: str) -> str:
        model_root = Path(getattr(self.cfg, "face_model_root", Path.home() / ".insightface" / "models")).expanduser()
        fallback_name = getattr(self.cfg, "face_fallback_model_name", requested_name)

        requested_path = model_root / requested_name
        if requested_path.exists():
            return requested_name

        fallback_path = model_root / fallback_name
        if requested_name != fallback_name and fallback_path.exists():
            logger.warning(
                f"InsightFace model '{requested_name}' is not installed in {model_root}. "
                f"Using local fallback '{fallback_name}'."
            )
            return fallback_name

        return requested_name

    @staticmethod
    def _load_insightface_model(model_name: str, providers: list[str], device: str) -> FaceAnalysis:
        model = FaceAnalysis(name=model_name, providers=providers)
        # ctx_id: 0 for GPU (CUDA), -1 for CPU/CoreML
        ctx_id = 0 if device == "cuda" else -1
        # Larger det_size improves small face detection
        model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return model

    def _compute_quality(self, face_crop: np.ndarray, det_score: float,
                         pose: Optional[np.ndarray] = None) -> FaceQuality:
        """Compute quality metrics for a face crop."""
        blur = compute_blur_score(face_crop)
        h, w = face_crop.shape[:2]
        face_size = min(h, w)

        yaw, pitch = 0.0, 0.0
        if pose is not None and len(pose) >= 2:
            yaw = float(pose[1]) if len(pose) > 1 else 0.0
            pitch = float(pose[0]) if len(pose) > 0 else 0.0

        return FaceQuality(
            blur_score=blur,
            face_size=face_size,
            detection_score=det_score,
            yaw=yaw,
            pitch=pitch
        )

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, torch.Tensor, np.ndarray, FaceQuality]]:
        """Detect faces and extract embeddings with quality metrics.

        Returns: List of (bbox_xyxy, score, embedding, face_crop, quality)
        """
        if self.provider == "insightface" and FaceAnalysis is not None:
            faces = self.model.get(frame)
            results = []
            for f in faces:
                box = f.bbox.astype(float)  # x1,y1,x2,y2
                score = float(f.det_score)
                if score < self.cfg.face_confidence:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                face_crop = frame[y1:y2, x1:x2].copy()

                # Skip very small faces
                if face_crop.shape[0] < self.cfg.min_face_size or face_crop.shape[1] < self.cfg.min_face_size:
                    continue

                # Get pose from InsightFace (if available)
                pose = getattr(f, 'pose', None)
                quality = self._compute_quality(face_crop, score, pose)

                # Skip low quality faces
                if quality.overall_quality < self.cfg.min_face_quality:
                    logger.debug(f"Skipping low quality face: {quality.overall_quality:.2f}")
                    continue

                emb = torch.tensor(f.normed_embedding, dtype=torch.float32)
                results.append((box, score, emb, face_crop, quality))
            return results

        # FaceNet fallback
        boxes, probs = self.mtcnn.detect(frame)
        results = []
        if boxes is None:
            return results
        aligned = self.mtcnn(frame, save_path=None)
        if aligned is None:
            return results

        # Handle different return formats from MTCNN
        if isinstance(aligned, torch.Tensor):
            if aligned.dim() == 3:
                # Single face: [3, 160, 160] -> [1, 3, 160, 160]
                aligned = aligned.unsqueeze(0)
            elif aligned.dim() == 4:
                # Multiple faces: [N, 3, 160, 160] - already correct
                pass
            else:
                return results
        elif isinstance(aligned, list):
            # List of tensors - stack them
            aligned = torch.stack([a for a in aligned if a is not None])
            if len(aligned) == 0:
                return results
        else:
            return results

        embeddings = self.model(aligned.to(self.device))
        embeddings = F.normalize(embeddings, dim=1)
        for box, prob, emb in zip(boxes, probs, embeddings):
            if prob is None or prob < self.cfg.face_confidence:
                continue
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            face_crop = frame[y1:y2, x1:x2].copy()

            # Skip very small faces
            if face_crop.shape[0] < self.cfg.min_face_size or face_crop.shape[1] < self.cfg.min_face_size:
                continue

            quality = self._compute_quality(face_crop, float(prob))
            if quality.overall_quality < self.cfg.min_face_quality:
                continue

            results.append((box.astype(float), float(prob), emb.cpu(), face_crop, quality))
        return results


class EmbeddingStore:
    """In-memory store for cosine matching with centroid-based comparison.

    Key improvements for reducing duplicate IDs:
    1. Uses quality-weighted centroid instead of max similarity
    2. Considers both centroid and recent embeddings for matching
    3. Stricter creation logic with global duplicate check
    """

    def __init__(self, max_history: int = 20):
        self.samples: Dict[str, EmbeddingSample] = {}
        self.counter = 0
        self.max_history = max_history

    def _new_label(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def _update_centroid(self, sample: EmbeddingSample) -> None:
        """Recompute quality-weighted centroid for a sample."""
        if not sample.vectors:
            sample.centroid = None
            return

        if not sample.quality_scores or len(sample.quality_scores) != len(sample.vectors):
            # Fallback: uniform weights
            weights = torch.ones(len(sample.vectors))
        else:
            weights = torch.tensor(sample.quality_scores)
            weights = weights / weights.sum()  # normalize

        stacked = torch.stack(sample.vectors)
        weighted_sum = (stacked * weights.unsqueeze(1)).sum(dim=0)
        sample.centroid = F.normalize(weighted_sum, dim=0)

    def get_vector(self, label: str) -> torch.Tensor | None:
        """Get the centroid vector for a label (preferred) or last vector."""
        sample = self.samples.get(label)
        if sample:
            if sample.centroid is not None:
                return sample.centroid
            return sample.vectors[-1] if sample.vectors else None
        return None

    def get_centroid(self, label: str) -> torch.Tensor | None:
        """Get the centroid for a label."""
        sample = self.samples.get(label)
        return sample.centroid if sample else None

    def find_best(self, emb: torch.Tensor, use_centroid: bool = True) -> tuple[str | None, float]:
        """Find best matching label using centroid comparison.

        Args:
            emb: Query embedding
            use_centroid: If True, compare against centroid. If False, use max over history.
        """
        best_label = None
        best_score = -1.0

        for lbl, sample in self.samples.items():
            if use_centroid and sample.centroid is not None:
                # Primary: compare against centroid
                centroid_score = self.cosine(emb, sample.centroid)
                # Also check recent embeddings for robustness
                recent_scores = [self.cosine(emb, v) for v in sample.vectors[-5:]]
                recent_max = max(recent_scores) if recent_scores else 0.0
                # Combined score: 70% centroid + 30% best recent
                score = 0.7 * centroid_score + 0.3 * recent_max
            else:
                # Fallback: max similarity across history
                score = max(self.cosine(emb, vec) for vec in sample.vectors) if sample.vectors else 0.0

            if score > best_score:
                best_score = score
                best_label = lbl

        return best_label, best_score

    def find_all_matches(self, emb: torch.Tensor, threshold: float) -> List[Tuple[str, float]]:
        """Find all labels matching above threshold, sorted by score descending."""
        matches = []
        for lbl, sample in self.samples.items():
            if sample.centroid is not None:
                score = self.cosine(emb, sample.centroid)
            else:
                score = max(self.cosine(emb, vec) for vec in sample.vectors) if sample.vectors else 0.0
            if score >= threshold:
                matches.append((lbl, score))
        return sorted(matches, key=lambda x: x[1], reverse=True)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def add_embedding(self, label: str, emb: torch.Tensor, quality: float = 1.0) -> None:
        """Add embedding to existing label with quality score."""
        if label not in self.samples:
            return
        sample = self.samples[label]
        sample.vectors.append(emb)
        sample.quality_scores.append(quality)
        # Trim to max history
        if len(sample.vectors) > self.max_history:
            sample.vectors = sample.vectors[-self.max_history:]
            sample.quality_scores = sample.quality_scores[-self.max_history:]
        # Update centroid
        self._update_centroid(sample)

    def match_or_create(
        self,
        emb: torch.Tensor,
        prefix: str,
        threshold: float,
        image: np.ndarray | None = None,
        save_dir: Path | None = None,
        force_threshold: float | None = None,
        create_threshold: float | None = None,
        quality: float = 1.0,
        strict_mode: bool = False,
        track_frames_seen: int = 1,
        patience_frames: int = 1,
    ) -> tuple[str, bool, float]:
        """Return (label, created_new, best_score).

        Args:
            threshold: Primary similarity threshold for matching
            force_threshold: Secondary threshold to force match (prevents duplicates)
            create_threshold: Minimum score to even consider creating new ID
            quality: Quality score for this embedding
            strict_mode: If True (for bags), use force_threshold logic instead of aggressive merging
            track_frames_seen: How many frames this track has been observed
            patience_frames: Wait this many frames before creating new ID
        """
        if not self.samples:
            # First bag ever - but check patience
            if track_frames_seen < patience_frames:
                # Not enough frames - return empty (no ID yet)
                return "", False, 0.0
            label = self._new_label(prefix)
            path = self._maybe_save(image, save_dir, label)
            self.samples[label] = EmbeddingSample(
                vectors=[emb],
                label=label,
                image_path=path,
                centroid=emb.clone(),
                quality_scores=[quality]
            )
            return label, True, 1.0

        best_label, best_score = self.find_best(emb, use_centroid=True)

        # Primary match: above threshold
        if best_label and best_score >= threshold:
            self.add_embedding(best_label, emb, quality)
            return best_label, False, best_score

        # For strict mode (bags): use force_threshold to prevent duplicates
        if strict_mode:
            # Force-match if above force_threshold (even if below primary)
            if force_threshold is not None and best_label and best_score >= force_threshold:
                logger.debug(f"Force-matching bag to {best_label} (score={best_score:.3f})")
                self.add_embedding(best_label, emb, quality)
                return best_label, False, best_score

            # Patience check - don't create ID until enough frames observed
            if track_frames_seen < patience_frames:
                # Return best match tentatively (not creating new ID yet)
                if best_label:
                    return best_label, False, best_score
                return "", False, 0.0

            # Create new ID only after patience period and truly no match
            label = self._new_label(prefix)
            path = self._maybe_save(image, save_dir, label)
            self.samples[label] = EmbeddingSample(
                vectors=[emb],
                label=label,
                image_path=path,
                centroid=emb.clone(),
                quality_scores=[quality]
            )
            logger.info(f"Created new bag ID {label} (best_score was {best_score:.3f}, frames={track_frames_seen})")
            return label, True, best_score

        # --- Below is aggressive matching for faces only (strict_mode=False) ---

        # Secondary match: above force threshold (to reduce duplicates)
        if force_threshold is not None and best_label and best_score >= force_threshold:
            self.add_embedding(best_label, emb, quality)
            return best_label, False, best_score

        # Before creating new ID: double-check with stricter centroid comparison
        if best_label and best_score >= (force_threshold or threshold) - 0.1:
            # Very close to threshold - prefer existing ID to avoid duplication
            logger.debug(f"Near-threshold match ({best_score:.3f}), reusing {best_label}")
            self.add_embedding(best_label, emb, quality)
            return best_label, False, best_score

        # Create threshold check
        if create_threshold is not None and best_score >= create_threshold:
            # Score is above create_threshold but below match threshold
            # Check if this might be same person with different angle/lighting
            all_matches = self.find_all_matches(emb, create_threshold)
            if all_matches:
                # Prefer existing ID when in doubt
                top_match, top_score = all_matches[0]
                logger.debug(f"Create-threshold match ({top_score:.3f}), reusing {top_match}")
                self.add_embedding(top_match, emb, quality)
                return top_match, False, top_score

        # Very low score - still prefer existing if any reasonable match (faces only)
        if best_label and best_score >= 0.35:
            logger.debug(f"Low-score match ({best_score:.3f}), reusing {best_label} to avoid duplicate")
            self.add_embedding(best_label, emb, quality)
            return best_label, False, best_score

        # Create new ID only if truly no match
        label = self._new_label(prefix)
        path = self._maybe_save(image, save_dir, label)
        self.samples[label] = EmbeddingSample(
            vectors=[emb],
            label=label,
            image_path=path,
            centroid=emb.clone(),
            quality_scores=[quality]
        )
        logger.info(f"Created new ID {label} (best_score was {best_score:.3f})")
        return label, True, best_score

    def merge_labels(self, keep_label: str, merge_label: str) -> bool:
        """Merge two labels if they represent the same person."""
        if keep_label not in self.samples or merge_label not in self.samples:
            return False

        keep_sample = self.samples[keep_label]
        merge_sample = self.samples[merge_label]

        # Add all vectors from merge to keep
        keep_sample.vectors.extend(merge_sample.vectors)
        keep_sample.quality_scores.extend(merge_sample.quality_scores)

        # Trim to max history (keep highest quality)
        if len(keep_sample.vectors) > self.max_history:
            # Sort by quality and keep best
            paired = list(zip(keep_sample.vectors, keep_sample.quality_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            keep_sample.vectors = [p[0] for p in paired[:self.max_history]]
            keep_sample.quality_scores = [p[1] for p in paired[:self.max_history]]

        # Recalculate centroid
        self._update_centroid(keep_sample)

        # Remove merged label
        del self.samples[merge_label]
        logger.info(f"Merged {merge_label} into {keep_label}")
        return True

    def _maybe_save(self, image: np.ndarray | None, save_dir: Path | None, label: str) -> str | None:
        if image is None or save_dir is None:
            return None
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"{label}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return str(filename)
        except Exception as e:
            logger.warning(f"Failed to save image for {label}: {e}")
            return None
