from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_model(model) -> None:
    model.eval()
    model.requires_grad_(False)


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported feature dtype: {name}")


def inspect_clip_projection(model) -> Dict[str, object]:
    base = unwrap_model(model).base_model
    visual = base.visual
    info = {
        "visual_projection_name": "base_model.visual.proj" if hasattr(visual, "proj") else None,
        "text_projection_name": "base_model.text_projection" if hasattr(base, "text_projection") else None,
        "visual_projection_shape": None,
        "text_projection_shape": None,
        "patch_grid": infer_patch_grid(model),
    }
    if hasattr(visual, "proj") and visual.proj is not None:
        info["visual_projection_shape"] = list(visual.proj.shape)
    if hasattr(base, "text_projection") and base.text_projection is not None:
        info["text_projection_shape"] = list(base.text_projection.shape)
    return info


def infer_patch_grid(model) -> Tuple[Optional[int], Optional[int]]:
    visual = unwrap_model(model).base_model.visual
    if hasattr(visual, "num_y") and hasattr(visual, "num_x"):
        return int(visual.num_y), int(visual.num_x)

    if hasattr(visual, "positional_embedding"):
        num_tokens = int(visual.positional_embedding.shape[0])
        num_patches = num_tokens - 1
        side = int(num_patches ** 0.5)
        if side * side == num_patches:
            return side, side

    return None, None


@dataclass
class GlobalFeatureSet:
    qfeats: torch.Tensor
    gfeats: torch.Tensor
    qids: torch.Tensor
    gids: torch.Tensor
    captions: List[str]


@dataclass
class HiddenFeatureSet:
    text_features: torch.Tensor
    text_mask: torch.Tensor
    token_ids: torch.Tensor
    image_features: torch.Tensor
    patch_grid: Tuple[Optional[int], Optional[int]]
    feature_dtype: str


class GlobalHiddenExtractor:
    def __init__(self, model, device: torch.device, feature_dtype: str = "float16"):
        self.model = unwrap_model(model)
        self.device = device
        self.feature_dtype = dtype_from_name(feature_dtype)
        self.feature_dtype_name = feature_dtype

    def extract_global_features(self, txt_loader, img_loader) -> GlobalFeatureSet:
        qids, gids, qfeats, gfeats = [], [], [], []
        self.model.eval()

        with torch.inference_mode():
            for pid, caption in txt_loader:
                caption = caption.to(self.device)
                text_feat = self.model.encode_text(caption).detach().cpu()
                qids.append(pid.view(-1).cpu())
                qfeats.append(text_feat.float())

            for pid, image in img_loader:
                image = image.to(self.device)
                image_feat = self.model.encode_image(image).detach().cpu()
                gids.append(pid.view(-1).cpu())
                gfeats.append(image_feat.float())

        captions = []
        txt_set = getattr(txt_loader, "test_txt_set", None)
        if txt_set is not None and hasattr(txt_set, "captions"):
            captions = list(txt_set.captions)
        else:
            captions = [""] * sum(x.numel() for x in qids)

        return GlobalFeatureSet(
            qfeats=torch.cat(qfeats, dim=0),
            gfeats=torch.cat(gfeats, dim=0),
            qids=torch.cat(qids, dim=0).long(),
            gids=torch.cat(gids, dim=0).long(),
            captions=captions,
        )

    def extract_hidden_features(self, txt_loader, img_loader) -> HiddenFeatureSet:
        text_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        token_chunks: List[torch.Tensor] = []
        image_chunks: List[torch.Tensor] = []
        base = self.model.base_model
        self.model.eval()

        with torch.inference_mode():
            for _, caption in txt_loader:
                caption = caption.to(self.device)
                token_seq, _ = base.encode_text(caption.long())
                token_seq = F.normalize(token_seq.float(), p=2, dim=-1)

                caption_cpu = caption.detach().cpu()
                eot_positions = caption_cpu.argmax(dim=-1)
                valid = caption_cpu.ne(0)
                if valid.shape[1] > 0:
                    valid[:, 0] = False
                for row, eot_pos in enumerate(eot_positions.tolist()):
                    valid[row, eot_pos] = False

                text_chunks.append(token_seq.detach().cpu().to(self.feature_dtype))
                mask_chunks.append(valid.cpu())
                token_chunks.append(caption_cpu.long())

            for _, image in img_loader:
                image = image.to(self.device)
                image_seq, _ = base.encode_image(image)
                patch_seq = image_seq[:, 1:, :]
                patch_seq = F.normalize(patch_seq.float(), p=2, dim=-1)
                image_chunks.append(patch_seq.detach().cpu().to(self.feature_dtype))

        text_full = torch.cat(text_chunks, dim=0)
        mask_full = torch.cat(mask_chunks, dim=0)
        token_full = torch.cat(token_chunks, dim=0)
        text_features, text_mask, token_ids = self._compact_text_tokens(
            text_full, mask_full, token_full
        )

        return HiddenFeatureSet(
            text_features=text_features,
            text_mask=text_mask,
            token_ids=token_ids,
            image_features=torch.cat(image_chunks, dim=0),
            patch_grid=infer_patch_grid(self.model),
            feature_dtype=self.feature_dtype_name,
        )

    def _compact_text_tokens(
        self,
        text_full: torch.Tensor,
        mask_full: torch.Tensor,
        token_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = mask_full.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        max_len = max(max_len, 1)
        dim = int(text_full.shape[-1])

        compact = torch.zeros(
            (text_full.shape[0], max_len, dim),
            dtype=self.feature_dtype,
        )
        compact_mask = torch.zeros((text_full.shape[0], max_len), dtype=torch.bool)
        compact_tokens = torch.zeros((text_full.shape[0], max_len), dtype=torch.long)

        for i in range(text_full.shape[0]):
            keep = mask_full[i]
            length = int(keep.sum().item())
            if length == 0:
                continue
            compact[i, :length] = text_full[i, keep]
            compact_mask[i, :length] = True
            compact_tokens[i, :length] = token_full[i, keep]

        return compact, compact_mask, compact_tokens

