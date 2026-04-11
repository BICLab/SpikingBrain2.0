from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .modeling_sse_swa_moba import SseSwaMobaForCausalLM


class SseSwaMobaVLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """Initial vLLM multimodal adapter for SSE-SWA-MoBA VL.

    This first version focuses on:
    - visual encoder + text decoder wiring
    - multimodal embedding merge path in vLLM
    - HF -> vLLM weight prefix remapping for common SPB2-VL checkpoints
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # HF top-level VL model prefixes
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.language_model = SseSwaMobaForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> dict[str, Any] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` is required for image inputs.")

        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> dict[str, Any] | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None
        if video_grid_thw is None:
            raise ValueError("`video_grid_thw` is required for video inputs.")

        if pixel_values_videos is not None:
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> list[tuple[str, dict[str, Any]]]:
        mm_inputs: list[tuple[str, dict[str, Any]]] = []
        has_image = False
        has_video = False

        # Preserve kwargs order to stay aligned with placeholder order as much
        # as possible in an initial adapter.
        for key in kwargs:
            if key in ("pixel_values", "image_embeds") and not has_image:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_inputs.append(("image", image_input))
                has_image = True
            if key in ("pixel_values_videos", "video_embeds") and not has_video:
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_inputs.append(("video", video_input))
                has_video = True

        return mm_inputs

    def _process_image_input(
        self, image_input: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def embed_multimodal(self, **kwargs: object):
        mm_inputs = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_inputs:
            return None

        multimodal_embeddings: list[torch.Tensor] = []
        for modality, mm_input in mm_inputs:
            if modality == "image":
                multimodal_embeddings.extend(self._process_image_input(mm_input))
            elif modality == "video":
                multimodal_embeddings.extend(self._process_video_input(mm_input))

        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=True,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # `kwargs` reserved for future multimodal runtime metadata.
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
