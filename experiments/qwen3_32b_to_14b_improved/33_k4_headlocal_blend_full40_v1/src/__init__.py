"""Train-only k4 + matched-head affine blend candidate."""

from .blend import (
    ProtocolError,
    canonical_sha256,
    file_ref,
    fit_convex_alpha,
    fit_fold_alpha,
    dense_from_head_local,
    embed_k4_dense,
    fuse_dense_affines,
    validate_selection_subset,
    source_batch_to_position_layer_major,
)

__all__ = [
    "ProtocolError",
    "canonical_sha256",
    "file_ref",
    "fit_convex_alpha",
    "fit_fold_alpha",
    "dense_from_head_local",
    "embed_k4_dense",
    "fuse_dense_affines",
    "validate_selection_subset",
    "source_batch_to_position_layer_major",
]
