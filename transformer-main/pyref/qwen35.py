import json
import math
from dataclasses import dataclass
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F
from torch.nn.modules import conv


@dataclass
class Qwen35Config:
    # Token latent space representation size
    hidden_size: int
    # Number of decoder layers
    num_layers: int
    # "linear_attention" or "full_attention" for each layer
    layer_types: list[str]
    # Intermediate dimension of the post-attention feedforward network
    intermediate_size: int
    # If the input embedding matrix is the same as the output logit matrix
    embedding_tying: bool

    # Full attention dimensions
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    rope_theta: float
    rotary_dim: int

    # Gated DeltaNet dimensions
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    # Normalization
    rms_norm_eps: float

    def queries_size(self) -> int:
        return self.num_query_heads * self.head_size

    def keys_size(self) -> int:
        return self.num_kv_heads * self.head_size

    def values_size(self) -> int:
        return self.num_kv_heads * self.head_size

    def linear_keys_size(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    def linear_values_size(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    def linear_conv_size(self) -> int:
        return self.linear_keys_size() * 2 + self.linear_values_size()


@dataclass
class Qwen35Cache:
    # Full attention layers use a normal KV cache.
    keys: torch.Tensor
    values: torch.Tensor
    # Linear attention layers use a short convolution state and recurrent state.
    conv_states: torch.Tensor
    recurrent_states: torch.Tensor
    seq_len: int = 0


def zero_centered_rms_norm(
    weight: torch.Tensor, hidden_state: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """RMSNorm where the learned scale is 1 + weight, as in Qwen3-Next."""
    variance = hidden_state.float().pow(2).mean(dim=-1, keepdim=True)
    hidden_state = hidden_state.float() * torch.rsqrt(variance + eps)
    return (hidden_state * (1.0 + weight.float())).to(weight.dtype)


def gated_rms_norm(
    weight: torch.Tensor,
    hidden_state: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    variance = hidden_state.float().pow(2).mean(dim=-1, keepdim=True)
    hidden_state = hidden_state.float() * torch.rsqrt(variance + eps)
    hidden_state = hidden_state.to(weight.dtype) * weight
    return hidden_state * silu(gate.float()).to(hidden_state.dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm.to(x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid linear unit"""
    return x / (1 + torch.exp(-x))


def linear(
    weight: torch.Tensor, hidden_state: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = weight @ hidden_state
    if bias is not None:
        output += bias
    return output


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_to_qk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    position_idx: int,
    theta_base: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = min(rotary_dim, queries.shape[-1], keys.shape[-1])
    if rotary_dim == 0:
        return queries, keys

    thetas = 1.0 / (
        theta_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=queries.device)
            / rotary_dim
        )
    )
    angles = position_idx * thetas
    cos = torch.concat([angles.cos(), angles.cos()]).to(queries.dtype)
    sin = torch.concat([angles.sin(), angles.sin()]).to(queries.dtype)

    query_rope = (queries[..., :rotary_dim] * cos) + (
        rotate_half(queries[..., :rotary_dim]) * sin
    )
    key_rope = (keys[..., :rotary_dim] * cos) + (
        rotate_half(keys[..., :rotary_dim]) * sin
    )
    return (
        torch.cat((query_rope, queries[..., rotary_dim:]), dim=-1),
        torch.cat((key_rope, keys[..., rotary_dim:]), dim=-1),
    )


class TensorLoader:
    """Small helper so the reference can load single-file or sharded HF checkpoints."""

    def __init__(self, model_dir: Path):
        self.tensor_to_file: dict[str, Path] = {}
        for file in sorted(model_dir.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt") as tensors:
                for key in tensors.keys():
                    self.tensor_to_file[key] = file
        if not self.tensor_to_file:
            raise FileNotFoundError(f"no safetensors files found in {model_dir}")

    def get_tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        with safetensors.safe_open(
            self.tensor_to_file[name], framework="pt"
        ) as tensors:
            return tensors.get_tensor(name).to(dtype)

    def get_optional_tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor | None:
        if name not in self.tensor_to_file:
            return None
        return self.get_tensor(name, dtype)

    def get_concat_tensor(
        self, names: list[str], dtype: torch.dtype, dim: int = 0
    ) -> torch.Tensor:
        return torch.cat([self.get_tensor(name, dtype) for name in names], dim=dim)


@dataclass
class Qwen35FullAttnLayer:
    config: Qwen35Config
    layer_num: int

    input_layernorm: torch.Tensor  # (hidden_size, )
    post_attention_layernorm: torch.Tensor  # (hidden_size, )

    q_proj_weight: torch.Tensor
    q_proj_bias: torch.Tensor
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor
    o_proj_weight: torch.Tensor
    o_proj_bias: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor

    up_proj_weight: torch.Tensor
    gate_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor

    def forward(
        self,
        cache: Qwen35Cache,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = cache.seq_len

        # pre-attention norm.
        hidden_state_normed = zero_centered_rms_norm(
            self.input_layernorm, hidden_state, self.config.rms_norm_eps
        )

        # compute queries + gate for current token.
        # huggingface implementation merges the query and gate projection matrices for some reason:
        # first half is queries, second half is gates.
        queries, gate = (
            linear(self.q_proj_weight, hidden_state_normed, self.q_proj_bias)
            .view((self.config.num_query_heads, 2 * self.config.head_size))
            .chunk(2, dim=-1)
        )
        queries = zero_centered_rms_norm(
            self.q_norm_weight, queries, self.config.rms_norm_eps
        )
        gate = gate.reshape((self.config.queries_size(),))

        # compute key vectors for current token.
        new_keys = linear(
            self.k_proj_weight, hidden_state_normed, self.k_proj_bias
        ).view((self.config.num_kv_heads, self.config.head_size))
        new_keys = zero_centered_rms_norm(
            self.k_norm_weight, new_keys, self.config.rms_norm_eps
        )
        queries, new_keys = apply_rope_to_qk(
            queries, new_keys, seq_len, self.config.rope_theta, self.config.rotary_dim
        )
        cache.keys[seq_len, self.layer_num] = new_keys

        # compute new values for current token.
        new_values = linear(
            self.v_proj_weight, hidden_state_normed, self.v_proj_bias
        ).view((self.config.num_kv_heads, self.config.head_size))
        cache.values[seq_len, self.layer_num] = new_values

        # repeat each key/value in cache for group query
        # (T, Hkv, D) -> (Hq, T, D)
        group_size = self.config.num_query_heads // self.config.num_kv_heads
        keys = (
            cache.keys[: seq_len + 1, self.layer_num]
            .repeat_interleave(group_size, dim=1)
            .permute(1, 0, 2)
        )
        values = (
            cache.values[: seq_len + 1, self.layer_num]
            .repeat_interleave(group_size, dim=1)
            .permute(1, 0, 2)
        )

        # GQA computation.
        # matmul queries by keys transposed for each head, sum over embedding dim.
        scores = torch.einsum("hd,htd->ht", queries, keys) / math.sqrt(
            self.config.head_size
        )
        attention = F.softmax(scores, dim=-1)
        # matmul attention by values for each head, sum over tokens.
        weighted_values = torch.einsum("ht,htd->hd", attention, values)
        # force back into shape of queries and gate the output.
        weighted_values = weighted_values.view((self.config.queries_size(),))
        weighted_values *= torch.sigmoid(gate)
        attention_output = linear(self.o_proj_weight, weighted_values, self.o_proj_bias)

        # residual connection.
        hidden_state += attention_output

        # FFN
        ffn_input = zero_centered_rms_norm(
            self.post_attention_layernorm, hidden_state, self.config.rms_norm_eps
        )
        ffn_output = self.down_proj_weight @ (
            silu(self.gate_proj_weight @ ffn_input) * (self.up_proj_weight @ ffn_input)
        )

        # residual connection.
        hidden_state += ffn_output

        return hidden_state


@dataclass
class Qwen35LinearAttentionLayer:
    config: Qwen35Config
    layer_num: int

    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor

    in_proj_qkv_weight: torch.Tensor
    in_proj_z_weight: torch.Tensor
    in_proj_b_weight: torch.Tensor
    in_proj_a_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    conv1d_bias: torch.Tensor
    dt_bias: torch.Tensor
    A_log: torch.Tensor
    norm_weight: torch.Tensor
    out_proj_weight: torch.Tensor
    out_proj_bias: torch.Tensor

    up_proj_weight: torch.Tensor
    gate_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor

    def forward(
        self,
        cache: Qwen35Cache,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        # pre-attention norm.
        hidden_state_normed = zero_centered_rms_norm(
            self.input_layernorm, hidden_state, self.config.rms_norm_eps
        )

        # for linear attention, there are multiple value heads per qk instead of multiple q per kv.
        group_size = (
            self.config.linear_num_value_heads // self.config.linear_num_key_heads
        )

        # compute linear projections of input.
        # qkv projection is merged in single matrix.
        qkv = self.in_proj_qkv_weight @ hidden_state_normed
        gates = self.in_proj_z_weight @ hidden_state_normed
        beta_raw = self.in_proj_b_weight @ hidden_state_normed
        decay_raw = self.in_proj_a_weight @ hidden_state_normed

        # split qkv into individual tensors.
        queries, keys, values = torch.split(
            qkv,
            [
                self.config.linear_keys_size(),
                self.config.linear_keys_size(),
                self.config.linear_values_size(),
            ],
        )
        queries = queries.view(
            (self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        )
        keys = keys.view(
            (self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        )
        values = values.reshape(
            (self.config.linear_num_value_heads, self.config.linear_value_head_dim)
        )
        gates = gates.reshape(
            (self.config.linear_num_value_heads, self.config.linear_value_head_dim)
        )

        beta = torch.sigmoid(beta_raw.reshape((self.config.linear_num_value_heads,)))
        decay = -self.A_log.float().exp() * F.softplus(
            decay_raw.reshape((self.config.linear_num_value_heads,)).float()
            + self.dt_bias.float()
        )

        # convolution over rolling window of past states.
        # first flatten and concatenate qkv matrices for grouped convolution.
        mixed_qkv = torch.cat(
            (queries.reshape(-1), keys.reshape(-1), values.reshape(-1))
        )
        conv_state = cache.conv_states[self.layer_num]
        # pop earliest state and replace with new qkv.
        conv_state[:-1] = conv_state[1:].clone()
        conv_state[-1] = mixed_qkv
        # actual convolution.
        conv_weight = self.conv1d_weight.squeeze(1)
        mixed_qkv = (conv_state.T * conv_weight).sum(dim=-1)
        if self.conv1d_bias is not None:
            mixed_qkv += self.conv1d_bias
        mixed_qkv = silu(mixed_qkv)

        # unflatten mixed qkv back into separate matrices
        queries, keys, values = torch.split(
            mixed_qkv,
            [
                self.config.linear_keys_size(),
                self.config.linear_keys_size(),
                self.config.linear_values_size(),
            ],
            dim=-1,
        )
        queries = queries.view(
            (self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        )
        keys = keys.view(
            (self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        )
        values = values.view(
            (self.config.linear_num_value_heads, self.config.linear_value_head_dim)
        )

        # repeat queries and keys by group size to match the number of value heads.
        if group_size > 1:
            queries = queries.repeat_interleave(group_size, dim=0)
            keys = keys.repeat_interleave(group_size, dim=0)
        # qk norm
        queries = l2norm(queries) / math.sqrt(self.config.linear_key_head_dim)
        keys = l2norm(keys)

        # delta net recurrent update:
        # state = decay * state + key @ (beta * (value - state @ key)).T
        state = cache.recurrent_states[self.layer_num]
        # decay is of shape (H,) so we broadcast coeffs to be (H, 1, 1) for matmul.
        decay_coeffs = decay.exp().to(state.dtype).view((-1, 1, 1))
        state *= decay_coeffs
        # use current state (memory) to try and predict what values should be for each head.
        predicted_values = torch.einsum("hkv,hk->hv", state, keys)
        # use the delta between true values and predicted values to adjust state (memory).
        # multiply by memory overwrite factor beta, broadcasted for same reasons as decay.
        deltas = (values - predicted_values) * beta.view((-1, 1))
        # use outer product of keys and deltas to update state.
        # intuitively, kv-correlations (state/memory) so that it would've been
        # correct for new mixture of keys and values.
        state += torch.einsum("hk,hv->hkv", keys, deltas)
        # get full attention (q @ K.T) @ V equivalent from q and kv-correlations (state).
        weighted_values = torch.einsum("hkv,hk->hv", state, queries)

        # gated attention.
        gated_weighted_values = gated_rms_norm(
            self.norm_weight, weighted_values, gates, self.config.rms_norm_eps
        )
        attention_output = linear(
            self.out_proj_weight,
            gated_weighted_values.view((self.config.linear_values_size(),)),
            self.out_proj_bias,
        )

        # residual connection.
        hidden_state += attention_output

        # pre-FFN layernorm
        ffn_input = zero_centered_rms_norm(
            self.post_attention_layernorm, hidden_state, self.config.rms_norm_eps
        )
        # FFN
        ffn_output = self.down_proj_weight @ (
            silu(self.gate_proj_weight @ ffn_input) * (self.up_proj_weight @ ffn_input)
        )

        # residual connection
        hidden_state += ffn_output

        return hidden_state


class Qwen35Model:
    config: Qwen35Config
    # embedding weight: (vocab_size, hidden_size)
    embedding_weight: torch.Tensor
    layers: list[Qwen35FullAttnLayer | Qwen35LinearAttentionLayer]
    # final layernorm: (hidden_size)
    final_layernorm: torch.Tensor
    # output logit weight: (vocab_size, hidden_size)
    lm_head_weight: torch.Tensor

    def __init__(self, model_dir: Path, dtype: torch.dtype = torch.float32):
        with open(model_dir / "config.json", "r") as f:
            config_obj = json.load(f)
        config_obj = config_obj.get("text_config", config_obj)

        head_size = config_obj.get(
            "head_dim", config_obj["hidden_size"] // config_obj["num_attention_heads"]
        )
        rope_params = config_obj.get("rope_parameters", {})
        rotary_dim = int(head_size * rope_params.get("partial_rotary_factor", 1.0))
        layer_types = config_obj.get("layer_types")
        if layer_types is None:
            full_attention_interval = config_obj.get("full_attention_interval", 4)
            layer_types = [
                "full_attention"
                if (i + 1) % full_attention_interval == 0
                else "linear_attention"
                for i in range(config_obj["num_hidden_layers"])
            ]

        self.config = Qwen35Config(
            hidden_size=config_obj["hidden_size"],
            num_layers=config_obj["num_hidden_layers"],
            layer_types=layer_types,
            intermediate_size=config_obj["intermediate_size"],
            embedding_tying=config_obj.get("tie_word_embeddings", False),
            num_query_heads=config_obj["num_attention_heads"],
            num_kv_heads=config_obj["num_key_value_heads"],
            head_size=head_size,
            rope_theta=rope_params.get("rope_theta", config_obj.get("rope_theta", 1e6)),
            rotary_dim=rotary_dim,
            linear_num_key_heads=config_obj.get("linear_num_key_heads", 16),
            linear_num_value_heads=config_obj.get("linear_num_value_heads", 32),
            linear_key_head_dim=config_obj.get("linear_key_head_dim", 128),
            linear_value_head_dim=config_obj.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=config_obj.get("linear_conv_kernel_dim", 4),
            rms_norm_eps=config_obj.get("rms_norm_eps", 1e-6),
        )

        tensors = TensorLoader(model_dir)
        tensor_prefix = (
            "model.language_model"
            if "model.language_model.embed_tokens.weight" in tensors.tensor_to_file
            else "model"
        )
        self.embedding_weight = tensors.get_tensor(
            f"{tensor_prefix}.embed_tokens.weight", dtype
        )
        self.layers = []
        for i in range(self.config.num_layers):
            layer_type = self.config.layer_types[i]
            prefix = f"{tensor_prefix}.layers.{i}"

            input_layernorm = tensors.get_tensor(
                f"{prefix}.input_layernorm.weight", dtype
            )
            post_attention_layernorm = tensors.get_tensor(
                f"{prefix}.post_attention_layernorm.weight", dtype
            )
            up_proj_weight = tensors.get_tensor(f"{prefix}.mlp.up_proj.weight", dtype)
            gate_proj_weight = tensors.get_tensor(
                f"{prefix}.mlp.gate_proj.weight", dtype
            )
            down_proj_weight = tensors.get_tensor(
                f"{prefix}.mlp.down_proj.weight", dtype
            )

            if layer_type == "full_attention":
                self.layers.append(
                    Qwen35FullAttnLayer(
                        config=self.config,
                        layer_num=i,
                        input_layernorm=input_layernorm,
                        post_attention_layernorm=post_attention_layernorm,
                        q_proj_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.q_proj.weight", dtype
                        ),
                        q_proj_bias=tensors.get_optional_tensor(
                            f"{prefix}.self_attn.q_proj.bias", dtype
                        ),
                        k_proj_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.k_proj.weight", dtype
                        ),
                        k_proj_bias=tensors.get_optional_tensor(
                            f"{prefix}.self_attn.k_proj.bias", dtype
                        ),
                        v_proj_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.v_proj.weight", dtype
                        ),
                        v_proj_bias=tensors.get_optional_tensor(
                            f"{prefix}.self_attn.v_proj.bias", dtype
                        ),
                        o_proj_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.o_proj.weight", dtype
                        ),
                        o_proj_bias=tensors.get_optional_tensor(
                            f"{prefix}.self_attn.o_proj.bias", dtype
                        ),
                        q_norm_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.q_norm.weight", dtype
                        ),
                        k_norm_weight=tensors.get_tensor(
                            f"{prefix}.self_attn.k_norm.weight", dtype
                        ),
                        up_proj_weight=up_proj_weight,
                        gate_proj_weight=gate_proj_weight,
                        down_proj_weight=down_proj_weight,
                    )
                )
            elif layer_type == "linear_attention":
                if (
                    f"{prefix}.linear_attn.in_proj_qkvz.weight"
                    in tensors.tensor_to_file
                ):
                    in_proj_qkvz_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_qkvz.weight", dtype
                    )
                    qkv_size = (
                        self.config.linear_keys_size() * 2
                        + self.config.linear_values_size()
                    )
                    in_proj_qkv_weight = in_proj_qkvz_weight[:qkv_size]
                    in_proj_z_weight = in_proj_qkvz_weight[qkv_size:]
                else:
                    in_proj_qkv_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_qkv.weight", dtype
                    )
                    in_proj_z_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_z.weight", dtype
                    )

                if f"{prefix}.linear_attn.in_proj_ba.weight" in tensors.tensor_to_file:
                    in_proj_ba_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_ba.weight", dtype
                    )
                    split = self.config.linear_num_value_heads
                    in_proj_b_weight = in_proj_ba_weight[:split]
                    in_proj_a_weight = in_proj_ba_weight[split:]
                else:
                    in_proj_b_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_b.weight", dtype
                    )
                    in_proj_a_weight = tensors.get_tensor(
                        f"{prefix}.linear_attn.in_proj_a.weight", dtype
                    )

                self.layers.append(
                    Qwen35LinearAttentionLayer(
                        config=self.config,
                        layer_num=i,
                        input_layernorm=input_layernorm,
                        post_attention_layernorm=post_attention_layernorm,
                        in_proj_qkv_weight=in_proj_qkv_weight,
                        in_proj_z_weight=in_proj_z_weight,
                        in_proj_b_weight=in_proj_b_weight,
                        in_proj_a_weight=in_proj_a_weight,
                        conv1d_weight=tensors.get_tensor(
                            f"{prefix}.linear_attn.conv1d.weight", dtype
                        ),
                        conv1d_bias=tensors.get_optional_tensor(
                            f"{prefix}.linear_attn.conv1d.bias", dtype
                        ),
                        dt_bias=tensors.get_tensor(
                            f"{prefix}.linear_attn.dt_bias", dtype
                        ),
                        A_log=tensors.get_tensor(f"{prefix}.linear_attn.A_log", dtype),
                        norm_weight=tensors.get_tensor(
                            f"{prefix}.linear_attn.norm.weight", dtype
                        ),
                        out_proj_weight=tensors.get_tensor(
                            f"{prefix}.linear_attn.out_proj.weight", dtype
                        ),
                        out_proj_bias=tensors.get_optional_tensor(
                            f"{prefix}.linear_attn.out_proj.bias", dtype
                        ),
                        up_proj_weight=up_proj_weight,
                        gate_proj_weight=gate_proj_weight,
                        down_proj_weight=down_proj_weight,
                    )
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r}")

        self.final_layernorm = tensors.get_tensor(f"{tensor_prefix}.norm.weight", dtype)
        if (
            self.config.embedding_tying
            or "lm_head.weight" not in tensors.tensor_to_file
        ):
            self.lm_head_weight = self.embedding_weight
        else:
            self.lm_head_weight = tensors.get_tensor("lm_head.weight", dtype)

    def allocate_cache(
        self,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Qwen35Cache:
        dtype = dtype or self.embedding_weight.dtype
        device = device or self.embedding_weight.device
        return Qwen35Cache(
            keys=torch.zeros(
                (
                    max_seq_len,
                    self.config.num_layers,
                    self.config.num_kv_heads,
                    self.config.head_size,
                ),
                dtype=dtype,
                device=device,
            ),
            values=torch.zeros(
                (
                    max_seq_len,
                    self.config.num_layers,
                    self.config.num_kv_heads,
                    self.config.head_size,
                ),
                dtype=dtype,
                device=device,
            ),
            conv_states=torch.zeros(
                (
                    self.config.num_layers,
                    self.config.linear_conv_kernel_dim,
                    self.config.linear_conv_size(),
                ),
                dtype=dtype,
                device=device,
            ),
            recurrent_states=torch.zeros(
                (
                    self.config.num_layers,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                ),
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, cache: Qwen35Cache, input_tok_id: int, temperature: float) -> int:
        """
        cache: persistent KV/recurrent state for autoregressive decoding.
        input_tok_id: last token in the sequence
        temperature: sampling parameter, deterministic=0
        """
        if cache.seq_len >= cache.keys.shape[0]:
            raise ValueError("cache is full")

        # take initial hidden state from embedding table
        hidden_state = self.embedding_weight[input_tok_id].squeeze().detach().clone()

        for layer in self.layers:
            hidden_state = layer.forward(cache, hidden_state)

        # final layernorm
        hidden_state = zero_centered_rms_norm(
            self.final_layernorm, hidden_state, self.config.rms_norm_eps
        )

        output_scores = hidden_state @ self.lm_head_weight.T
        if temperature == 0:
            # softmax is monotonic, so we can sample the largest value directly
            new_token = torch.argmax(output_scores)
        else:
            probs = torch.softmax(output_scores.float() / temperature, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1).squeeze()

        cache.seq_len += 1
        return int(new_token)
