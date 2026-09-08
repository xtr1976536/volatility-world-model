"""Anchor-conditioned market/asset recurrent state-space model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def _normal_params(raw: Tensor, min_scale: float = 0.05) -> tuple[Tensor, Tensor]:
    mu, raw_scale = raw.chunk(2, dim=-1)
    scale = F.softplus(raw_scale) + min_scale
    return mu, scale


def _sample(mu: Tensor, scale: Tensor, deterministic: bool = False) -> Tensor:
    return mu if deterministic else mu + scale * torch.randn_like(mu)


def _gaussian_kl(q_mu: Tensor, q_sd: Tensor, p_mu: Tensor, p_sd: Tensor) -> Tensor:
    q_var = q_sd.square()
    p_var = p_sd.square()
    return 0.5 * ((q_var + (q_mu - p_mu).square()) / p_var - 1.0 + 2.0 * (torch.log(p_sd) - torch.log(q_sd)))


def _low_rank_gaussian_nll(target: Tensor, mean: Tensor, scale: Tensor, loadings: Tensor) -> Tensor:
    """Negative log likelihood under ``BB' + diag(scale^2)`` covariance.

    The matrix determinant lemma and Woodbury identity keep this O(N q^2)
    rather than repeatedly factoring an N-by-N covariance matrix.
    """
    residual = target - mean
    inverse_diag = scale.square().reciprocal()
    if loadings.shape[-1] == 0:
        return 0.5 * (residual.square().mul(inverse_diag).sum(dim=-1)
                      + 2.0 * torch.log(scale).sum(dim=-1)
                      + target.shape[-1] * torch.log(torch.as_tensor(2.0 * torch.pi, device=target.device, dtype=target.dtype)))
    weighted_loading = loadings * inverse_diag.unsqueeze(-1)
    eye = torch.eye(loadings.shape[-1], device=target.device, dtype=target.dtype)
    middle = eye.unsqueeze(0) + torch.matmul(loadings.transpose(1, 2), weighted_loading)
    projected = torch.einsum("bn,bnq->bq", residual * inverse_diag, loadings)
    correction = torch.linalg.solve(middle, projected.unsqueeze(-1)).squeeze(-1)
    quadratic = (residual.square() * inverse_diag).sum(dim=-1) - (projected * correction).sum(dim=-1)
    logdet = (2.0 * torch.log(scale)).sum(dim=-1) + torch.linalg.slogdet(middle).logabsdet
    return 0.5 * (quadratic + logdet + target.shape[-1] * torch.log(torch.as_tensor(2.0 * torch.pi, device=target.device)))


def market_features(observation: Tensor) -> Tensor:
    """Cross-sectional summary used by the shared market state.

    Input is ``[batch, time, asset, feature]`` with feature order log-RV,
    log-IV, clipped return.  Output is ``[batch, time, 6]``.
    """
    return torch.cat([observation.mean(dim=2), observation.std(dim=2, unbiased=False)], dim=-1)


@dataclass
class RSSMConfig:
    n_assets: int
    feature_dim: int = 3
    market_deterministic_dim: int = 96
    market_stochastic_dim: int = 24
    asset_deterministic_dim: int = 96
    asset_stochastic_dim: int = 24
    asset_embedding_dim: int = 24
    hidden_dim: int = 128
    consistency_weight: float = 0.05
    kl_weight: float = 0.25
    free_nats: float = 1.0
    rollout_gamma: float = 0.9
    min_scale: float = 0.05
    public_shock_dim: int = 2
    public_shock_scale: float = 0.05
    observation_loss_weight: float = 0.25


class DualStateRSSM(nn.Module):
    """A passive two-level RSSM for anchor-relative volatility innovations.

    The market state is shared across all assets.  Asset states receive the
    shared market deterministic state and their own observation embedding,
    preserving a common transition with heterogeneous responses.
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        n, f = cfg.n_assets, cfg.feature_dim
        md, ms, ad, ass, ae, hd = (cfg.market_deterministic_dim, cfg.market_stochastic_dim,
                                   cfg.asset_deterministic_dim, cfg.asset_stochastic_dim,
                                   cfg.asset_embedding_dim, cfg.hidden_dim)
        self.n_assets = n
        self.asset_embedding = nn.Embedding(n, ae)
        self.market_gru = nn.GRUCell(ms + 6, md)
        self.market_obs_encoder = _mlp(6, hd, hd)
        self.market_prior = _mlp(md, hd, 2 * ms)
        self.market_post = _mlp(md + hd, hd, 2 * ms)
        self.asset_gru = nn.GRUCell(ass + f + ae + md, ad)
        self.asset_prior = _mlp(ad + md, hd, 2 * ass)
        self.asset_obs_encoder = _mlp(f + md, hd, hd)
        self.asset_post = _mlp(ad + hd + md, hd, 2 * ass)
        state_dim = md + ms + ad + ass + ae
        self.innovation_decoder = _mlp(state_dim, hd, 2)
        # Standard world-model observation head: prior states must also
        # explain the future observed feature vector, not only the RV residual.
        self.observation_decoder = _mlp(state_dim, hd, 2 * f)
        self.public_loading_decoder = _mlp(state_dim, hd, cfg.public_shock_dim)
        self.gate_decoder = nn.Sequential(nn.Linear(md + ms, hd), nn.SiLU(), nn.Linear(hd, 1))

    def _asset_ids(self, batch: int, device: torch.device) -> Tensor:
        return torch.arange(self.n_assets, device=device).view(1, self.n_assets).expand(batch, -1)

    def _context_step(self, state: Dict[str, Tensor], obs: Tensor, deterministic: bool) -> Dict[str, Tensor]:
        hm, zm, ha, za = state["hm"], state["zm"], state["ha"], state["za"]
        b = obs.shape[0]
        mf = market_features(obs[:, None]).squeeze(1)
        hm = self.market_gru(torch.cat([zm, mf], dim=-1), hm)
        m_emb = self.market_obs_encoder(mf)
        q_m_mu, q_m_sd = _normal_params(self.market_post(torch.cat([hm, m_emb], dim=-1)), self.cfg.min_scale)
        zm = _sample(q_m_mu, q_m_sd, deterministic)
        ids = self._asset_ids(b, obs.device)
        emb = self.asset_embedding(ids)
        hm_a = hm[:, None, :].expand(-1, self.n_assets, -1)
        asset_input = torch.cat([za, obs, emb, hm_a], dim=-1).reshape(b * self.n_assets, -1)
        ha = self.asset_gru(asset_input, ha.reshape(b * self.n_assets, -1)).reshape(b, self.n_assets, -1)
        asset_emb = self.asset_obs_encoder(torch.cat([obs, hm_a], dim=-1))
        q_a_mu, q_a_sd = _normal_params(self.asset_post(torch.cat([ha, asset_emb, hm_a], dim=-1)), self.cfg.min_scale)
        za = _sample(q_a_mu, q_a_sd, deterministic)
        return {"hm": hm, "zm": zm, "ha": ha, "za": za}

    def observe(self, context: Tensor, deterministic: bool = False) -> Dict[str, Tensor]:
        if context.ndim != 4 or context.shape[2] != self.n_assets:
            raise ValueError("context must have shape [batch,time,asset,feature]")
        b, _, _, _ = context.shape
        device = context.device
        state = {
            "hm": torch.zeros(b, self.cfg.market_deterministic_dim, device=device),
            "zm": torch.zeros(b, self.cfg.market_stochastic_dim, device=device),
            "ha": torch.zeros(b, self.n_assets, self.cfg.asset_deterministic_dim, device=device),
            "za": torch.zeros(b, self.n_assets, self.cfg.asset_stochastic_dim, device=device),
        }
        for t in range(context.shape[1]):
            state = self._context_step(state, context[:, t], deterministic)
        return state

    def _prior_step(self, state: Dict[str, Tensor], deterministic: bool = False) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        hm, zm, ha, za = state["hm"], state["zm"], state["ha"], state["za"]
        b = hm.shape[0]
        zeros = torch.zeros(b, 6, device=hm.device, dtype=hm.dtype)
        hm = self.market_gru(torch.cat([zm, zeros], dim=-1), hm)
        p_m_mu, p_m_sd = _normal_params(self.market_prior(hm), self.cfg.min_scale)
        zm = _sample(p_m_mu, p_m_sd, deterministic)
        ids = self._asset_ids(b, hm.device)
        emb = self.asset_embedding(ids)
        hm_a = hm[:, None, :].expand(-1, self.n_assets, -1)
        future_zeros = torch.zeros(b, self.n_assets, self.cfg.feature_dim, device=hm.device, dtype=hm.dtype)
        asset_input = torch.cat([za, future_zeros, emb, hm_a], dim=-1)
        ha = self.asset_gru(asset_input.reshape(b * self.n_assets, -1), ha.reshape(b * self.n_assets, -1)).reshape(b, self.n_assets, -1)
        p_a_mu, p_a_sd = _normal_params(self.asset_prior(torch.cat([ha, hm_a], dim=-1)), self.cfg.min_scale)
        za = _sample(p_a_mu, p_a_sd, deterministic)
        next_state = {"hm": hm, "zm": zm, "ha": ha, "za": za}
        gate = torch.sigmoid(self.gate_decoder(torch.cat([hm, zm], dim=-1)))
        features = torch.cat([hm_a, zm[:, None, :].expand(-1, self.n_assets, -1), ha, za, emb], dim=-1)
        mu, sd = _normal_params(self.innovation_decoder(features), self.cfg.min_scale)
        loadings = self.public_loading_decoder(features)
        loadings = torch.tanh(loadings) * 0.5
        # The gate routes the innovation mean only.  The scale remains an
        # independent uncertainty estimate and cannot be inflated by a rule.
        mu = gate[:, None, :] * mu
        outputs = {"mu": mu.squeeze(-1), "sd": sd.squeeze(-1), "gate": gate.squeeze(-1),
                   "loadings": loadings,
                   "p_m_mu": p_m_mu, "p_m_sd": p_m_sd,
                   "p_a_mu": p_a_mu, "p_a_sd": p_a_sd}
        return next_state, outputs

    def _decode(self, state: Dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode one joint residual observation from a latent state."""
        hm, zm, ha, za = state["hm"], state["zm"], state["ha"], state["za"]
        batch = hm.shape[0]
        ids = self._asset_ids(batch, hm.device)
        emb = self.asset_embedding(ids)
        hm_a = hm[:, None, :].expand(-1, self.n_assets, -1)
        features = torch.cat([hm_a, zm[:, None, :].expand(-1, self.n_assets, -1), ha, za, emb], dim=-1)
        mu, sd = _normal_params(self.innovation_decoder(features), self.cfg.min_scale)
        gate = torch.sigmoid(self.gate_decoder(torch.cat([hm, zm], dim=-1)))
        loadings = torch.tanh(self.public_loading_decoder(features)) * 0.5
        return gate * mu.squeeze(-1), sd.squeeze(-1), loadings, gate.squeeze(-1)

    def _decode_observation(self, state: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        hm, zm, ha, za = state["hm"], state["zm"], state["ha"], state["za"]
        batch = hm.shape[0]
        ids = self._asset_ids(batch, hm.device)
        emb = self.asset_embedding(ids)
        hm_a = hm[:, None, :].expand(-1, self.n_assets, -1)
        features = torch.cat([hm_a, zm[:, None, :].expand(-1, self.n_assets, -1), ha, za, emb], dim=-1)
        mu, sd = _normal_params(self.observation_decoder(features), self.cfg.min_scale)
        return mu, sd

    def _posterior_step(self, transitioned: Dict[str, Tensor], obs: Tensor) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        hm, ha = transitioned["hm"], transitioned["ha"]
        b = obs.shape[0]
        mf = market_features(obs[:, None]).squeeze(1)
        q_m_emb = self.market_obs_encoder(mf)
        q_m_mu, q_m_sd = _normal_params(self.market_post(torch.cat([hm, q_m_emb], dim=-1)), self.cfg.min_scale)
        ids = self._asset_ids(b, obs.device)
        emb = self.asset_embedding(ids)
        hm_a = hm[:, None, :].expand(-1, self.n_assets, -1)
        q_a_emb = self.asset_obs_encoder(torch.cat([obs, hm_a], dim=-1))
        q_a_mu, q_a_sd = _normal_params(self.asset_post(torch.cat([ha, q_a_emb, hm_a], dim=-1)), self.cfg.min_scale)
        q_state = {"hm": hm, "zm": q_m_mu + q_m_sd * torch.randn_like(q_m_mu),
                   "ha": ha, "za": q_a_mu + q_a_sd * torch.randn_like(q_a_mu)}
        return q_state, {"q_m_mu": q_m_mu, "q_m_sd": q_m_sd, "q_a_mu": q_a_mu, "q_a_sd": q_a_sd}

    def loss(self, context: Tensor, future: Tensor, anchor: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute prior-only multi-step innovation NLL and latent alignment.

        ``context`` is ``[B,L,N,3]``; ``future`` is ``[B,H,N,3]``; ``anchor``
        is standardized log-anchor ``[B,H,N]``.  The decoder target is the
        future standardized log-RV minus the causal standardized anchor.
        """
        if future.ndim != 4 or anchor.ndim != 3 or future.shape[0] != context.shape[0]:
            raise ValueError("invalid context/future/anchor shapes")
        state_free = self.observe(context)
        state_teacher = {k: v for k, v in state_free.items()}
        target = future[..., 0] - anchor
        h = future.shape[1]
        free_nll = torch.zeros((), device=context.device)
        teacher_nll = torch.zeros((), device=context.device)
        market_kl = torch.zeros((), device=context.device)
        asset_kl = torch.zeros((), device=context.device)
        consistency = torch.zeros((), device=context.device)
        gaps = []
        gates = []
        loading_penalty = torch.zeros((), device=context.device)
        observation_nll = torch.zeros((), device=context.device)
        for step in range(h):
            weight = self.cfg.rollout_gamma ** step
            state_free, free_out = self._prior_step(state_free)
            free_nll = free_nll + weight * _low_rank_gaussian_nll(target[:, step], free_out["mu"], free_out["sd"], free_out["loadings"]).mean()
            obs_mu, obs_sd = self._decode_observation(state_free)
            observation_nll = observation_nll + weight * (0.5 * (((future[:, step] - obs_mu) / obs_sd).square() + 2.0 * torch.log(obs_sd))).mean()
            if free_out["loadings"].shape[-1] > 0:
                loading_penalty = loading_penalty + weight * free_out["loadings"].square().mean()
            prior_teacher, prior_out = self._prior_step(state_teacher)
            state_teacher, post_out = self._posterior_step(prior_teacher, future[:, step])
            kl_m = _gaussian_kl(post_out["q_m_mu"], post_out["q_m_sd"], prior_out["p_m_mu"], prior_out["p_m_sd"]).mean()
            kl_a = _gaussian_kl(post_out["q_a_mu"], post_out["q_a_sd"], prior_out["p_a_mu"], prior_out["p_a_sd"]).mean()
            market_kl = market_kl + weight * kl_m
            asset_kl = asset_kl + weight * kl_a
            consistency = consistency + weight * ((post_out["q_m_mu"] - prior_out["p_m_mu"]).square().mean()
                                                   + (post_out["q_a_mu"] - prior_out["p_a_mu"]).square().mean()
                                                   + 0.25 * (torch.log(post_out["q_m_sd"]) - torch.log(prior_out["p_m_sd"])).square().mean()
                                                   + 0.25 * (torch.log(post_out["q_a_sd"]) - torch.log(prior_out["p_a_sd"])).square().mean())
            t_mu, t_sd, t_loadings, _ = self._decode(state_teacher)
            teacher_nll = teacher_nll + weight * _low_rank_gaussian_nll(target[:, step], t_mu, t_sd, t_loadings).mean()
            gaps.append((post_out["q_m_mu"] - prior_out["p_m_mu"]).abs().mean() + (post_out["q_a_mu"] - prior_out["p_a_mu"]).abs().mean())
            gates.append(prior_out["gate"].mean())
        denom = sum(self.cfg.rollout_gamma ** i for i in range(h))
        free_nll, teacher_nll, market_kl, asset_kl, consistency, observation_nll = (x / denom for x in (free_nll, teacher_nll, market_kl, asset_kl, consistency, observation_nll))
        kl_free = torch.clamp(market_kl + asset_kl, min=self.cfg.free_nats)
        loading_penalty = loading_penalty / denom
        total = (free_nll + 0.5 * teacher_nll + self.cfg.kl_weight * kl_free
                 + self.cfg.consistency_weight * consistency
                 + self.cfg.public_shock_scale * loading_penalty
                 + self.cfg.observation_loss_weight * observation_nll)
        parts = {"loss": total.detach(), "prior_rollout_nll": free_nll.detach(), "teacher_nll": teacher_nll.detach(),
                 "market_kl": market_kl.detach(), "asset_kl": asset_kl.detach(),
                 "consistency": consistency.detach(), "latent_gap": torch.stack(gaps).mean().detach(),
                 "gate_mean": torch.stack(gates).mean().detach(),
                 "loading_penalty": loading_penalty.detach(),
                 "observation_nll": observation_nll.detach()}
        return total, parts

    @torch.no_grad()
    def imagine(self, context: Tensor, horizon: int = 5, n_samples: int = 500) -> dict[str, Tensor]:
        if context.shape[0] != 1:
            raise ValueError("imagine currently accepts one forecast origin")
        state = self.observe(context, deterministic=True)
        state = {k: v.expand(n_samples, *v.shape[1:]).contiguous() for k, v in state.items()}
        residuals, gates, loadings = [], [], []
        for _ in range(horizon):
            state, out = self._prior_step(state)
            public = torch.randn(n_samples, self.cfg.public_shock_dim, device=context.device, dtype=context.dtype)
            idiosyncratic = torch.randn_like(out["mu"])
            residuals.append(out["mu"] + torch.einsum("naq,nq->na", out["loadings"], public)
                             + out["sd"] * idiosyncratic)
            gates.append(out["gate"])
            loadings.append(out["loadings"])
        return {"samples": torch.stack(residuals, dim=1), "gate": torch.stack(gates, dim=1),
                "loadings": torch.stack(loadings, dim=1)}


# Public name used in the manuscript and downstream scripts.
DualStateRSSM = DualStateRSSM
