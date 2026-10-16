"""HIP DFlash: sample target distributions and accept only matching draft prefix."""
import torch
import torch.nn.functional as F

def sample_target_tokens(logits, info, *, draft_token_num):
    if logits.ndim != 2 or draft_token_num < 1:
        raise ValueError("invalid logits or draft width")
    bs = len(info)
    rows, vocab = logits.shape
    if rows != bs * draft_token_num:
        raise ValueError("logits length mismatch")
    if torch.distributed.is_initialized():
        from sglang.srt.distributed.parallel_state import get_tp_group
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group
    else:
        tp_group, tp_rank = None, 0
    if tp_rank == 0:
        ts = info.temperatures.reshape(-1).to(device=logits.device, dtype=torch.float32)
        greedy = ts <= 1e-6
        t = torch.where(greedy,torch.ones_like(ts),ts.clamp_min(1e-6))
        scaled = logits.float() / t.repeat_interleave(draft_token_num)[:,None]
        use_k = bool(getattr(info, "need_top_k_sampling", False))
        use_p = bool(getattr(info, "need_top_p_sampling", False))
        if use_k:
            ks = info.top_ks.reshape(-1).to(device=logits.device,dtype=torch.int64).clamp(1,vocab)
            mk = int(ks.max().item())
            values, token_ids = torch.topk(scaled,k=mk,dim=-1)
            ranks = torch.arange(mk,device=logits.device)[None,:]
            values = values.masked_fill(ranks >= ks.repeat_interleave(draft_token_num)[:,None],float("-inf"))
            probs = F.softmax(values,dim=-1)
        else:
            token_ids = None
            probs = F.softmax(scaled,dim=-1)
        if use_p:
            ps = info.top_ps.reshape(-1).to(device=logits.device,dtype=torch.float32).repeat_interleave(draft_token_num)
            if token_ids is None:
                probs,token_ids = torch.sort(probs,descending=True,dim=-1)
            mask = (probs.cumsum(dim=-1)-probs) < ps[:,None]
            probs = probs.masked_fill(~mask,0.0)
            probs = probs/probs.sum(dim=-1,keepdim=True)
        if not bool(torch.isfinite(probs).all().item()):
            raise RuntimeError("nonfinite sampling probabilities")
        idx=torch.multinomial(probs,1)
        sampled=token_ids.gather(1,idx).reshape(-1) if token_ids is not None else idx.reshape(-1)
        if bool(greedy.any().item()):
            sampled=torch.where(greedy.repeat_interleave(draft_token_num),scaled.argmax(dim=-1).to(sampled.dtype),sampled)
        sampled=sampled.reshape(bs,draft_token_num).long().contiguous()
    else:
        sampled=torch.empty((bs,draft_token_num),device=logits.device,dtype=torch.long)
    if tp_group is not None and tp_group.world_size > 1:
        tp_group.broadcast(sampled,src=0)
    return sampled
