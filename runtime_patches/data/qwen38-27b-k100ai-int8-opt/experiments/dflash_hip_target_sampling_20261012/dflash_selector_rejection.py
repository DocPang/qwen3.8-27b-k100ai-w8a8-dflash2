"""Research-only exact sparse-q DFlash2 rejection sampler.

Greedy path is left to existing verifier. For non-greedy rows draft must have
actually been sampled from q_rows through the selector. Does not modify production.
"""
import torch
import torch.nn.functional as F

def verify_sparse_selector(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    selector_ids: torch.Tensor,
    selector_q: torch.Tensor,
    sampling_info,
):
    bs, width = candidates.shape
    gamma = width - 1
    vocab = target_logits.shape[-1]
    assert target_logits.shape == (bs * width, vocab)
    assert selector_ids.shape[:2] == (bs, gamma)
    assert selector_q.shape == selector_ids.shape
    # A sparse p representation is sufficient: q has support <= selector_top_k.
    t = sampling_info.temperatures.reshape(-1).float().clamp_min(1e-5)
    # A zero temperature row becomes a target argmax distribution.
    ks = sampling_info.top_ks.reshape(-1).to(torch.long).clamp(1,vocab)
    assert bool(getattr(sampling_info,"need_top_k_sampling",True)), "sparse reference requires top-k"
    maxk = int(ks.max().item())
    if maxk > 1024: raise ValueError("top-k over 1024 is outside optimized path")
    v, ids = torch.topk(target_logits.reshape(bs,width,vocab),k=maxk,dim=-1)
    v = v.float() / t[:,None,None]
    rank = torch.arange(maxk,device=ids.device)[None,None,:]
    v = v.masked_fill(rank>=ks[:,None,None], float("-inf"))
    p = F.softmax(v,dim=-1)
    if bool(getattr(sampling_info,"need_top_p_sampling",False)):
        ps = sampling_info.top_ps.reshape(-1).float().clamp_min(1e-6)
        keep = (p.cumsum(-1)-p) < ps[:,None,None]
        p = p.masked_fill(~keep,0)
        p = p / p.sum(-1,keepdim=True)
    # Probability of each selected proposal under exact p and sparse q.
    proposed = candidates[:,1:]
    p_at_draft = torch.sum(
        (ids[:,:gamma,:] == proposed[:,:,None]).to(p.dtype) * p[:,:gamma,:],dim=-1
    )
    q_at_draft = torch.sum(
        (selector_ids == proposed[:,:,None]).to(selector_q.dtype) * selector_q,dim=-1
    )
    # Draft ids are guaranteed in q support by the selector walk.
    p_over_q=(p_at_draft/q_at_draft).clamp(max=1)
    accept=torch.rand((bs,gamma),device=ids.device)<p_over_q
    accept_len=accept.to(torch.int32).cumprod(-1).sum(-1)
    # Reject correction: sample from residual normalized max(0, p-q).
    # Only p's top-k support needs storage.
    q_on_p = (
        (selector_ids[:,:,:,None] == ids[:,:gamma,None,:]).to(selector_q.dtype)
        * selector_q[:,:,:,None]
    ).sum(-2)
    residual=(p[:,:gamma,:]-q_on_p).clamp_min(0)
    row=torch.arange(bs,device=ids.device)
    fail_pos=accept_len.to(torch.long).clamp(max=gamma-1)
    correction=residual[row,fail_pos,:]
    # If p==q then rejection is impossible; fallback protects from 0/0.
    denom=correction.sum(-1,keepdim=True)
    correction=torch.where(denom>1e-12,correction/denom.clamp_min(1e-12),p[row,fail_pos,:])
    use_bonus=(accept_len==gamma)[:,None]
    dist=torch.where(use_bonus,p[row,gamma,:],correction)
    row_tokens=torch.where(use_bonus,ids[row,gamma,:],ids[row,fail_pos,:])
    sampled_ix=torch.multinomial(dist,1)
    bonus=row_tokens.gather(1,sampled_ix).reshape(-1)
    return accept_len.to(torch.int32),bonus.to(torch.int64)

if __name__=="__main__":
    import types
    torch.manual_seed(42)
    N=15000
    p=torch.tensor([.05,.25,.60,.10]);q=torch.tensor([.50,.10,.30,.10])
    proposal=torch.multinomial(q.expand(N,-1),1).reshape(N)
    cand=torch.stack([torch.zeros_like(proposal),proposal],-1)
    p0=torch.log(p).expand(N*2,-1).clone()
    ids=torch.arange(4).view(1,1,4).expand(N,1,4).clone()
    qs=q.view(1,1,4).expand(N,1,4).clone()
    info=types.SimpleNamespace(temperatures=torch.ones(N),top_ks=torch.full((N,),4),
        top_ps=torch.ones(N),need_top_k_sampling=True,need_top_p_sampling=False)
    a,b=verify_sparse_selector(candidates=cand,target_logits=p0,
        selector_ids=ids,selector_q=qs,sampling_info=info)
    emitted=torch.where(a>0,proposal,b)
    freq=torch.bincount(emitted,minlength=4).float()/N
    expected_overlap=torch.minimum(p,q).sum().item()
    print("MONTE_CARLO", "freq",freq.tolist(),"p",p.tolist(),
        "mean_accept",round(a.float().mean().item(),4),
        "expected_overlap",round(expected_overlap,4))
    assert torch.allclose(freq,p,atol=.02), "distribution mismatch"
    assert abs(a.float().mean().item()-expected_overlap)<.02
    print("SPARSE_REJECTION_CPU_PASS")
