import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from readout import readout_moments,ridge


def test_grouped_readout_matches_direct_design_and_preserves_cross_tokens():
    torch.manual_seed(13)
    q=torch.randn(4,7,3);k=torch.randn(2,9,3)
    x=torch.randn(9,5,dtype=torch.float64);y=torch.randn(9,6,dtype=torch.float64)
    g,c,yy,mass,_=readout_moments(q,k,x,y,query_block=3)
    a=torch.softmax(q@k.repeat_interleave(2,0).mT/3**.5,dim=-1).double()
    torch.testing.assert_close(mass,a.mean((0,1)),atol=1e-8,rtol=1e-6)
    for h in range(2):
        ah=a[2*h:2*h+2].reshape(14,9);z=ah@x;t=ah@y[:,3*h:3*h+3]
        torch.testing.assert_close(g[h],z.T@z/14,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(c[h],z.T@t/14,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(yy[h],t.square().sum()/14,atol=1e-7,rtol=1e-6)
    # Nonidentical query heads must NOT collapse to their mean before AX.
    mean_z=(a[:2].mean(0)@x)
    assert not torch.allclose(g[0],mean_z.T@mean_z/7,atol=1e-4)
    # A^T A has cross-token terms, unlike token mass diagonal.
    assert not torch.allclose(g[0],x.T@(mass[:,None]*x),atol=1e-4)


def test_ridge_solves_regularized_objective_and_not_pseudoinverse():
    torch.manual_seed(8)
    x=torch.randn(20,5,dtype=torch.float64);x[:,4]=x[:,0]
    y=torch.randn(20,3,dtype=torch.float64)
    w,info=ridge(x.T@x,x.T@y)
    expected=torch.linalg.solve(x.T@x+.01*torch.eye(5,dtype=torch.float64),x.T@y)
    torch.testing.assert_close(w.double(),expected,atol=1e-7,rtol=1e-6)
    assert info['normal_residual']<1e-10


def test_batched_head_solves_equal_separate_solves():
    torch.manual_seed(9)
    x=torch.randn(3,17,7,dtype=torch.float64);y=torch.randn(3,17,2,dtype=torch.float64)
    w,_=ridge(x.mT@x,x.mT@y)
    for h in range(3):torch.testing.assert_close(w[h],ridge(x[h].T@x[h],x[h].T@y[h])[0])


def test_bias_requires_actual_attention_mass_for_partial_prefix():
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    w=torch.eye(2,dtype=torch.float64);b=torch.tensor([2.,3.],dtype=torch.float64)
    a=torch.tensor([[.1,.2]],dtype=torch.float64)
    torch.testing.assert_close(a@(x@w+b),a@x@w+a.sum(-1,keepdim=True)*b)
    assert not torch.allclose(a@(x@w+b),a@x@w+b)


def test_readout_error_cancellation_is_not_weighted_mse():
    a=torch.tensor([[.5,.5]],dtype=torch.float64)
    e=torch.tensor([[2.],[-2.]],dtype=torch.float64)
    assert (a@e).square().sum()==0
    assert (a.flatten()*e.square().flatten()).sum()==4


def test_query_chunking_invariant():
    torch.manual_seed(6)
    q=torch.randn(4,11,8);k=torch.randn(2,13,8)
    x=torch.randn(13,9);y=torch.randn(13,16)
    aa=readout_moments(q,k,x,y,1);bb=readout_moments(q,k,x,y,11)
    for a,b in zip(aa[:4],bb[:4]):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_readout_ridge_improves_own_objective():
    torch.manual_seed(16)
    q=torch.randn(4,13,4);k=torch.randn(2,17,4)
    x=torch.randn(17,6,dtype=torch.float64);y=torch.randn(17,8,dtype=torch.float64)
    g,c,yy,mass,_=readout_moments(q,k,x,y)
    wr,_=ridge(g,c)
    wb,_=ridge(x.T@(mass[:,None]*x),x.T@(mass[:,None]*y))
    wb=wb.reshape(6,2,4).permute(1,0,2).double();wr=wr.double()
    def loss(w):return (w*(g@w)).sum()-2*(w*c).sum()+yy.sum()+.01*w.square().sum()
    assert loss(wr)<=loss(wb)+1e-10
