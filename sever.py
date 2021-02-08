from collections import deque
import numpy as np
import math
import torch
import time

def hessian_vector_product_sever(actor_grad_logp, p, cg_damping=1e-1):
    return torch.mv(actor_grad_logp.permute(1,0),torch.mv(actor_grad_logp,p)) + p * cg_damping # cg_damping = 0.1

def conjugate_gradient_sever(actor_grad_logp, b, x = None, nsteps=10, residual_tol = 1e-10):
    if x is None:
        x = torch.zeros(b.size())
        r = b.clone()
    else:
        r = b.clone()-hessian_vector_product_sever(actor_grad_logp, x, cg_damping=1e-1)
    p = r.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps): # nsteps = 10
        f_Ax = hessian_vector_product_sever(actor_grad_logp, p, cg_damping=1e-1)
        alpha = rdotr / torch.dot(p, f_Ax)
        x += alpha * p
        r -= alpha * f_Ax
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol: # residual_tol = 0.0000000001
            break
    return x


def Sever_CG(actor_loss_grad, actor_grad_logp, n, nsteps = 10 , r = 4, p = 0.05):
    search_dir = None
    indices = list(range(n))
    for i in range(r):
        start_time = time.time()
        search_dir = conjugate_gradient_sever(actor_grad_logp[indices], actor_loss_grad[indices].mean(dim=0), x = search_dir, nsteps=nsteps)
#         print("--- conjugate_gradient_sever: %s seconds ---" % (time.time() - start_time))
        grads = actor_grad_logp[indices] * torch.mv(actor_grad_logp[indices],search_dir).unsqueeze(dim=1) -actor_loss_grad[indices] ## Fx-g
        mean_grads = grads.mean(dim=(-2,), keepdim=True)
        grads = grads-grads.mean(dim=(-2,), keepdim=True)
        
        start_time = time.time()
        u, s, v = torch.svd_lowrank(grads)
#         print("--- svd time: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        top_right_eigenvector = v[:,0]
        u, s, v = torch.svd_lowrank(grads)
        top_right_eigenvector = v[:,0]

        outlier_score = torch.mv(grads,top_right_eigenvector)**2
        _, topk_index = torch.topk(outlier_score,k=round(n*p))
        for index in sorted(topk_index.tolist(), reverse=True):
            del indices[index]
#         print("--- time after svd: %s seconds ---" % (time.time() - start_time))
    return search_dir, indices