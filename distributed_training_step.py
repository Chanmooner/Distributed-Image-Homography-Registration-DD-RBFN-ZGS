import torch
from zgs import zgs_update


def distributed_training_step(agent, features, targets, neighbor_params):
    
    loss = agent.compute_loss(features, targets)
    loss.backward()

    local_param = agent.get_parameters()
    local_grad = torch.cat(
        [p.grad.view(-1) for p in agent.model.parameters()]
    )

    delta = zgs_update(
        local_grad,
        local_param,
        neighbor_params,
        agent.alpha
    )

    agent.set_parameters(local_param + delta)
    agent.model.zero_grad()

    return loss.item()
