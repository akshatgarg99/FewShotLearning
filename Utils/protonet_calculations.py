import torch


def pairwise_distances(x, y):
    # calculate the pairwise distance between a query and
    # the protypes(mean embedding) of classes. (many ways to calculate it)
    n_x = x.shape[0]
    n_y = y.shape[0]
    distances = (x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)).pow(2).sum(dim=2)
    return distances


def compute_prototypes(support, k, n):
    # compute the mean embedding for a class using n sample support images
    return support.reshape(k, n, -1).mean(dim=1)


def proto_net_episode(model, optimiser, loss_fn, x, y, n_shot, k_way, q_queries, train=True):
    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()
    # calculate the embeddings
    embeddings = model(x)

    # divide the sample in support and queries
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)
    distances = pairwise_distances(queries, prototypes)
    # log_p_y = (-distances).log_softmax(dim=1)

    # cross entropy loss
    loss = loss_fn(-distances, y)
    # predictions
    y_pred = (-distances).softmax(dim=1)

    if train:
        loss.backward()
        optimiser.step()
        return loss, y_pred
    else:
        result = torch.argmin(distances, dim=1)
        total = y.size(0)
        correct = result.eq(y.data).cpu().sum()
        return correct, total
