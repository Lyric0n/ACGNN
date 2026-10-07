"""Construction of the four-bucket conversation graph used by ACGNN.

Every dialogue yields ``2 * len`` nodes: the first ``len`` carry the text (main
modality) representations, the following ``len`` carry the fused audio-visual
(auxiliary) representations. Nodes are the only carriers of information between the
two modalities -- the cross-modal edges are directed from the auxiliary to the main
nodes, so information flows audio-visual -> text.

Four edge buckets are produced, and each bucket is consumed by its own GAT branch:

======================  ==========================================================
bucket                  definition
======================  ==========================================================
intra-modal same spk    text-text pairs of the same speaker
intra-modal diff spk    text-text pairs of different speakers
cross-modal same spk    auxiliary -> text pairs of the same speaker, within window
cross-modal diff spk    auxiliary -> text pairs of different speakers, within window
======================  ==========================================================

The cross-modal window is controlled by ``win_aux``: ``0`` disables cross-modal
edges, ``1`` keeps only the aligned (same-utterance) pair, and ``w >= 2`` keeps
aligned plus all pairs whose distance lies in ``[1, w - 1]``.
"""

import torch


def build_heterogeneous_graph(main_mod, aux_mod, dia_len, qmask, win_aux=3, no_cuda=False):
    device = main_mod.device
    batch_size = main_mod.size(1)

    node_counts = [2 * len_i for len_i in dia_len]
    total_nodes = sum(node_counts)
    node_features = torch.zeros(total_nodes, main_mod.size(2), device=device)

    cum_counts = [0] + torch.cumsum(torch.tensor(node_counts, device=device), dim=0).tolist()[:-1]
    for i, offset in enumerate(cum_counts):
        len_i = dia_len[i]
        node_features[offset:offset + len_i] = main_mod[:len_i, i]
        node_features[offset + len_i:offset + 2 * len_i] = aux_mod[:len_i, i]

    edges_type1 = []
    edges_type2 = []
    edges_type3 = []
    edges_type4 = []

    global_offset = 0
    for i in range(batch_size):
        len_i = dia_len[i]
        if len_i == 0:
            continue

        main_start = global_offset
        aux_start = global_offset + len_i
        global_offset += 2 * len_i

        speaker = torch.argmax(qmask[:len_i, i], dim=1)

        t_indices = torch.arange(len_i, device=device)
        j_indices = torch.arange(len_i, device=device)
        j_grid, t_grid = torch.meshgrid(j_indices, t_indices, indexing='ij')

        same_spk_mask = (speaker[j_grid] == speaker[t_grid]) & (j_grid != t_grid)
        j_same, t_same = j_grid[same_spk_mask], t_grid[same_spk_mask]
        edges_type1.append(torch.stack([main_start + j_same, main_start + t_same]))

        diff_spk_mask = (speaker[j_grid] != speaker[t_grid])
        j_diff, t_diff = j_grid[diff_spk_mask], t_grid[diff_spk_mask]
        edges_type2.append(torch.stack([main_start + j_diff, main_start + t_diff]))

        if win_aux == 0:
            continue

        self_edges = torch.stack([aux_start + t_indices, main_start + t_indices])

        if win_aux == 1:
            edges_type3.append(self_edges)
        else:
            dist_matrix = torch.abs(j_grid - t_grid)
            window_mask = (dist_matrix >= 1) & (dist_matrix <= (win_aux - 1))

            same_aux_mask = window_mask & (speaker[j_grid] == speaker[t_grid])
            j_same_aux, t_same_aux = j_grid[same_aux_mask], t_grid[same_aux_mask]
            edges_type3.append(torch.cat([
                self_edges,
                torch.stack([aux_start + j_same_aux, main_start + t_same_aux]),
            ], dim=1))

            diff_aux_mask = window_mask & (speaker[j_grid] != speaker[t_grid])
            j_diff_aux, t_diff_aux = j_grid[diff_aux_mask], t_grid[diff_aux_mask]
            edges_type4.append(torch.stack([aux_start + j_diff_aux, main_start + t_diff_aux]))

    def concat_edges(edge_list):
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(edge_list, dim=1)

    edge_index_type1 = concat_edges(edges_type1)
    edge_index_type2 = concat_edges(edges_type2)
    edge_index_type3 = concat_edges(edges_type3) if win_aux > 0 else torch.empty(
        (2, 0), dtype=torch.long, device=device)
    edge_index_type4 = concat_edges(edges_type4) if win_aux > 1 else torch.empty(
        (2, 0), dtype=torch.long, device=device)

    if no_cuda:
        node_features = node_features.cpu()
        edge_index_type1 = edge_index_type1.cpu()
        edge_index_type2 = edge_index_type2.cpu()
        edge_index_type3 = edge_index_type3.cpu()
        edge_index_type4 = edge_index_type4.cpu()

    edge_indices = (edge_index_type1, edge_index_type2, edge_index_type3, edge_index_type4)
    return node_features, edge_indices


def extract_main_features(all_features, dia_len):
    main_features = []
    start_idx = 0
    for length in dia_len:
        main_features.append(all_features[start_idx:start_idx + length])
        start_idx += 2 * length
    return torch.cat(main_features, dim=0)
