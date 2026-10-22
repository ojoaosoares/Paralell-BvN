#include "bvn/euler_trail.hpp"
#include <vector>

namespace bvn {

void color_open_trails(
    const std::vector<OpenTrail>& open_trails,
    int num_vertices,
    std::vector<uint8_t>& edge_color
) {
    if (open_trails.empty()) return;

    const int K = static_cast<int>(open_trails.size());
    struct SuperAdj {
        int to;
        int trail_id;
        bool forward; // true: start_v -> end_v, false: end_v -> start_v
    };

    std::vector<int> deg(num_vertices, 0);
    std::vector<int> active_vertices;
    for (int t = 0; t < K; ++t) {
        if (deg[open_trails[t].start_v] == 0) active_vertices.push_back(open_trails[t].start_v);
        deg[open_trails[t].start_v]++;
        if (deg[open_trails[t].end_v] == 0) active_vertices.push_back(open_trails[t].end_v);
        deg[open_trails[t].end_v]++;
    }

    std::vector<int> offset(num_vertices + 1, 0);
    int total_inc = 0;
    for (int v : active_vertices) {
        offset[v] = total_inc;
        total_inc += deg[v];
    }
    std::vector<int> head = offset;
    std::vector<int> end_offset(num_vertices, 0);
    for (int v : active_vertices) {
        end_offset[v] = offset[v] + deg[v];
    }

    std::vector<SuperAdj> adj(total_inc);
    std::vector<int> fill_ptr = offset;
    for (int t = 0; t < K; ++t) {
        int u = open_trails[t].start_v;
        int v = open_trails[t].end_v;
        adj[fill_ptr[u]++] = {v, t, true};
        adj[fill_ptr[v]++] = {u, t, false};
    }

    std::vector<uint8_t> used_trail(K, 0);

    // Extract closed cycles of super-edges in H and toggle Red/Blue across original edges
    for (int root : active_vertices) {
        while (head[root] < end_offset[root]) {
            int curr = root;
            uint8_t cur_color = 0;

            while (true) {
                int chosen_trail = -1;
                int next_v = -1;
                bool forward = true;

                while (head[curr] < end_offset[curr]) {
                    const auto& item = adj[head[curr]++];
                    if (!used_trail[item.trail_id]) {
                        used_trail[item.trail_id] = 1;
                        chosen_trail = item.trail_id;
                        next_v = item.to;
                        forward = item.forward;
                        break;
                    }
                }

                if (chosen_trail == -1) {
                    break;
                }

                const auto& edges = open_trails[chosen_trail].edges;
                if (forward) {
                    for (size_t i = 0; i < edges.size(); ++i) {
                        edge_color[edges[i]] = cur_color;
                        cur_color ^= 1;
                    }
                } else {
                    for (size_t i = edges.size(); i > 0; --i) {
                        edge_color[edges[i - 1]] = cur_color;
                        cur_color ^= 1;
                    }
                }

                curr = next_v;
                if (curr == root) {
                    break;
                }
            }
        }
    }
}

} // namespace bvn
