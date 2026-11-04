#include "bvn/euler_parallel.hpp"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <utility>
#include <vector>

namespace bvn {

namespace {

constexpr int kFree = -1;

template <class T>
using ScratchVec = std::vector<T, NoInitAllocator<T>>; // pages first-touched by the parallel passes

/**
 * Per-odd-edge state, packed so one cycle step (read the partner link, claim, write the color)
 * touches a single 16-byte slot: the segment walk is a random pointer chase over E edges and is
 * bound by cache misses, so separate pL / pR / seg / rel arrays would cost up to 4 misses per step.
 */
struct alignas(16) EdgeNode {
    int pL;      // partner edge at the L endpoint
    int pR;      // partner edge at the R endpoint
    int seg;     // owning segment (kFree until claimed)
    uint8_t rel; // color relative to the segment start
};

inline bool cas_int(int* p, int expected, int desired) {
    return __atomic_compare_exchange_n(p, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

inline int chunk_begin(int64_t total, int c, int chunks) {
    return static_cast<int>((total * c) / chunks);
}

/**
 * Per-vertex odd-degree histogram per chunk, turned into exclusive offsets (vertex-major,
 * chunk-minor) in hist, with base[x] = first slot of vertex x and base[n] = E.
 */
void endpoint_offsets(int n, int E, const uint16_t* endpoint, int chunks, int* hist, int* base) {
    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        int* hc = hist + static_cast<size_t>(c) * n;
        std::fill(hc, hc + n, 0);
        const int lo = chunk_begin(E, c, chunks);
        const int hi = chunk_begin(E, c + 1, chunks);
        for (int k = lo; k < hi; ++k) {
            hc[endpoint[k]]++;
        }
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }
    int run = 0;
    for (int x = 0; x < n; ++x) {
        base[x] = run;
        for (int c = 0; c < chunks; ++c) {
            int& slot = hist[static_cast<size_t>(c) * n + x];
            const int cnt = slot;
            slot = run;
            run += cnt;
        }
    }
    base[n] = run;
}

/**
 * Pairs the odd edges at every L vertex (pL) and R vertex (pR); consecutive edges of a vertex
 * become partners (each vertex has an even number of odd edges). Also resets seg to kFree.
 * If the edges are grouped by L endpoint (runs of equal ou[k], as the root and the parallel split
 * produce), the L side needs only the run offsets; otherwise both sides use a counting sort.
 */
void pair_odd_edges(int n, int E, const uint16_t* ou, const uint16_t* ov, bool grouped_by_u, int chunks, EdgeNode* node) {
    ScratchVec<int> hist(static_cast<size_t>(chunks) * n);
    ScratchVec<int> base(n + 1);
    ScratchVec<int> sorted(E);
    int* h = hist.data();
    int* bs = base.data();
    int* srt = sorted.data();

    for (int side = 0; side < 2; ++side) {
        const uint16_t* endpoint = (side == 0) ? ou : ov;
        endpoint_offsets(n, E, endpoint, chunks, h, bs);
        const bool direct = (side == 0) && grouped_by_u;

        if (direct) {
            // Run of vertex u is [base[u], base[u+1]): partner = base + ((k - base) ^ 1).
            #pragma omp taskloop grainsize(1)
            for (int c = 0; c < chunks; ++c) {
                double t0 = omp_get_wtime();
                const int lo = chunk_begin(E, c, chunks);
                const int hi = chunk_begin(E, c + 1, chunks);
                for (int k = lo; k < hi; ++k) {
                    const int b = bs[ou[k]];
                    node[k].pL = b + ((k - b) ^ 1);
                    node[k].seg = kFree;
                }
                get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
            }
            continue;
        }

        #pragma omp taskloop grainsize(1)
        for (int c = 0; c < chunks; ++c) {
            double t0 = omp_get_wtime();
            int* hc = h + static_cast<size_t>(c) * n;
            const int lo = chunk_begin(E, c, chunks);
            const int hi = chunk_begin(E, c + 1, chunks);
            for (int k = lo; k < hi; ++k) {
                srt[hc[endpoint[k]]++] = k;
            }
            get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
        }

        #pragma omp taskloop grainsize(1)
        for (int c = 0; c < chunks; ++c) {
            double t0 = omp_get_wtime();
            const int x_lo = chunk_begin(n, c, chunks);
            const int x_hi = chunk_begin(n, c + 1, chunks);
            for (int i = bs[x_lo]; i < bs[x_hi]; i += 2) {
                const int a = srt[i];
                const int b = srt[i + 1];
                if (side == 0) {
                    node[a].pL = b;
                    node[b].pL = a;
                    node[a].seg = kFree;
                    node[b].seg = kFree;
                } else {
                    node[a].pR = b;
                    node[b].pR = a;
                }
            }
            get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
        }
    }
}

/**
 * Extends segment `sid` from edge `start` along the pairing cycle, taking the L partner first
 * when left_first is set (forward) or the R partner first (backward), alternating afterwards.
 * Returns true if the cycle closed on the segment itself; otherwise records the boundary
 * (last owned edge, first foreign edge).
 */
bool extend_segment(
    int start,
    bool left_first,
    int sid,
    EdgeNode* node,
    std::vector<std::pair<int, int>>& boundaries
) {
    int cur = start;
    uint8_t c = node[start].rel;
    bool left = left_first;
    while (true) {
        const int nx = left ? node[cur].pL : node[cur].pR;
        if (cas_int(&node[nx].seg, kFree, sid)) {
            c ^= 1;
            node[nx].rel = c;
            cur = nx;
            left = !left;
            continue;
        }
        if (__atomic_load_n(&node[nx].seg, __ATOMIC_ACQUIRE) == sid) {
            return true;
        }
        boundaries.emplace_back(cur, nx);
        return false;
    }
}

int uf_find(std::vector<int>& parent, std::vector<uint8_t>& parity, int x, uint8_t& out_parity) {
    int root = x;
    uint8_t p = 0;
    while (parent[root] != root) {
        p ^= parity[root];
        root = parent[root];
    }
    // Path compression: re-point every node on the path at the root with its parity to the root.
    int cur = x;
    uint8_t cur_p = p;
    while (parent[cur] != cur) {
        const int next = parent[cur];
        const uint8_t next_p = cur_p ^ parity[cur];
        parent[cur] = root;
        parity[cur] = cur_p;
        cur = next;
        cur_p = next_p;
    }
    out_parity = p;
    return root;
}

} // namespace

void color_odd_edges_parallel(
    int n,
    int E,
    const uint16_t* ou,
    const uint16_t* ov,
    bool grouped_by_u,
    uint8_t* color,
    int num_tasks
) {
    if (E == 0) return;
    const int chunks = std::max(1, std::min(num_tasks, E / 256));

    ScratchVec<EdgeNode> node_vec(E);
    EdgeNode* node = node_vec.data();
    pair_odd_edges(n, E, ou, ov, grouped_by_u, chunks, node);

    // Phase 2: concurrent segment growth. Each chunk seeds segments at its own unclaimed edges,
    // so threads start far apart on long cycles and only meet at segment boundaries.
    int next_sid = 0;
    int* next_sid_p = &next_sid;
    std::vector<std::vector<std::pair<int, int>>> boundaries(chunks);
    std::vector<std::pair<int, int>>* bnd = boundaries.data();

    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        const int lo = chunk_begin(E, c, chunks);
        const int hi = chunk_begin(E, c + 1, chunks);
        int spare = kFree;
        for (int e = lo; e < hi; ++e) {
            if (__atomic_load_n(&node[e].seg, __ATOMIC_RELAXED) != kFree) continue;
            const int sid = (spare != kFree) ? spare : __atomic_fetch_add(next_sid_p, 1, __ATOMIC_RELAXED);
            if (!cas_int(&node[e].seg, kFree, sid)) {
                spare = sid;
                continue;
            }
            spare = kFree;
            node[e].rel = 0;
            if (!extend_segment(e, true, sid, node, bnd[c])) {
                extend_segment(e, false, sid, node, bnd[c]);
            }
        }
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }

    // Phase 3: stitch segments. Across a boundary (a, x) the final colors must differ:
    // rel[a] ^ flip[seg[a]] != rel[x] ^ flip[seg[x]]  =>  flip[sa] ^ flip[sx] = 1 ^ rel[a] ^ rel[x].
    double ts0 = omp_get_wtime();
    const int num_seg = next_sid;
    std::vector<int> parent(num_seg);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<uint8_t> parity(num_seg, 0);
    for (int c = 0; c < chunks; ++c) {
        for (const auto& [a, x] : boundaries[c]) {
            uint8_t pa = 0, px = 0;
            const int ra = uf_find(parent, parity, node[a].seg, pa);
            const int rx = uf_find(parent, parity, node[x].seg, px);
            if (ra == rx) continue; // Both sides recorded the same link, or the loop is closed
            parent[ra] = rx;
            parity[ra] = static_cast<uint8_t>(pa ^ px ^ 1 ^ node[a].rel ^ node[x].rel);
        }
    }
    std::vector<uint8_t> flip_vec(num_seg, 0);
    for (int s = 0; s < num_seg; ++s) {
        uint8_t p = 0;
        uf_find(parent, parity, s, p);
        flip_vec[s] = p;
    }
    get_thread_workspace().active_time_ms += (omp_get_wtime() - ts0) * 1000.0;

    const uint8_t* flip = flip_vec.data();
    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        const int lo = chunk_begin(E, c, chunks);
        const int hi = chunk_begin(E, c + 1, chunks);
        for (int e = lo; e < hi; ++e) {
            color[e] = node[e].rel ^ flip[node[e].seg];
        }
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }
}

bool euler_split_parallel(
    const std::vector<WeightedEdge>& edges,
    int n,
    int num_tasks,
    EulerWorkspace& ws,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    const int E_in = static_cast<int>(edges.size());
    const int chunks = std::max(1, std::min(num_tasks, E_in / 1024));
    const WeightedEdge* in = edges.data();

    std::vector<int> odd_off(chunks + 1, 0), red_off(chunks + 1, 0), blue_off(chunks + 1, 0);
    std::vector<uint8_t> chunk_sorted(chunks, 1);
    int* odd_p = odd_off.data();
    int* red_p = red_off.data();
    int* blue_p = blue_off.data();
    uint8_t* sorted_p = chunk_sorted.data();

    // Count odd edges per chunk and check whether the input is grouped by u (non-decreasing).
    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        const int lo = chunk_begin(E_in, c, chunks);
        const int hi = chunk_begin(E_in, c + 1, chunks);
        int cnt = 0;
        uint8_t ok = 1;
        for (int i = lo; i < hi; ++i) {
            cnt += in[i].w & 1;
            if (i > lo && in[i].u < in[i - 1].u) ok = 0;
        }
        odd_p[c + 1] = cnt;
        sorted_p[c] = ok;
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }
    bool grouped_by_u = true;
    for (int c = 0; c < chunks; ++c) {
        odd_p[c + 1] += odd_p[c];
        const int lo = chunk_begin(E_in, c, chunks);
        if (!sorted_p[c] || (lo > 0 && in[lo].u < in[lo - 1].u)) grouped_by_u = false;
    }
    const int E = odd_p[chunks];
    const bool symmetric = (E == 0);

    ws.splits++;
    ws.edges_scanned += E_in;
    ws.odd_edges_walked += E;
    if (symmetric) ws.sym_pruned++;

    ScratchVec<uint16_t> ou_vec(E), ov_vec(E);
    ScratchVec<uint8_t> color_vec(E);
    uint16_t* ou = ou_vec.data();
    uint16_t* ov = ov_vec.data();
    uint8_t* color = color_vec.data();

    if (!symmetric) {
        #pragma omp taskloop grainsize(1)
        for (int c = 0; c < chunks; ++c) {
            double t0 = omp_get_wtime();
            const int lo = chunk_begin(E_in, c, chunks);
            const int hi = chunk_begin(E_in, c + 1, chunks);
            int k = odd_p[c];
            for (int i = lo; i < hi; ++i) {
                if (in[i].w & 1) {
                    ou[k] = in[i].u;
                    ov[k] = in[i].v;
                    ++k;
                }
            }
            get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
        }
        color_odd_edges_parallel(n, E, ou, ov, grouped_by_u, color, num_tasks);
    }

    // Count, then fill, the Red/Blue halves: w = 2q + r gives Red q + [r && red], Blue q + [r && blue].
    // Both passes keep the input order, so outputs stay grouped by u when the input was.
    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        const int lo = chunk_begin(E_in, c, chunks);
        const int hi = chunk_begin(E_in, c + 1, chunks);
        int k = odd_p[c];
        int rc = 0, bc = 0;
        for (int i = lo; i < hi; ++i) {
            const int w = in[i].w;
            const int half = w >> 1;
            int r = 0, b = 0;
            if (w & 1) {
                const int is_blue = color[k++];
                r = 1 - is_blue;
                b = is_blue;
            }
            rc += (half + r) > 0;
            bc += (half + b) > 0;
        }
        red_p[c + 1] = rc;
        blue_p[c + 1] = symmetric ? 0 : bc;
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }
    for (int c = 0; c < chunks; ++c) {
        red_p[c + 1] += red_p[c];
        blue_p[c + 1] += blue_p[c];
    }
    out_red.resize(red_p[chunks]);
    out_blue.resize(blue_p[chunks]);
    WeightedEdge* red_out = out_red.data();
    WeightedEdge* blue_out = out_blue.data();

    #pragma omp taskloop grainsize(1)
    for (int c = 0; c < chunks; ++c) {
        double t0 = omp_get_wtime();
        const int lo = chunk_begin(E_in, c, chunks);
        const int hi = chunk_begin(E_in, c + 1, chunks);
        int k = odd_p[c];
        WeightedEdge* rp = red_out + red_p[c];
        WeightedEdge* bp = blue_out + blue_p[c];
        for (int i = lo; i < hi; ++i) {
            const WeightedEdge& e = in[i];
            const int half = e.w >> 1;
            int r = 0, b = 0;
            if (e.w & 1) {
                const int is_blue = color[k++];
                r = 1 - is_blue;
                b = is_blue;
            }
            if (half + r > 0) *rp++ = {e.u, e.v, half + r};
            if (!symmetric && half + b > 0) *bp++ = {e.u, e.v, half + b};
        }
        get_thread_workspace().active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    }

    return symmetric;
}

} // namespace bvn
