//
// Created by atrostan on 02/04/23.
//

#include "CSR.h"
#include "FastHilbert.h"
#include "IndexedEdge.h"
#include "RecursiveBlock.h"
// #include "omp.h"
#include "util.h"
#include <chrono>
#include <omp.h>

using namespace std::chrono;

#ifndef RHUBARB_RHUBARB_H
#define RHUBARB_RHUBARB_H

auto now() { return std::chrono::high_resolution_clock::now(); }
template<typename T>
auto elapsed_ms(time_point<system_clock, T> end,
                time_point<system_clock, T> start) {
  return std::chrono::duration_cast<milliseconds>(end - start).count();
}

/**
 * given an input Compressed Sparse Column and Compressed Sparse Row of a graph,
 * preprocess the graph using Recursive-Hilbert Blocking:
 * 1. Compute the in, out degree vectors
 * 2. Partition the graph into blocks using Recursive Hilbert Blocking
 */
template<typename EdgeType>
class Rhubarb {
  public:
  using Block = RecursiveBlock<EdgeType>;
  using IdxdEdge = IndexedEdge<EdgeType>;

  inline static constexpr bool Weighted = std::is_same<EdgeType, WeightedEdge>::value;

  CSR &in_csr;
  uint32_t num_nodes;
  uint64_t num_edges;
  uint64_t
          nn;// hyperceiling(n): the smallest power of 2 greater than num_nodes
  int num_threads;

  uint64_t hilbert_hyperceil;
  uint8_t hilbert_order;

  uint32_t max_num_edges;
  // recurse at least this many times before counting the number of edges in a
  // block
  uint32_t min_depth = 5;
  uint32_t n_nonempty_blocks = 0;
  uint64_t total_es_assigned = 0;
  // thread-private offset - used to insert blocks to thread private block
  // vector
  std::vector<uint32_t> block_offsets;
  std::vector<uint32_t> nnz_block_offsets;

  uint64_t rhb_time = 0;
  uint64_t flatten_time = 0;
  uint64_t greedy_merge_time = 0;
  uint64_t agg_edges_time = 0;
  uint64_t preproc_time = 0;

  std::vector<std::vector<Block>> priv_bs;// thread-private blocks
  std::vector<Block> bs;                  // flattened, sorted recursive hilbert blocks


  /**
 * @brief Construct a new Rhubarb object
 * Rhubarb default constructor
 * Takes in references to in-csr and initializes number of nodes, edges
 * 
 * @param _in_csr 
 * @param _num_threads 
 * @param _max_num_edges 
 * @param block_edges array of hilbert-ordered edges
 * @param block_index array of offsets that point to start + end of 
 * hilbert-ordered blocks
 * @param block_weights array of weights (correspond to hilbert-ordered edges)
 * @param _weighted is the graph weighted or not
 */
  Rhubarb(CSR &_in_csr, int _num_threads,
          uint32_t _max_num_edges, uint32_t *&block_edges,
          uint32_t *&block_index, double *&block_weights)
      : in_csr(_in_csr), num_threads(_num_threads),
        max_num_edges(_max_num_edges) {
    uint max_threads = omp_get_max_threads();
    max_threads = 1;
    num_threads = max_threads;
    num_nodes = in_csr.num_nodes();
    num_edges = in_csr.num_edges();

    nn = hyperceiling(num_nodes);
    hilbert_hyperceil = nn;
    hilbert_order = log2(nn);
    priv_bs.resize(num_threads);
    block_offsets.resize(num_threads);
    nnz_block_offsets.resize(num_threads);
    auto rhb_start_time = now();
    recursive_hilbert_block();
    auto rhb_end_time = now();
    auto flat_start_time = now();
    compute_nnz_block_offsets();
    flatten_block_vector();

    // std::string unmerged_rec_bs_path =
    //         fmt::format("./unmerged_{}-rec_bs", max_num_edges);
    // write_blocks_metadata(unmerged_rec_bs_path);

    auto flat_end_time = now();
    rhb_time = elapsed_ms(rhb_end_time, rhb_start_time);
    flatten_time = elapsed_ms(flat_end_time, flat_start_time);
    fmt::print("weighted?: {}\n", Weighted);
    // before flattening the edge array try to greedily merge adjacency blocks
    // in parallel
    auto merge_start = now();
    std::vector<uint32_t> thread_merge_bounds(num_threads + 1);

    uint32_t n_blocks_per_thread = (n_nonempty_blocks - 1) / num_threads;
    for (uint t = 0; t < num_threads; ++t) {
      thread_merge_bounds[t] = n_blocks_per_thread * t;
      thread_merge_bounds[t + 1] = n_blocks_per_thread * (t + 1);
    }
    thread_merge_bounds[num_threads] = n_nonempty_blocks;

    // parallel_greedy_merge(thread_merge_bounds);
    std::vector<uint32_t> nnz_blocks_per_thread(num_threads + 1);
    hilbert_order_edges_in_blocks(thread_merge_bounds, nnz_blocks_per_thread);
    auto merge_end = now();
    greedy_merge_time = elapsed_ms(merge_end, merge_start);
    std::vector<uint32_t> orig_block_idcs;

    // since the original block vector will contain empty blocks after the
    // parallel greedy merge,
    // compute the map between nonzero block index to original block index
    uint32_t cum_nnz_block_idx = 0;
    for (uint t = 0; t < num_threads + 1; t++) {
      cum_nnz_block_idx += nnz_blocks_per_thread[t];
      nnz_blocks_per_thread[t] = cum_nnz_block_idx;
    }
    uint32_t nnz_blocks_after_merge = nnz_blocks_per_thread[num_threads];

    orig_block_idcs.resize(nnz_blocks_after_merge);

    auto agg_start = now();
    compute_original_block_idx_map(orig_block_idcs, thread_merge_bounds,
                                   nnz_blocks_per_thread);

    convert_bs_to_flattened_es(block_edges, block_index, block_weights,
                               nnz_blocks_after_merge, orig_block_idcs);
    auto agg_end = now();
    agg_edges_time = elapsed_ms(agg_end, agg_start);

    fmt::print("Greedily merged {} blocks into {} blocks\n", n_nonempty_blocks,
               nnz_blocks_after_merge);
    n_nonempty_blocks = nnz_blocks_after_merge;

    preproc_time = rhb_time + flatten_time + agg_edges_time + greedy_merge_time;

    return;
  }


  /**
 * In parallel, recursively partition the graph into blocks whose sidelength is
 * a power of 2. Each block will contain at most max_num_edges;
 */
  void recursive_hilbert_block() {
    // overallocate each thread private vector to avoid push_back
    uint32_t max_bs_per_thread = num_nodes / num_threads;
    for (uint i = 0; i < num_threads; ++i)
      priv_bs[i].resize(max_bs_per_thread);

      // parallel region for recursive subdivision into tasks
#pragma omp parallel num_threads(num_threads) \
        shared(in_csr, bs, priv_bs, block_offsets)
    {
      // only a single thread initiates the first recursive call
#pragma omp single
      {
        // get this thread's thread id - needed to index into the threadprivate
        // vectors
        int tid = omp_get_thread_num();
        recur(0, nn, 0, nn, tid, 0);
      }
      // wait until all tasks (recursive calls) have completed (returned)
#pragma omp taskwait
    }

// since we've overallocated the private block vector, resize them
#pragma omp parallel num_threads(num_threads)
    {
      int tid = omp_get_thread_num();
      priv_bs[tid].resize(block_offsets[tid]);
    }
  }


  /**
 * recursive call used to partition the graph into blocks/subproblems
 * a subproblem is defined by:
 * mn_u - the minimum u/source vertex in this block
 * mx_u - the maximum u/source vertex in this block
 * mn_v - the minimum v/destination vertex in this block
 * mx_v - the maximum v/destination vertex in this block
 *
 * in order to avoid expensive sequential computation for the inital recursive
 * calls, we recur at least to a depth == min_depth
 *
 * At each recursive step, we
 * - split the adjacency matrix (represented using an in-csr) into 4 blocks
 * (quadrants).
 *   - sequentially compute the number of edges in the quadrant
 *     - if a quadrant contains more than `max_num_edges`, recurse
 *     - else, append the block to that thread's private block vector, and
 * return
 *
 * todo a top-down implementation is used since once the max number of edges has
 * been achieved, recursion stops. This was found to be more efficient than a
 * bottom-up approach:
 * one where we recurse to a minimum block sidelength, and
 * sum the number of edges in each quadrant on the way back up the recursive
 * call stack. A bottom-up approach may be more efficient for either:
 * - dense graphs with no empty regions in the adj. mat.
 * - vertex orderings with no empty regions in the adj. mat.
 * @param mn_u
 * @param mx_u
 * @param mn_v
 * @param mv_v
 * @param tid
 * @param curr_depth
 */
  void recur(uint32_t mn_u, uint32_t mx_u, uint32_t mn_v, uint32_t mx_v,
             int tid, uint32_t curr_depth) {
    if (mn_u > num_nodes or mn_v > num_nodes)
      return;// out-of-bounds
    uint32_t len_u = mx_u - mn_u;
    uint32_t len_v = mx_v - mn_v;
    uint32_t md_u = (mn_u + mx_u) / 2;
    uint32_t md_v = (mn_v + mx_v) / 2;
    // first recursive call, recur immediately
    if (len_u == nn) {
      // haven't recursed deep enough (subproblems too expensive), recur
    } else if (curr_depth < min_depth) {
    } else {
      uint32_t num_es_in_block = seq_n_es_in_block(mn_u, mx_u, mn_v, mx_v);
      // this block has a small enough number of edges, add it to this thread's
      // block vector
      if (num_es_in_block == 0)
        return;
      if (num_es_in_block < max_num_edges) {
        uint32_t &b_offset = block_offsets[tid];
        Block &rb = priv_bs[tid][b_offset];
        rb.es.resize(num_es_in_block);

        rb.nnz = num_es_in_block;
        rb.x = mn_u;
        rb.y = mn_v;
        uint32_t v_end = mx_v;
        v_end > num_nodes ? v_end = num_nodes : v_end = v_end;
        rb.side_len = mx_v - mn_v;
        // rb.idx = xy2d(hyperceiling(num_nodes), rb.x, rb.y);
        rb.idx = xy2h(rb.x, rb.y, hilbert_order);// fast hilbert
        populate_and_sort_edges(mn_u, mx_u, mn_v, mx_v, rb, tid);
        b_offset += 1;
        return;
      }
    }

    // split into 4 recursive tasks
    // top left
#pragma omp task shared(in_csr, bs, priv_bs, block_offsets) private(tid)
    {
      tid = omp_get_thread_num();
      recur(mn_u, md_u, mn_v, md_v, tid, curr_depth + 1);
    }

    // top right
#pragma omp task shared(in_csr, bs, priv_bs, block_offsets) private(tid)
    {
      tid = omp_get_thread_num();
      recur(mn_u, md_u, md_v, mx_v, tid, curr_depth + 1);
    }

    // bottom left
#pragma omp task shared(in_csr, bs, priv_bs, block_offsets) private(tid)
    {
      tid = omp_get_thread_num();
      recur(md_u, mx_u, mn_v, md_v, tid, curr_depth + 1);
    }

    // bottom right
#pragma omp task shared(in_csr, bs, priv_bs, block_offsets) private(tid)
    {
      tid = omp_get_thread_num();
      recur(md_u, mx_u, md_v, mx_v, tid, curr_depth + 1);
    }
  }

  /**
 * Sequentially count the number of edges contained in the block defined by the
 * bounds from the input parameters
 *
 * @param mn_u
 * @param mx_u
 * @param mn_v
 * @param mx_v
 * @return
 */
  uint32_t seq_n_es_in_block(uint32_t mn_u, uint32_t mx_u, uint32_t mn_v,
                             uint32_t mx_v) {
    return in_csr.num_edges_in_block(mn_u, mx_u, mn_v, mx_v, max_num_edges);
  }


  /**
 * Once we've identified that a Recursive-Hilbert Block contains a number
 * of edges <= `max_num_edges`, we populate that block's flattened edges
 * array with the edges in that block, assign each edge a hilbert index,
 * and sort the edges by ascending hilbert index
 * @param mn_u
 * @param mx_u
 * @param mn_v
 * @param mx_v
 * @param rb
 */
  void populate_and_sort_edges(uint32_t mn_u, uint32_t mx_u,
                               uint32_t mn_v, uint32_t mx_v,
                               Block &rb, uint tid) {
    uint64_t m = 0;
    // clip the end point
    mx_v > num_nodes ? mx_v = num_nodes : mx_v = mx_v;
    // iterate over the in-neighbourhoods of the vertices in this block
    // and append them to the block's flattened edges array
    for (uint32_t v = mn_v; v < mx_v; ++v) {
      uint64_t start = in_csr.offset(v);
      uint64_t end = in_csr.offset(v + 1);

      if (start == end)
        continue;
      auto [low_idx, high_idx] =
              in_csr.get_first_last_offsets_to_vertices_in_block(start, end, mn_u, mx_u);

      for (uint64_t offset = low_idx; offset < high_idx; offset++) {
        uint32_t u = in_csr.get_vertex(offset);

        auto &edge = rb.es[m].e;
        if constexpr (std::is_same_v<EdgeType, WeightedEdge>) {
          edge.weight = in_csr.get_weight(offset);
        }
        edge.src = u;
        edge.dest = v;
        rb.es[m].h_idx = xy2h(u, v, hilbert_order);
        m++;
      }
    }
    // hilbert order edges in block
    std::sort(rb.es.begin(), rb.es.end(),
              [](const auto &a, const auto &b) -> bool {
                return a.h_idx < b.h_idx;
              });
  }


  // void hilbert_order_edges_in_block(RecursiveBlock &rb) {
  //   // create a temporary vector to store the reordered edges
  //   uint32_t nnz = rb.nnz;
  //   std::vector<IndexedEdge> copy(nnz);
  //   for (uint32_t j = 0; j < nnz; ++j) {
  //     uint32_t src = rb.es[j * 2];
  //     uint32_t dest = rb.es[j * 2 + 1];
  //     copy[j].src = src;
  //     copy[j].dest = dest;
  //     if (weighted()) {
  //       copy[j].wt = rb.wts[j];
  //     }

  //     // block sidelength guaranteed to be a power of 2, so it's safe to use
  //     // the sidelength value when computing the edges' hilbert indices
  //     //                copy[j].h_idx = xy2d(hyperceiling(rb.side_len), src,
  //     //                dest);

  //     // use the hyperceil(num_nodes) as input to xy2d to maintain global
  //     // coherence between ordering of neighbouring blocks

  //     // copy[j].h_idx = xy2d(hyperceiling(num_nodes), src, dest);// hilbert order
  //     copy[j].h_idx = xy2h(src, dest, hilbert_order);// fast hilbert order
  //     // copy[j].h_idx = xy2m(src, dest); // morton order
  //   }
  //   // ips4o::sort(copy.begin(), copy.end(),
  //   std::sort(copy.begin(), copy.end(),
  //             [](const IndexedEdge &a, const IndexedEdge &b) -> bool {
  //               return a.h_idx < b.h_idx;
  //             });
  //   // copy the (now sorted) edges back to the edges array
  //   for (uint32_t j = 0; j < nnz; ++j) {
  //     rb.es[j * 2] = copy[j].src;
  //     rb.es[j * 2 + 1] = copy[j].dest;
  //     if (weighted()) {
  //       rb.wts[j] = copy[j].wt;
  //     }
  //   }
  // }

  /**
 * In parallel, iterate over the blocks of the graph.
 * Hilbert order the edges within block.
 * @param thread_merge_bounds
 * @param nnz_blocks_per_thread
 */
  void hilbert_order_edges_in_blocks(
          std::vector<uint32_t> &thread_merge_bounds,
          std::vector<uint32_t> &nnz_blocks_per_thread) {

    // count the number of nonzero blocks in each thread's range
#pragma omp parallel num_threads(num_threads)
    {
      uint32_t nnz_blocks_for_thread = 0;
      uint tid = omp_get_thread_num();
      uint32_t start = thread_merge_bounds[tid];
      uint32_t end = thread_merge_bounds[tid + 1];
      for (uint32_t i = start; i < end; ++i) {
        if (bs[i].nnz > 0) {
          nnz_blocks_for_thread++;
          // sort the edges within all nonempty blocks
          // hilbert_order_edges_in_block(bs[i]);
          // std::sort(bs[i].es.begin(), bs[i].es.end(),
          //           [](const IdxdEdge &a, const IdxdEdge &b) -> bool {
          //             return a.h_idx < b.h_idx;
          //           });
        }
      }
      nnz_blocks_per_thread[tid + 1] = nnz_blocks_for_thread;
    }
  }


  /**
 * In order to flatten the block edges in parallel, divide up the blocks
 * among the available threads
 * @param orig_block_idcs
 * @param thread_merge_bounds
 * @param nnz_blocks_per_thread
 */
  void compute_original_block_idx_map(
          std::vector<uint32_t> &orig_block_idcs,
          std::vector<uint32_t> &thread_merge_bounds,
          std::vector<uint32_t> &nnz_blocks_per_thread) {
#pragma omp parallel num_threads(num_threads)
    {
      uint32_t nnz_blocks_for_thread = 0;
      uint tid = omp_get_thread_num();
      uint32_t nnz_block_offset = nnz_blocks_per_thread[tid];
      uint32_t start = thread_merge_bounds[tid];
      uint32_t end = thread_merge_bounds[tid + 1];
      for (uint32_t i = start; i < end; ++i) {
        if (bs[i].nnz > 0) {
          orig_block_idcs[nnz_blocks_for_thread + nnz_block_offset] = i;
          nnz_blocks_for_thread++;
        }
      }
    }
  }


  /**
 * After recursive hilbert blocking, iterate over the RHBlocks of the graph,
 * and place the edges of the graph in a flattened array of uint32_t.
 *
 * The edges in block_edges array will be sorted the hilbert curve.
 * The start and end points of the edges that lie in block i are delineated
 * using the block index array.
 * Specifically, the edges of block i start at block_index[i] and end at
 * block_index[i+1]
 * @param block_edges
 * @param block_index
 * @param nnz_blocks_after_merge
 * @param orig_block_idcs
 */
  void convert_bs_to_flattened_es(
          uint32_t *&block_edges, uint32_t *&block_index, double *&block_weights,
          uint32_t nnz_blocks_after_merge, std::vector<uint32_t> &orig_block_idcs) {
    // convert from vector of objects to flattened array with an additional index
    // that  marks the beginning and end of edges within a block
    block_edges = new uint32_t[num_edges * 2]();// flattened
    if constexpr (std::is_same_v<EdgeType, WeightedEdge>)
      block_weights = new double[num_edges]();
    block_index = new uint32_t[nnz_blocks_after_merge + 1]();

    uint64_t cum_block_idx = 0;
    // todo issue with compiling of parallel prefix sum with openmp
    // perform it sequentially
    // #pragma omp parallel
    //     {
    // #pragma omp for reduction(inscan, + : cum_block_idx)
    //       for (uint32_t i = 1; i < nnz_blocks_after_merge + 1; ++i) {
    //         uint32_t n_edges = bs[orig_block_idcs[i - 1]].nnz;
    //         cum_block_idx += n_edges * 2;
    // #pragma omp scan inclusive(cum_block_idx)
    //         block_index[i] = cum_block_idx;
    //       }
    //     }

    for (uint32_t i = 1; i < nnz_blocks_after_merge + 1; ++i) {
      cum_block_idx += bs[orig_block_idcs[i - 1]].nnz * 2;
      block_index[i] = cum_block_idx;
    }

    block_index[nnz_blocks_after_merge] = num_edges * 2;

    // copy over the edges from the block vector to the flattened block_edges
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (uint32_t i = 0; i < nnz_blocks_after_merge; ++i) {
      uint64_t block_edges_start = block_index[i];
      uint64_t block_wts_start = block_index[i] / 2;
      Block &b = bs[orig_block_idcs[i]];
      std::vector<IdxdEdge> &es = b.es;

      for (const auto &edge: es) {
        block_edges[block_edges_start++] = edge.e.src;
        block_edges[block_edges_start++] = edge.e.dest;
        if constexpr (std::is_same_v<EdgeType, WeightedEdge>) {
          block_weights[block_wts_start++] = edge.w.weight;
        }
      }

      // __builtin_memcpy(&block_edges[block_edges_start],
      //  &es.data()[0],
      //  es.size() * sizeof(uint32_t));
    }
  }

  /**
 * After recursive hilbert blocking, some block may contain zero edges
 * To ignore those blocks when flattening the thread-private block vectors,
 * compute the number of nonzero blocks per thread
 *
 */
  void compute_nnz_block_offsets() {
#pragma omp parallel num_threads(num_threads) reduction(+ : n_nonempty_blocks)
    {
      uint tid = omp_get_thread_num();
      uint32_t n_nonempty = 0;
      for (uint32_t j = 0; j < block_offsets[tid]; j++) {
        Block &b = priv_bs[tid][j];
        if (b.nnz > 0) {
          n_nonempty_blocks++;
          n_nonempty++;
        }
      }
      nnz_block_offsets[tid] = n_nonempty;
    }
  }


  /**
 * Flatten the 2D thread-private block vector
 * std::move all thread-private blocks to a global vector of hilbert blocks
 * and sort the blocks by ascending hilbert index
 */
  void flatten_block_vector() {
    bs.resize(n_nonempty_blocks);
    std::vector<uint32_t> cumulative_block_offsets(num_threads + 1, 0);
    uint64_t accum = 0;
    for (int i = 1; i < num_threads + 1; ++i) {
      accum += nnz_block_offsets[i - 1];
      cumulative_block_offsets[i] = accum;
    }

    // using the cumulative, nonzero block offsets, copy blocks from the 2d
    // vector to a flattened version
    // compute the total number of edges assigned within to all blocks
    // for verification this number should be equal to the number of edges in the
    // graph
#pragma omp parallel num_threads(num_threads) reduction(+ : total_es_assigned)
    {
      int tid = omp_get_thread_num();
      uint64_t k = cumulative_block_offsets[tid];
      for (uint32_t j = 0; j < block_offsets[tid]; j++) {
        Block &rb = priv_bs[tid][j];
        if (rb.nnz > 0) {
          Block &b = bs[k];
          bs[k] = std::move(priv_bs[tid][j]);
          // b.x = rb.x;
          // b.y = rb.y;
          // b.idx = rb.idx;
          // b.side_len = rb.side_len;
          // b.nnz = rb.nnz;

          // b.es = std::move(rb.es);
          // if (weighted())
          //   b.wts = std::move(rb.wts);
          k++;
        }
      }
      // erase all of thread-private recursive blocks
    }

    // sort the blocks by ascending hilbert index
    std::sort(dpl::execution::par_unseq,
              bs.begin(), bs.end(),
              [](const Block &a, const Block &b) -> bool {
                return a.idx < b.idx;
              });
  }


  /**
 * Write the block array to a text file
 * For each block, write its:
 * - x, y coordinates,
 * - sidelength
 * - hilbert index
 * - number of nonzeros
 * @param out_path
 */
  void write_blocks_metadata(std::string out_path) {
    std::ofstream output_file(out_path);
    for (const auto &b: bs) {
      output_file << fmt::format("{} {} {} {} {} \n", b.x, b.y, b.side_len, b.idx,
                                 b.nnz);
    }
    output_file.close();
  }

  /**
 * after sorting by hilbert index both:
 *   - the recursive hilbert blocks
 *   - the edges within each block
 * write all the edges within all the blocks
 * (used for verification of correctness of hilbert ordering of edges)
 * @param path
 */
  void write_edges(std::string out_path) {
    uint64_t global_h_idx = 0;
    std::ofstream output_file(out_path);
    for (uint32_t i = 0; i < n_nonempty_blocks; ++i) {
      Block &b = bs[i];
      //    uint32_t *es = b.es;
      std::vector<uint32_t> es;
      for (uint32_t j = 0; j < b.nnz; ++j) {
        output_file << b.es[j].e.src << " " << b.es[j].e.dest << " " << global_h_idx
                    << "\n";
        global_h_idx++;
      }
    }
    output_file.close();
  }

  /**
 * Writes the flattened edge vector to a text file
 * Writes the block index to a text file
 * @param edges_path
 * @param index_path
 * @param block_edges
 * @param block_index
 * @param num_blocks
 */
  void write_block_edges(std::string edges_path, std::string index_path,
                         uint32_t *&block_edges, uint32_t *&block_index,
                         uint32_t num_blocks) {
    std::ofstream efile(edges_path);
    std::ofstream ifile(index_path);

    ifile << "0"
          << "\n";
    for (uint32_t i = 0; i < num_blocks; ++i) {
      uint64_t block_edges_start = block_index[i];
      uint64_t block_edges_end = block_index[i + 1];
      ifile << block_edges_end << "\n";
      for (uint64_t j = block_edges_start; j < block_edges_end; j += 2) {
        uint32_t src = block_edges[j];
        uint32_t dest = block_edges[j + 1];
        efile << src << " " << dest << "\n";
      }
    }

    efile.close();
    ifile.close();
  }

  /**
 * @brief Print the execution time breakdown of Recursive Hilbert Blocking.
 *
 * @param graph_name
 */
  void info(std::string graph_name) {
    fmt::print("Rhubarb partitioned {} into {} blocks\n", graph_name,
               n_nonempty_blocks);
    fmt::print("Total Preprocessing Time: {} ms \n"
               "| Recursive Hilbert Blocking: {} "
               "| Flattening Thread Private Blocks: {} "
               "| Greedy Block Merge: {} "
               "| Edge aggregation: {} |"
               "\n",
               preproc_time, rhb_time, flatten_time, greedy_merge_time,
               agg_edges_time);
  }

  void release_resources() {
    block_offsets.clear();
    block_offsets.shrink_to_fit();
    nnz_block_offsets.clear();
    nnz_block_offsets.shrink_to_fit();

    std::vector<Block>().swap(bs);
    bs.clear();
    bs.shrink_to_fit();
    for (auto &vec: priv_bs) {
      std::vector<Block>().swap(vec);
      vec.clear();
      vec.shrink_to_fit();
    }
    std::vector<std::vector<Block>>().swap(priv_bs);
    priv_bs.clear();
    priv_bs.shrink_to_fit();
  }
};


#endif// RHUBARB_RHUBARB_H
