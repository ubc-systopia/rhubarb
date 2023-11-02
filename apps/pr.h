
#include "RecursiveBlock.h"
#include "fmt/core.h"
#include "spray.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <oneapi/dpl/execution>
#include <vector>
#include <likwid-marker.h>
#include <likwid.h>
#include "util.h"


#if DOUBLE_SCORE_T == 1
using ScoreT = double;
#else 
using ScoreT = float;
#endif


inline void block_map(ScoreT *&incoming_total, ScoreT *&outgoing_contrib,
                      uint32_t num_blocks, uint32_t n,
                      uint32_t *&block_edges, uint32_t *&block_index,
                      uint32_t dynamic_group_size, uint num_threads,
                      uint32_t totalsize) {
  BlockReductionBWIDTH<ScoreT> x_p(totalsize, incoming_total);

#pragma omp parallel reduction(+ : x_p) num_threads(num_threads)
  {
    LIKWID_MARKER_START("pr");
#pragma omp for schedule(nonmonotonic:dynamic, dynamic_group_size)
    for (uint32_t i = 0; i < num_blocks; ++i) {
      uint64_t block_edges_start = block_index[i];
      uint64_t block_edges_end = block_index[i + 1];
      for (uint64_t j = block_edges_start; j < block_edges_end; j += 2) {
        x_p[block_edges[j + 1]] += outgoing_contrib[block_edges[j]];
      }
    }
    LIKWID_MARKER_STOP("pr");
  }
}

inline void vertex_map(ScoreT *&incoming_total, ScoreT *&outgoing_contrib,
                       ScoreT *&scores, uint32_t n, ScoreT base_score,
                       ScoreT alpha, std::vector<ScoreT> &inv_degs,
                       uint num_threads) {
#pragma omp parallel for simd schedule(static) num_threads(num_threads)
  for (uint32_t i = 0; i < n; i++) {
    scores[i] = base_score + alpha * incoming_total[i];
    outgoing_contrib[i] = scores[i] * inv_degs[i];
    incoming_total[i] = 0;
  }
}

void zero(ScoreT *&incoming_total, ScoreT *&outgoing_contrib, ScoreT *&scores,
          uint32_t n) {
#pragma omp parallel for
  for (uint32_t i = 0; i < n; ++i) {
    incoming_total[i] = 0;
    outgoing_contrib[i] = 0;
    scores[i] = 0;
  }
}

inline uint64_t
pr_compute(std::vector<uint32_t> &out_degs,
           uint32_t n, uint32_t num_blocks,
           uint num_iterations, uint32_t *&block_edges, uint32_t *&block_index,
           uint32_t dynamic_group_size, uint num_threads, ScoreT *&scores) {

  uint32_t totalsize = ((n / BWIDTH) + 1) * BWIDTH;
  fmt::print("totalsize: {}\n", totalsize);
  ScoreT *incoming_total = new ScoreT[totalsize]();
  ScoreT *outgoing_contrib = new ScoreT[n]();
  
  const ScoreT alpha = 0.85;
  ScoreT base_score = (1.0f - alpha) / n;

  std::vector<ScoreT> inv_degs(n);
#pragma omp parallel for
  for (uint32_t i = 0; i < n; ++i) {
    inv_degs[i] = 1.0f / ScoreT(out_degs[i]);
  }


  //  uint32_t num_blocks = bs.size();
  zero(incoming_total, outgoing_contrib, scores, n);
//  fmt::print("PageRank init complete. Starting {} iterations.\n", num_iterations);
  auto start_time = std::chrono::high_resolution_clock::now();
  for (uint iter = 0; iter < num_iterations; ++iter) {
    block_map(incoming_total, outgoing_contrib, num_blocks, n, block_edges,
              block_index, dynamic_group_size, num_threads, totalsize);
    vertex_map(incoming_total, outgoing_contrib, scores, n, base_score, alpha,
               inv_degs, num_threads);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  uint64_t runtime = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();
  // fmt::print("runtime: {}\n", runtime);
  delete[] incoming_total;
  delete[] outgoing_contrib;
  return runtime;
}

template <typename ScoreT>
bool pr_verif(ScoreT *&scores, std::vector<double> &pr, uint32_t n,
              std::vector<uint32_t> &iso_map) {
  std::vector<ScoreT> mapped_results(n);
#pragma omp parallel for
  for (uint32_t i = 0; i < n; ++i) {
    mapped_results[i] = scores[iso_map[i]];
  }
  bool valid_pr = std::equal(dpl::execution::par_unseq, mapped_results.begin(),
                             mapped_results.end(), pr.begin(),
                             [](double value1, double value2) {
                               constexpr double epsilon = 1e-4;
                               return std::fabs(value1 - value2) < epsilon;
                             });

  return valid_pr;
}