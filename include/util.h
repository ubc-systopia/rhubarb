//
// Created by atrostan on 29/03/23.
//
#include "CSR.h"
#include "WeightedEdge.h"
#include "fmt/core.h"
#include "spray.hpp"
#include <cstdint>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <queue>
#include <vector>

#ifndef RHUBARB_UTIL_H
#define RHUBARB_UTIL_H

/**
 * @brief Traverse an edgelist in parallel, and return the maximum vertex id present in the edgelist
 * 
 * @tparam Edge either a weighted or unweighted edge
 * @param edge_list 
 * @return uint32_t 
 */
template<typename Edge>
uint32_t get_max_vertex_id(std::vector<Edge> &es) {
  uint64_t m = es.size();
  uint32_t max_id = 0;
#pragma omp parallel for reduction(max : max_id)
  for (uint64_t i = 0; i < m; ++i) {

    if constexpr (std::is_same_v<Edge, WeightedEdge>) {
      max_id = std::max({es[i].src, es[i].dest, max_id});
    } else {
      max_id = std::max({es[i].first, es[i].second, max_id});
    }
  }
  return max_id;
}
void check_valid_block_width(uint32_t n);
bool invalid_block_width(uint32_t num_vertices);
uint32_t largest_valid_block_width(uint32_t num_vertices);
void get_degs_from_csrs(std::vector<uint32_t> &in_degs, CSR &in_csr,
                        std::vector<uint32_t> &out_degs, CSR &out_csr);
uint64_t hyperceiling(uint32_t n);
void rot(int64_t n, int64_t *x, int64_t *y, int64_t rx, int64_t ry);
int64_t xy2d(int64_t n, int64_t x, int64_t y);
uint64_t split_by_3(uint32_t a);
uint64_t xy2m(uint32_t x, uint32_t y);
template<typename Edge>
void create_bipartite_csrs(std::vector<Edge> &es,
                           std::string graph_dir,
                           uint32_t n1, uint32_t n2) {

  uint32_t N = get_max_vertex_id<Edge>(es) + 1;
  fmt::print("N: {}\n", N);
  uint32_t *out_degs = new uint32_t[N]();
  uint32_t *in_degs = new uint32_t[N]();
  uint64_t m = es.size();
  CSR in_csr = CSR(N, m, true, true);
  CSR out_csr = CSR(N, m, false, true);
  // create the weighted in-csr and out-csr representations
  par_populate_csrs(es, n1, n2, N, out_degs, in_degs, in_csr, out_csr);

  std::string out_csr_path = fmt::format("{}/{}", graph_dir, "out_csr.bin");
  std::string in_csr_path = fmt::format("{}/{}", graph_dir, "in_csr.bin");
  std::string gmat_text_edge_list_path = fmt::format("{}/{}", graph_dir, "graphmat_comp.net0");

  out_csr.write_as_binary(out_csr_path);
  in_csr.write_as_binary(in_csr_path);

  std::string ligra_adj_path = fmt::format("{}/orig_ligra_adj", graph_dir);
  out_csr.write_ligra_adj(ligra_adj_path);

  // also write an edge list whose first vertex id == 1 (required for GraphMat)
  fmt::print("Writing 1+text edge list to: {}..\n", gmat_text_edge_list_path);
  // std::ofstream gmat_outfile(gmat_text_edge_list_path);
  out_csr.write_as_text_edge_list(gmat_text_edge_list_path, 1);

  // bipartite_cm(in_csr, out_csr, out_degs, in_degs, n1, n2, m, es);
  // clean up
  delete[] in_degs;
  delete[] out_degs;
}


template<typename Edge>
void par_populate_csrs(std::vector<Edge> &es,
                       uint32_t n1, uint32_t n2, uint32_t N,
                       uint32_t *&out_degs,
                       uint32_t *&in_degs,
                       CSR &in_csr,
                       CSR &out_csr) {
  // uint32_t mx_in = 0, mx_out = 0;

  uint32_t reduce_arr_size = ((N / BWIDTH) + 1) * BWIDTH;

  auto [mx_in, mx_out] = par_compute_degrees(in_degs, out_degs, es, N, reduce_arr_size);
  fmt::print("mx_in, mx_out: {} {}\n", mx_in, mx_out);
  // sort by source
  std::sort(dpl::execution::par_unseq, es.begin(), es.end(),
            [](const Edge &l, const Edge &r) -> bool {
              if (l.src == r.src) {
                return l.dest < r.dest;
              }
              return l.src < r.src;
            });

  out_csr.par_populate(out_degs, es);

  // sort by dest
  std::sort(dpl::execution::par_unseq, es.begin(), es.end(),
            [](const Edge &l, const Edge &r) -> bool {
              if (l.dest == r.dest) {
                return l.src < r.src;
              }
              return l.dest < r.dest;
            });
  in_csr.par_populate(in_degs, es);
}
void undirected_cuthill_mckee(CSR &in_csr, CSR &out_csr, uint32_t n1,
                              std::vector<uint32_t> &cm_map);
template<typename Edge>
void bipartite_cm(CSR &in_csr, CSR &out_csr,
                  uint32_t *&out_degs, uint32_t *&in_degs,
                  uint32_t n1, uint32_t n2, uint64_t m,
                  std::vector<Edge> &es) {
  using E = std::pair<uint32_t, uint32_t>;
  // sort V1, V2 by ascending degree, separately
  std::vector<E> V1(n1);
  std::vector<E> V2(n2);
#pragma omp parallel for
  for (uint32_t u = 0; u < n1; ++u)
    V1[u] = {u, out_degs[u]};

#pragma omp parallel for
  for (uint32_t v = n1; v < n1 + n2; ++v)
    V2[v - n1] = {v, in_degs[v]};

  std::sort(dpl::execution::par_unseq,
            V1.begin(),
            V1.end(),
            [](const E &a,
               const E &b) -> bool {
              if (a.second == b.second)
                return a.first < b.first;
              return a.second > b.second;
            });

  std::sort(dpl::execution::par_unseq,
            V2.begin(),
            V2.end(),
            [](const E &a,
               const E &b) -> bool {
              if (a.second == b.second)
                return a.first < b.first;
              return a.second > b.second;
            });

  // create the ascending degree sort isomorphism maps for the two vertex sets
  std::vector<uint32_t> iso_map(n1 + n2);
#pragma omp parallel for
  for (uint32_t i = 0; i < n1; ++i) {
    iso_map[V1[i].first] = i;
  }

#pragma omp parallel for
  for (uint32_t i = n1; i < n1 + n2; ++i) {
    iso_map[V2[i - n1].first] = i;
  }

  // translate the edgelist
#pragma omp parallel for
  for (uint64_t i = 0; i < m; ++i) {
    es[i].src = iso_map[es[i].src];
    es[i].dest = iso_map[es[i].dest];
    // weight remains unchanged
  }
  uint32_t N = n1 + n2;
  uint32_t *mapped_out_degs = new uint32_t[N]();
  uint32_t *mapped_in_degs = new uint32_t[N]();
  CSR mapped_in_csr = CSR(N, m, true, true);
  CSR mapped_out_csr = CSR(N, m, false, true);
  // create the weighted in-csr and out-csr representations
  par_populate_csrs<Edge>(es, n1, n2, N, mapped_out_degs, mapped_in_degs, mapped_in_csr, mapped_out_csr);

  // initialize the vertex order map
  std::vector<uint32_t> cm_map(n1 + n2, 0);

  //  vertex_set_bfs(mapped_in_csr, mapped_out_csr, n1, cm_map, true);
  //  fmt::print("cm_map: {}\n", cm_map);
  undirected_cuthill_mckee(mapped_in_csr, mapped_out_csr, n1, cm_map);
  //#pragma omp parallel for
  //  for (uint32_t v = n1; v < n1 + n2; ++v)
}


void vertex_set_bfs(CSR &in_csr, CSR &out_csr, uint32_t n1,
                    std::vector<uint32_t> &cm_map, bool v1);


template<typename contentType>
using BlockReductionBWIDTH = spray::AlignedBlockReduction<contentType, BWIDTH>;
#pragma omp declare reduction(                                          \
                + : BlockReductionBWIDTH<float> : BlockReductionBWIDTH< \
                                float>::ompReduce(&omp_out, &omp_in))   \
        initializer(BlockReductionBWIDTH<float>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(                                           \
                + : BlockReductionBWIDTH<double> : BlockReductionBWIDTH< \
                                double>::ompReduce(&omp_out, &omp_in))   \
        initializer(BlockReductionBWIDTH<double>::ompInit(&omp_priv, &omp_orig))


#pragma omp declare reduction(                                             \
                + : BlockReductionBWIDTH<uint32_t> : BlockReductionBWIDTH< \
                                uint32_t>::ompReduce(&omp_out, &omp_in))   \
        initializer(BlockReductionBWIDTH<uint32_t>::ompInit(&omp_priv, &omp_orig))

/**
 * Sort and remove duplicates from a std::vector<V>
 * @tparam T
 * @param v
 */
template<typename T>
void sort_and_remove_dups(std::vector<T> &v) {
  std::sort(dpl::execution::par_unseq, v.begin(), v.end());
  v.erase(std::unique(dpl::execution::par_unseq, v.begin(), v.end()), v.end());
}

/**
 * @brief measure elapsed time for the execution of a function
 * 
 * @tparam TimeDuration type of std::chrono unit to use (e.g. nano, milli)
 * @tparam Func 
 * @tparam Args 
 * @param func 
 * @param args 
 * @return auto either:
 * the duration it took to complete the execution of func, or 
 * a tuple of <return value, duration>
 */
template<typename TimeUnit, typename Func, typename... Args>
auto time_function_invocation(Func func, Args &&...args) {
  using FuncReturnType = decltype(std::invoke(func, std::forward<Args>(args)...));

  const auto start = std::chrono::high_resolution_clock::now();
  if constexpr (std::is_same_v<void, FuncReturnType>) {
    std::invoke(func, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<TimeUnit>(end - start);
  } else {
    FuncReturnType ret = std::invoke(func, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::tuple<FuncReturnType, TimeUnit>{
            ret, std::chrono::duration_cast<TimeUnit>(end - start)};
  }
}

/**
 * @brief Use an omp parallel max reduction to get the max value in array of T
 * 
 * @tparam T 
 * @param arr 
 * @param size 
 * @return T 
 */
template<typename T>
T par_max(T *arr, uint32_t size) {
  T mx = std::numeric_limits<T>::min();
#pragma omp parallel for reduction(max : mx)
  for (uint32_t i = 0; i < size; ++i) {
    mx = std::max(mx, arr[i]);
  }
  return mx;
}

/**
 * Use OpenMP Sparse Array Reductions (spray) to compute the in, out degrees
 * of the graph stored in edge_list
 *
 * return the max in, out degree of the graph
 * @param in
 * @param out
 * @param edge_list
 * @param n
 * @param mx_in
 * @param mx_out
 */
template<typename Edge>
std::pair<uint32_t, uint32_t>
par_compute_degrees(uint32_t *&in, uint32_t *&out,
                    std::vector<Edge> &edge_list,
                    uint32_t n, uint32_t reduce_arr_size) {
  BlockReductionBWIDTH<uint32_t> in_reduction(reduce_arr_size, in);
  BlockReductionBWIDTH<uint32_t> out_reduction(reduce_arr_size, out);

#pragma omp parallel for schedule(static) reduction(+ : in_reduction) \
        reduction(+ : out_reduction)
  for (uint32_t i = 0; i < edge_list.size(); ++i) {
    uint32_t u, v;

    if constexpr (std::is_same_v<Edge, WeightedEdge>) {
      u = edge_list[i].src;
      v = edge_list[i].dest;
    } else {
      u = edge_list[i].first;
      v = edge_list[i].second;
    }
    out_reduction[u]++;
    in_reduction[v]++;
  }

  return {par_max(in, n), par_max(out, n)};
}

/**
 * @brief Print the contents of an array 
 * 
 * @tparam T 
 * @param arr 
 * @param size 
 */
template<typename T>
void print_arr(const T *arr, const size_t n) {
  fmt::print("[");
  for (size_t i = 0; i < n - 1; ++i) {
    fmt::print("{}, ", arr[i]);
  }

  fmt::print("{}]\n", arr[n - 1]);
}


#endif//RHUBARB_UTIL_H
