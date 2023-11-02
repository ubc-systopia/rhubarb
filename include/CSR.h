//
// Created by atrostan on 29/03/23.
//

#include "WeightedEdge.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include <fstream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#ifndef RHUBARB_CSR_H
#define RHUBARB_CSR_H

/**
 * @brief A compressed sparse row representation of a directed graph
 * 
 */
class CSR {
  public:
  // empty constructor
  CSR() {
    index = nullptr;
    neighbours = nullptr;
    weights = nullptr;
    n = -1;// number of vertices
    m = -1;// number of edges
    in = false;
    weighted = false;
  }

  CSR(uint32_t n_, uint64_t m_, bool i, bool w = false) {
    in = i;
    n = n_;
    m = m_;
    index = new uint64_t[n + 1]();
    neighbours = new uint32_t[m]();
    weighted = w;
    if (weighted)
      weights = new double[m]();
  }

  // don't want this to be copied, too much data to move
  CSR(const CSR &other) = delete;

  // prefer move because too much data to copy
  CSR(CSR &&other)
      : index(other.index),
        neighbours(other.neighbours),
        weights(other.weights),
        n(other.n),
        m(other.m),
        in(other.in),
        weighted(other.weighted) {
    other.index = nullptr;
    other.neighbours = nullptr;
    other.weights = nullptr;
    other.n = -1;
    other.m = -1;
    other.in = false;
    other.weighted = false;
  }

  // want move assignment
  CSR &operator=(CSR &&c) {
    if (this != &c) {
      index = c.index;
      neighbours = c.neighbours;
      weights = c.weights;
      n = c.n;
      m = c.m;
      in = c.in;
      weighted = c.weighted;
      c.index = nullptr;
      c.neighbours = nullptr;
      c.weights = nullptr;
    }
    return *this;
  }

  /**
   * Class constructor
   * given a path to a binary CSR, and a direction flag
   * Compressed Sparse Row (Push-dir) vs. CS Column (Pull-dir)
   * @param in_path
   * @param i
   */
  CSR(std::string in_path, bool i, bool w = false) {
    fmt::print("Reading CSR from {}\n", in_path);
    std::ifstream infile(in_path, std::ios::binary | std::ios::in);
    uint32_t n_;
    uint64_t m_;
    infile.read(reinterpret_cast<char *>(&n_), sizeof(uint32_t));
    infile.read(reinterpret_cast<char *>(&m_), sizeof(uint64_t));

    weighted = w;
    in = i;
    n = n_;
    m = m_;

    this->index = new uint64_t[n + 1];
    this->neighbours = new uint32_t[m];

    if (weighted)
      this->weights = new double[m];

    infile.read(reinterpret_cast<char *>(this->index),
                (n + 1) * sizeof(uint64_t));

    infile.read(reinterpret_cast<char *>(this->neighbours),
                (m) * sizeof(uint32_t));

    if (weighted) {
      infile.read(reinterpret_cast<char *>(this->weights),
                  (m) * sizeof(double));
    }

    infile.close();
  }

  ~CSR() {
    release_resources();
  }

  void release_resources() {
    if (index != nullptr)
      delete[] index;

    if (neighbours != nullptr)
      delete[] neighbours;

    if (weights != nullptr)
      delete[] weights;
  }

  void print_contents() {
    fmt::print("Index: [");
    for (uint32_t i = 0; i < n; ++i) {
      fmt::print("{}, ", index[i]);
    }
    fmt::print("]\n");
    fmt::print("Neighbours: [");
    for (uint32_t i = 0; i < m; ++i) {

      if (weighted) {
        fmt::print("{} {}, ", neighbours[i], weights[i]);
      } else {
        fmt::print("{}, ", neighbours[i]);
      }
    }
    fmt::print("]\n");
  }

  void write_degrees(std::string out_path) {
    fmt::print("Writing degrees to {}\n", out_path);
    std::string line;
    std::ofstream output_file(out_path);
    if (output_file.is_open()) {
      for (uint32_t u = 0; u < n; ++u) {
        uint64_t start = index[u];
        uint64_t end = index[u + 1];
        output_file << end - start << "\n";
      }
      output_file.close();
    } else
      fmt::print("Unable to open file\n");
  }

  /**
   * @brief 
   * Write the CSR in ligra's adjacency graph format
   * https://github.com/jshun/ligra#input-format-for-ligra-applications
   * 
   * @param out_path 
   */
  void write_ligra_adj(std::string out_path) {
    fmt::print("Writing csr as adjacency graph to {}\n", out_path);
    std::string line;
    // flat edges' size has been preallocated and initialized
    std::ofstream output_file(out_path);
    if (output_file.is_open()) {
      if (weighted) {
        output_file << "WeightedAdjacencyGraph"
                    << "\n";
      } else {
        output_file << "AdjacencyGraph"
                    << "\n";
      }

      output_file << n << "\n";
      output_file << m << "\n";
      for (uint32_t i = 0; i < n; ++i) {
        output_file << index[i] << "\n";
      }
      for (uint32_t i = 0; i < m; ++i) {
        output_file << neighbours[i] << "\n";
      }
      if (weighted) {
        for (uint32_t i = 0; i < m; ++i) {

          output_file << weights[i]
                      << "\n";
        }
      }
      output_file.close();
    } else
      fmt::print("Unable to open file\n");
  }


  /**
   * write the csr's data to a binary file
   * @param out_path
   */
  void
  write_as_binary(std::string out_path) {
    std::ofstream out(out_path,
                      std::ios::binary | std::ios::out | std::ios::trunc);
    fmt::print("Writing csr to {}\n", out_path);
    uint32_t n_ = n;
    uint64_t m_ = m;
    out.write(reinterpret_cast<char *>(&n_), sizeof(uint32_t));
    out.write(reinterpret_cast<char *>(&m_), sizeof(uint64_t));
    out.write(reinterpret_cast<const char *>(this->index),
              (n_ + 1) * sizeof(uint64_t));
    out.write(reinterpret_cast<const char *>(this->neighbours),
              m_ * sizeof(uint32_t));
    if (weighted) {
      out.write(reinterpret_cast<const char *>(this->weights),
                m_ * sizeof(double));
    }

    out.close();
  }

  /**
   * write the csr's data as an edgelist
   * @param out_path
   */
  void write_as_text_edge_list(std::string out_path,
                               uint vertex_id_offset = 0) {
    fmt::print("Writing edgelist to {}\n", out_path);
    std::string line;
    // flat edges' size has been preallocated and initialized
    std::ofstream output_file(out_path);
    if (output_file.is_open()) {
      for (uint32_t u = 0; u < n; ++u) {
        uint64_t start = index[u];
        uint64_t end = index[u + 1];
        for (uint64_t offset = start; offset < end; ++offset) {
          uint32_t v = neighbours[offset];

          if (weighted) {
            double w = weights[offset];
            output_file << vertex_id_offset + u
                        << " "
                        << vertex_id_offset + v
                        << " "
                        << w
                        << "\n";
          } else {
            output_file << vertex_id_offset + u
                        << " "
                        << vertex_id_offset + v
                        << "\n";
          }
        }
      }
      output_file.close();
    } else
      fmt::print("Unable to open file\n");
  }

  /**
   * 
   * Given an array of out/in degrees and a sorted edgelist, populate CSR in 
   * parallel
   * Input edge list assumed to be sorted by either:
   * (src, dest) -> for out-csr
   * (dest, src) -> for in-csr
   * Correspondingly, input degrees array assumed to be either: out/in degrees.
   * @param degrees
   * @param edge_list
   */
  void par_populate(std::vector<uint32_t> &degrees,
                    std::vector<std::pair<uint32_t, uint32_t>> &edge_list) {
    uint64_t offset = 0;

    std::exclusive_scan(
            dpl::execution::par_unseq, degrees.begin(), degrees.end(), index, 0,
            [](uint32_t l, uint32_t r) -> uint32_t { return l + r; });

#pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < n; ++u) {
      uint32_t degree = degrees[u];
      uint32_t neigh_count = 0;
      for (uint32_t i = 0; i < degree; ++i) {
        uint32_t v = -1;
        if (in) {
          v = edge_list[index[u] + i].first;
        } else {
          v = edge_list[index[u] + i].second;
        }
        neighbours[index[u] + neigh_count] = v;
        ++neigh_count;
      }
    }
    index[n] = m;
  }

  /**
   * @brief Get the max degree in the CSR
   * an in-csr (CSC) returns the max in-degree,
   * an out-csr (CSR) returns the max out-degree
   * 
   * @return uint32_t 
   */
  uint32_t max_degree() {
    uint32_t max_degree = 0;
#pragma omp parallel for reduction(max : max_degree)
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t deg = index[i + 1] - index[i];
      if (deg > max_degree) max_degree = deg;
    }

    return max_degree;
  }

  /**
   * Populate in parallel the Compressed sparse row (col) representation
   *
   * @param degrees array of either vertex in or out-degrees
   * @param edge_list A correspondingly sorted (either by ascending <src, dest>
   * or <dest, src> pairs) edgelist
   */
  void par_populate(uint32_t *&degrees,
                    std::vector<std::pair<uint32_t, uint32_t>> edge_list) {
    uint64_t offset = 0;
    // run an inclusive scan on the index array
    uint64_t cumulative_deg = 0;
    index[0] = 0;

    // store the prefix sum of degrees into the index array
#pragma omp parallel for reduction(inscan, + : cumulative_deg)
    for (uint32_t i = 1; i < n + 1; ++i) {
      cumulative_deg += degrees[i - 1];
#pragma omp scan inclusive(cumulative_deg)
      index[i] = cumulative_deg;
    }

#pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < n; ++u) {
      uint32_t degree = degrees[u];
      uint32_t neigh_count = 0;
      for (uint32_t i = 0; i < degree; ++i) {
        uint32_t v = -1;
        if (in) {
          v = edge_list[index[u] + i].first;
        } else {
          v = edge_list[index[u] + i].second;
        }
        neighbours[index[u] + neigh_count] = v;
        ++neigh_count;
      }
    }
    index[n] = m;
  }

  /**
   * Get the ids of the top k highest in/out-degree (depending on the type
   * of csr - in/out) vertices in the graph
   * @param k
   */
  std::vector<uint32_t> top_k_hubs(uint32_t k) {
    std::vector<std::pair<uint32_t, uint32_t>> degrees(n);
#pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i)
      degrees[i] = {i, index[i + 1] - index[i]};

    std::sort(dpl::execution::par_unseq,
              degrees.begin(),
              degrees.end(),
              [](const std::pair<uint32_t, uint32_t> &a,
                 const std::pair<uint32_t, uint32_t> &b) -> bool {
                if (a.second == b.second)
                  return a.first < b.first;
                return a.second > b.second;
              });

    std::vector<uint32_t> top_k(k);
    for (uint32_t i = 0; i < k; ++i) {
      top_k[i] = degrees[i].first;
    }
    return top_k;
  }

  void par_populate(uint32_t *&degrees, std::vector<WeightedEdge> &edge_list) {
    uint64_t offset = 0;
    // run an inclusive scan on the index array
    uint64_t cumulative_deg = 0;
    index[0] = 0;

#pragma omp parallel for reduction(inscan, + : cumulative_deg)
    for (uint32_t i = 1; i < n + 1; ++i) {
      cumulative_deg += degrees[i - 1];
#pragma omp scan inclusive(cumulative_deg)
      index[i] = cumulative_deg;
    }

#pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < n; ++u) {
      uint32_t degree = degrees[u];
      uint32_t neigh_count = 0;
      for (uint32_t i = 0; i < degree; ++i) {
        uint32_t v = -1;
        if (in) {
          v = edge_list[index[u] + i].src;
        } else {
          v = edge_list[index[u] + i].dest;
        }
        double w = edge_list[index[u] + i].weight;
        neighbours[index[u] + neigh_count] = v;
        weights[index[u] + neigh_count] = w;
        ++neigh_count;
      }
    }
    index[n] = m;
  }

  uint32_t num_nodes() { return n; }
  uint64_t num_edges() { return m; }

  uint32_t degree(uint32_t u) { return index[u + 1] - index[u]; }

  /**
   * @brief Get the first and last offsets to vertices in block object
   * 
   * @param start offset to the beginning of a vertex's neighbourhood
   * @param end offset to the end of a vertex's neighbourhood
   * @param mn the smallest vertex ID that belongs in this block
   * @param mx the largest vertex ID that belongs in this block
   * @return std::pair<uint64_t, uint64_t> 
   */
  std::pair<uint64_t, uint64_t>
  get_first_last_offsets_to_vertices_in_block(uint64_t start, uint64_t end,
                                              uint32_t mn, uint32_t mx) {
    uint32_t *low = std::lower_bound(neighbours + start,
                                     neighbours + end, mn);
    uint64_t low_idx = low - neighbours;

    uint32_t *high = std::lower_bound(neighbours + low_idx,
                                      neighbours + end, mx);
    uint64_t high_idx = high - neighbours;

    return {low_idx, high_idx};
  }


  /**
   * @brief Sequentially count the number of edges contained in the block 
   * defined by the bounds from the input parameters
   *
   * Iterates over the destination vertices defined by the range [mn_v, mx_v)
   * Uses two binary searches to find the indices into the in-csr's neighbour
   * array that correspond to mn_u, mx_u
   * The difference between the max, min index is the number of in-edges in this
   * block
   * 
   * @param mn_u 
   * @param mx_u 
   * @param mn_v 
   * @param mx_v 
   * @param max_num_edges 
   * @return uint32_t 
   */
  uint32_t num_edges_in_block(uint32_t mn_u, uint32_t mx_u,
                              uint32_t mn_v, uint32_t mx_v,
                              uint32_t max_num_edges) {
    uint32_t m = 0;
    // clip the end point
    mx_v > n ? mx_v = n : mx_v = mx_v;
    for (uint32_t v = mn_v; v < mx_v; ++v) {
      uint64_t start = index[v];
      uint64_t end = index[v + 1];
      if (start == end)
        continue;
      auto [low_idx, high_idx] =
              get_first_last_offsets_to_vertices_in_block(start, end, mn_u, mx_u);
      m += high_idx - low_idx;

      if (m > max_num_edges) {
        return m;
      }// too many edges in this block, recur further
    }
    return m;
  }

  uint64_t offset(uint32_t u) { return index[u]; }
  uint32_t get_vertex(uint64_t offset) { return neighbours[offset]; }
  double get_weight(uint64_t offset) { return weights[offset]; }
  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    uint32_t n_;
    uint32_t *neighbours_;
    uint64_t *index_;
    uint64_t start_offset_;

public:
    Neighborhood(uint32_t n, uint64_t *index, uint32_t *neighbours,
                 uint64_t start_offset) : n_(n), start_offset_(0),
                                          index_(index), neighbours_(neighbours) {
      uint64_t max_offset = end() - begin();
      start_offset_ = std::min(start_offset, max_offset);
    }
    typedef uint32_t *iterator;
    iterator begin() { return neighbours_ + index_[n_]; }
    iterator end() { return neighbours_ + index_[n_ + 1]; }
  };
  Neighborhood neigh(uint32_t n, uint64_t start_offset = 0) const {
    return Neighborhood(n, index, neighbours, start_offset);
  }

  /**
 * Return a pair of the average read/write distance of an edge orderings of a graph
 *
 * If the CSR is in-csr (CSC), returns the average rw distance for pull direction
 * iteration, or push direction for CSR.
 *
 * Average read distance:
 * The sum of absolute source ID difference between each pair of consecutive edges
 * / number of edges
 *
 * Average write distance:
 * The sum of absolute destination ID difference between each pair of consecutive
 * edges / number of edges
 *
 * @param csr a csr of a graph
 * @return
 */
  std::pair<double, double> avg_rw_dist() {
    double avg_u_dist = double(n) / m;

    uint64_t v_dist = 0;
#pragma omp parallel for reduction(+ : v_dist)
    for (uint32_t u = 0; u < n; u++) {
      uint64_t start = index[u];
      uint64_t end = index[u + 1];
      if (start == end) continue;
      for (uint64_t offset = start; offset < end - 1; offset++) {
        uint32_t v0 = neighbours[offset];
        uint32_t v1 = neighbours[offset + 1];
        v_dist += std::max(v0, v1) - std::min(v0, v1);
      }
    }
    return {avg_u_dist, double(v_dist) / m};
  }

  private:
  uint64_t *index = nullptr;
  uint32_t *neighbours = nullptr;
  double *weights = nullptr;
  uint32_t n;
  uint64_t m;
  bool in;
  bool weighted;
};

#endif// RHUBARB_CSR_H
