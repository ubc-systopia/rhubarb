//
// Created by atrostan on 29/03/23.
//

#include "util.h"


/**
 * Returns the smallest power of 2 greater than n
 * @param n
 * @return
 */
uint64_t hyperceiling(uint32_t n) {
  int B = 2;
  while (B < n) {
    B = B * 2;
  }
  return B;
}

// hilbert curve utility functions from wikipedia:
// https://en.wikipedia.org/wiki/Hilbert_curve
void rot(int64_t n, int64_t *x, int64_t *y, int64_t rx, int64_t ry) {
  if (ry == 0) {
    if (rx == 1) {
      *x = n - 1 - *x;
      *y = n - 1 - *y;
    }

    // swap x, y
    *x = *x + *y;
    *y = *x - *y;
    *x = *x - *y;
  }
}

// convert (x,y) to d (d == hilbert index)
int64_t xy2d(int64_t n, int64_t x, int64_t y) {
  int64_t rx, ry, s, d = 0;

  for (s = n / 2; s > 0; s /= 2) {
    rx = (x & s) > 0;
    ry = (y & s) > 0;
    d += s * s * ((3 * rx) ^ ry);
    rot(s, &x, &y, rx, ry);
  }

  return d;
}

// thank you to
// https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
uint64_t split_by_3(uint32_t a) {
  uint64_t x = a & 0x1fffff;// we only look at the first 21 bits
  x = (x | x << 32) &
      0x1f00000000ffff;// shift left 32 bits, OR with self, and
                       // 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) &
      0x1f0000ff0000ff;// shift left 32 bits, OR with self, and
                       // 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) &
      0x100f00f00f00f00f;// shift left 32 bits, OR with self, and
                         // 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) &
      0x10c30c30c30c30c3;// shift left 32 bits, OR with self, and
                         // 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

uint64_t xy2m(uint32_t x, uint32_t y) {
  uint64_t m = 0;
  m |= split_by_3(x) | split_by_3(y) << 1;
  return m;
}




/**
 * iterate over the neighbourhoods of either V1/V2 (specified by v1 bool param)
 * append vertices of both V1, V2 to the isomorphism map in order of visitation
 *
 * @param in_csr
 * @param out_csr
 * @param n1
 * @param cm_map
 * @param v1
 */
void vertex_set_bfs(CSR &in_csr, CSR &out_csr, uint32_t n1,
                    std::vector<uint32_t> &map, bool v1) {
  CSR *csr;

  std::vector<bool> visited(in_csr.num_nodes(), false);

  if (v1) csr = &out_csr;
  else
    csr = &in_csr;

  uint32_t u0 = v1 ? 0 : n1;
  uint32_t uN = v1 ? n1 : csr->num_nodes();
  uint32_t u_offset = v1 ? 0 : n1;
  uint32_t v_offset = v1 ? n1 : 0;
  for (uint32_t u = u0; u < uN; ++u) {
    map[u_offset++] = u;
    for (auto v: csr->neigh(u)) {
      // uint64_t start = csr->index[u];
      // uint64_t end = csr->index[u + 1];
      // uint32_t *ns = csr->neighbours;
      // map[u_offset++] = u;
      // for (uint64_t offset = start; offset < end; ++offset) {
      //   uint32_t v = ns[offset];
      if (not visited[v]) {
        map[v_offset++] = v;
        visited[v] = true;
      }
    }
  }
  fmt::print("u_offset, v_offset: {} {}\n", u_offset, v_offset);
  std::ofstream tmp("./tmp_bip");

  for (uint32_t i = 0; i < out_csr.num_nodes(); ++i) {
    tmp << map[i] << "\n";
  }

  tmp.close();
}

/**
 * @brief an invalid spray block reduction width is larger than 
 * (the number of vertices in the graph) / 8
 * 
 * @param num_vertices 
 * @return true 
 * @return false 
 */
bool invalid_block_width(uint32_t num_vertices) { 
  return BWIDTH > num_vertices / 8;
}

void check_valid_block_width(uint32_t n)
{
  if (invalid_block_width(n)) {
    uint32_t valid_bwidth = largest_valid_block_width(n);
    std::string except_str = fmt::format(
            "{} is an invalid block width for a graph with {} vertices.\n\t",
            BWIDTH, n);
    except_str += fmt::format("Recompile with BWIDTH <= {}", valid_bwidth);
    throw std::length_error(except_str);
  }
}

/**
 * @brief Given a number of vertices, the largest valid spray reduction block
 * width is the smallest power of 2 that is greater than 
 * (the number of vertices in the graph) / 16
 * 
 * @param num_vertices 
 * @return uint32_t 
 */
uint32_t largest_valid_block_width(uint32_t num_vertices) {
  return hyperceiling(num_vertices / 16);
}

/**
 * @brief
 * construct a "cuthill mckee" bipartite ordering
 * 1. sort V1, V2 by _ascending degree_
 * 2. perform an undirected BFS from the lowest degree v1
 * 3. assign the new indices to V1, V2 by order of visitation
 * break ties by ascending degree
 * @param in_csr
 * @param out_csr
 * @param out_degs
 * @param in_degs
 */
void undirected_cuthill_mckee(CSR &in_csr, CSR &out_csr, uint32_t n1,
                              std::vector<uint32_t> &cm_map) {
  std::queue<uint32_t> q;
  std::vector<bool> visited(out_csr.num_nodes(), 0);
  q.push(0);
  visited[0] = true;
  uint32_t n_vertices_assigned = 1;
  uint32_t v1_count = 0;
  uint32_t v2_count = 0;

  while (n_vertices_assigned < out_csr.num_nodes()) {
    while (not q.empty()) {
      uint32_t u = q.front();
      q.pop();
      if (u >= n1) {
        for (auto v: in_csr.neigh(u)) {
          cm_map[v1_count] = v;
          v1_count++;
          n_vertices_assigned++;
          visited[v] = true;
          q.push(v);
        }
      } else {
        for (auto v: in_csr.neigh(u)) {
          if (not visited[v]) {
            cm_map[v2_count + n1] = v;
            v2_count++;
            n_vertices_assigned++;
            visited[v] = true;
            q.push(v);
          }
        }
      }
    }
    // there may still exist unvisited vertices, find them and add to queue
    uint32_t t = std::distance(
            std::begin(visited),
            std::find_if(
                    std::begin(visited),
                    std::end(visited),
                    [](auto x) { return x == 0; }));
    q.push(t);
  }

  uint32_t sum = 0;
  for (uint32_t i = 0; i < out_csr.num_nodes(); ++i) {
    if (not visited[i]) {
      sum++;
      fmt::print("i: {}\n", i);
    }
  }
  fmt::print("v1_count, v2_count: {} {}\n", v1_count, v2_count);
  fmt::print("sum: {}\n", sum);

  std::ofstream tmp("./tmp_bip");

  for (uint32_t i = 0; i < out_csr.num_nodes(); ++i) {
    tmp << cm_map[i] << "\n";
  }

  tmp.close();

  // find first set bit
}

/**
 * Iterates in parallel over the diffs in the index array of the in, out csrs to
 * compute the in, out degree vectors, respectively.
 */
void get_degs_from_csrs(std::vector<uint32_t> &in_degs, CSR &in_csr,
                        std::vector<uint32_t> &out_degs, CSR &out_csr) {
#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < out_csr.num_nodes(); ++i) {
    out_degs[i] = out_csr.degree(i + 1);
    in_degs[i] = in_csr.degree(i + 1);
  }
}

