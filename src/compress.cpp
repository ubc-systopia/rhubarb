//
// Created by atrostan on 30/03/23.
//
#include "CSR.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "io.h"
#include "util.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <string>
#include <tuple>
#include <vector>
#include "EdgeListReader.h"
// needed for safe creation of dirs
namespace fs = std::filesystem;

/**
 * @brief Read an edge list file
 * Remap all vertex IDs in the edgelist so that vertex IDs lie in the range
 * [0, n), where n is the number of vertices in the graph
 * Sort the edgelist by ascending (source, destination)
 * And write the remapped edgelist to file
 * 
 * @tparam Edge 
 * @param input_path 
 * @param output_dir 
 * @param output_path 
 */
template<typename Edge>
void compress(std::string input_path, std::string output_dir,
              std::string output_path) {
  std::vector<Edge> es = read_edge_list<Edge>(input_path);
  sort_and_remove_dups<Edge>(es);
  uint64_t m = es.size();
  std::vector<uint32_t> flat_es(m * 2);

  //flatten edges
#pragma omp parallel for
  for (uint64_t i = 0; i < m; ++i) {
    flat_es[i * 2] = es[i].first;
    flat_es[i * 2 + 1] = es[i].second;
  }
  sort_and_remove_dups<uint32_t>(flat_es);

  uint32_t n = flat_es.size();
  uint32_t max_vertex_id = flat_es[n - 1];
  fmt::print("{} unique vertices, {} unique edges\n", n, m);
  fmt::print("Max Vertex ID: {}\n", max_vertex_id);

  // write the unique number of vertices, edges to file
  std::string n_m_path = fmt::format("{}/{}", output_dir, "n_m");
  std::ofstream n_m_file(n_m_path);
  n_m_file << n << "\n";
  n_m_file << m << "\n";
  n_m_file.close();
  // create a map from uncompressed vertex IDs to compressed
  std::vector<uint32_t> id_map(max_vertex_id + 1);
#pragma omp parallel for
  for (uint32_t i = 0; i < n; ++i) {
    id_map[flat_es[i]] = i;
  }

  // translate (inplace) the edgelist
#pragma omp parallel for
  for (uint32_t i = 0; i < m; ++i) {
    es[i].first = id_map[es[i].first];
    es[i].second = id_map[es[i].second];
  }
  // (re)sort the translated edgelist by ascending <src, dest>
  std::sort(dpl::execution::par_unseq, es.begin(), es.end());

  fmt::print("Writing compressed edgelist to: {}\n", output_path);
  write_text_edge_list(output_path, es);
}


/**
 * Given:
 * 	a. an input file (edgelist format)
 * 	b. the name of the graph
 * 	c. path of parent directory that will store the graph
 *
 * 1. Read the graph's edges
 * 2. Compress the graph's vertex ID space to the range [0, n),
 * where |V| = n
 * 3. Relabel the edges using the compressed vertex IDs
 * 4. Remove duplicate edges
 * 5. Remove self directed edges
 * 6. Sort the relabelled edgelist by ascending (src, dets)
 * 7. Save the edgelist as a textfile in ${data_dir}/${graph_name}/comp.net
 * @param argc
 * @param argv
 * @return
 */

int main(int argc, char *argv[]) {
  opterr = 0;
  int opt;
  uint32_t num_vertices = 0;
  std::string graph_name;
  std::string data_dir;
  std::string input_path;
  bool make_csrs = false;
  bool write_out_degrees = false;
  bool bipartite = false;
  while ((opt = getopt(argc, argv, "bi:g:d:")) != -1) {
    switch (opt) {
      case 'b':
        bipartite = !bipartite;
        break;
      case 'i':
        input_path = optarg;
        break;
      case 'g':
        graph_name = optarg;
        break;
      case 'd':
        data_dir = optarg;// provide absolute path
        break;
      default:
        break;
    }
  }
  std::string graphs_dir = fmt::format("{}/{}", data_dir, "graphs");
  std::string graph_dir = fmt::format("{}/{}", graphs_dir, graph_name);
  // create a graphs directory, if not exists
  fs::create_directory(graphs_dir);
  fs::create_directory(graph_dir);

  std::string graph_type = bipartite ? "bipartite" : "directed";

  fmt::print(
          "Reading {} edgelist from: {}\n",
          graph_type, input_path);


  std::string graph_fname = "comp.net";
  std::string output_edge_list_path =
          fmt::format("{}/{}", graph_dir, graph_fname);

  if (bipartite) {
    // compress_bipartite(input_path, num_uncompressed_edges,
    //  output_edge_list_path, graph_dir);
    return 0;
  }
  using Edge = std::pair<uint32_t, uint32_t>;
  compress<Edge>(input_path, graph_dir, output_edge_list_path);
  return 0;
}