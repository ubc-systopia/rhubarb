//
// Created by atrostan on 29/03/23.
//
#include "CSR.h"
#include "EdgeListReader.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "io.h"
#include "util.h"
#include <getopt.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <string>
#include <vector>

/**
 * Given a path to a compressed edgelist
 * (A compressed edgelist is one whose edges are guaranteed to be within the
 * the range [0, |V|))
 * And a path to an isomoprhism map of that graph's vertices, of the form:
 *
 * num_vertices
 * num_edges
 * 0 {Vertex 0's new id in the iso map}
 * 1 {Vertex 1's new id in the iso map}
 * ...
 * n-1 {Vertex n-1's new id in the iso map}
 *
 * Translate and sorts the edgelist, and writes the translated edgelist it to
 * file
 * @param argc
 * @param argv
 * @return
 */

int main(int argc, char *argv[]) {
  opterr = 0;
  int opt;
  uint32_t num_vertices = 0;// n vertices
  std::string graph_name = "";
  std::string data_dir = "";
  std::string iso_name = "";
  bool bipartite = false;
  while ((opt = getopt(argc, argv, "bg:d:o:")) != -1) {
    switch (opt) {
      case 'b':
        bipartite = !bipartite;
        break;
      case 'g':
        graph_name = optarg;
        break;
      case 'd':
        data_dir = optarg;
        break;
      case 'o':
        iso_name = optarg;
        break;
    }
  }
  // the name of the isomorphism map
  std::string graphs_dir = fmt::format("{}/{}", data_dir, "graphs");
  std::string graph_dir = fmt::format("{}/{}", graphs_dir, graph_name);
  std::string order_path = fmt::format("{}/{}", graph_dir, iso_name);
  std::string binary_order_path = fmt::format("{}/{}.bin", graph_dir, iso_name);

  fmt::print("order_path: {}\n", order_path);
  std::string text_edge_list_path = fmt::format("{}/{}", graph_dir, "comp.net");
  std::string gmat_text_edge_list_path = fmt::format("{}/{}", graph_dir, "graphmat_comp.net0");

  fmt::print("Reading text edge list from: {}..\n", text_edge_list_path);

  if (bipartite) {
    using Edge = WeightedEdge;
    uint64_t num_edges = 0;
    return 0;//todo
    std::vector<WeightedEdge> edge_list(num_edges);
    // read the unique number of vertices, edges to file
    std::string n_m_path = fmt::format("{}/{}", graph_dir, "n_m");
    std::ifstream n_m_file(n_m_path);
    uint32_t n1, n2;
    // read sizes of both vertex sets
    n_m_file >> n1;
    n_m_file >> n2;
    n_m_file.close();
    // read_weighted_edge_list(text_edge_list_path, edge_list);
    uint64_t m = edge_list.size();
    fmt::print("n1, n2, m: {} {} {}\n", n1, n2, m);

    create_bipartite_csrs(edge_list, graph_dir, n1, n2);

    return 0;
  }
  using Edge = std::pair<uint32_t, uint32_t>;

  std::vector<uint32_t> iso_map(num_vertices);
  // std::vector<Edge> edge_list(num_edges);

  // read_text_edge_list(text_edge_list_path, edge_list);
  std::vector<Edge> edge_list = read_edge_list<Edge>(text_edge_list_path);
  uint64_t m = edge_list.size();


  uint32_t n = 0;
  // assumption: vertex ids lie within the range [0, |V|)
  fmt::print("Getting max vertex id..\n");
  n = get_max_vertex_id(edge_list);
  fmt::print("Max vertex ID in edgelist: {}\n", n);
  n += 1;

  check_valid_block_width(n);

  iso_map.resize(n);
  if (iso_name ==
      "orig")// construct the dummy [vid:vid], original isomorphism map
  {
    fmt::print("Initializing dummy map..\n");
#pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i) {
      iso_map[i] = i;
    }

  } else {
    fmt::print("Reading text {}-vertex-order file from: {}\n", iso_name,
               order_path);
    read_text_vertex_ordering(order_path, iso_map);

    write_vector_as_bin<uint32_t>(binary_order_path, iso_map);
  }
  // save the vertex order file as binary (for future use)
  fmt::print("Writing binary {}-vertex-order file to: {}\n", iso_name,
             binary_order_path);
  write_vector_as_bin<uint32_t>(binary_order_path, iso_map);

  fmt::print("{} with {} vertices and {} edges\n", graph_name, n, m);

  if (iso_name != "orig") {
    fmt::print("Relabelling {}'s vertex IDs using {}\n", graph_name, iso_name);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
      uint32_t src = edge_list[i].first;
      uint32_t dest = edge_list[i].second;
      edge_list[i].first = iso_map[src];
      edge_list[i].second = iso_map[dest];
    }
  }
  fmt::print("Sorting mapped edgelist..\n");
  std::sort(dpl::execution::par_unseq, edge_list.begin(), edge_list.end());
  fmt::print("Sorted.\n");

  // align the size of the spray reduction array
  uint32_t reduce_arr_size = ((n / BWIDTH) + 1) * BWIDTH;
  fmt::print("reduce_arr_size: {}\n", reduce_arr_size);
  uint32_t *out_degs = new uint32_t[reduce_arr_size]();
  uint32_t *in_degs = new uint32_t[reduce_arr_size]();
  // uint32_t mx_in = 0, mx_out = 0;

  auto [mx_in, mx_out] =
          par_compute_degrees(in_degs, out_degs, edge_list, n, reduce_arr_size);

  fmt::print("{} - Max In-Degree: {} | Max out-Degree: {}\n", graph_name, mx_in,
             mx_out);

  // create the in-csr and out-csr representation
  CSR in_csr = CSR(n, m, true);
  CSR out_csr = CSR(n, m, false);
  fmt::print("Populating out-csr in parallel..\n");

  out_csr.par_populate(out_degs, edge_list);

  fmt::print("Sorting by dest..\n");

  // before populating in the in-csr, sort the edgelist by ascending (dest, src)
  std::sort(dpl::execution::par_unseq, edge_list.begin(), edge_list.end(),
            [](const Edge &l,
               const Edge &r) -> bool {
              if (l.second == r.second) {
                return l.first < r.first;
              }
              return l.second < r.second;
            });

  in_csr.par_populate(in_degs, edge_list);


  std::string out_csr_path =
          fmt::format("{}/{}_{}", graph_dir, iso_name, "out_csr.bin");
  std::string in_csr_path =
          fmt::format("{}/{}_{}", graph_dir, iso_name, "in_csr.bin");
  out_csr.write_as_binary(out_csr_path);
  in_csr.write_as_binary(in_csr_path);

  return 0;
  std::string ligra_adj_path = fmt::format("{}/{}_ligra_adj", graph_dir, iso_name);
  out_csr.write_ligra_adj(ligra_adj_path);

  // clean up
  delete[] in_degs;
  delete[] out_degs;
  // write mapped text edge list regardless, so that it can be preprocessed
  // using ligra, gpop
  std::string mapped_edgelist_path =
          fmt::format("{}/{}_comp.{}", graph_dir, iso_name, "net");

  std::string gmat_mapped_edgelist_path =
          fmt::format("{}/{}_graphmat_comp.{}", graph_dir, iso_name, "net0");

  // before writing the mapped edgelist to file, (re)sort the edgelist by
  // ascending (src, dest)
  std::sort(dpl::execution::par_unseq, edge_list.begin(), edge_list.end(),
            [](const Edge &l,
               const Edge &r) -> bool {
              if (l.first == r.first) {
                return l.second < r.second;
              }
              return l.first < r.first;
            });
  fmt::print("Writing {}-edgelist to: {}\n", iso_name, mapped_edgelist_path);
  write_text_edge_list(mapped_edgelist_path, edge_list);
  // also write an edge list whose first vertex id == 1 (required for GraphMat)
  fmt::print("Writing 1+text edge list to: {}..\n", gmat_mapped_edgelist_path);
  std::ofstream gmat_mapped_outfile(gmat_mapped_edgelist_path);
  for (uint32_t i = 0; i < edge_list.size(); ++i) {
    const auto &e = edge_list[i];
    gmat_mapped_outfile << fmt::format("{} {}\n", e.first + 1, e.second + 1);
  }
  gmat_mapped_outfile.close();
}