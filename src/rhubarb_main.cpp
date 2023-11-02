//
// Created by atrostan on 29/03/23.
//

// #include "../apps/cc.h"
// #include "../apps/cf.h"
#include "../apps/pr.h"
#include "CSR.h"
#include "Rhubarb.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "io.h"
#include "likwid_defines.h"
#include "util.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <set>
#include <vector>
// #include "ips4o.hpp"


using namespace std::chrono;

enum app_code {
  ePageRank,
  eConnectedComponents,
  eCollaborativeFiltering,
  eNone,
  eDefault
};

/**
 * Given a string of an edge-centric application:
 * One of: PageRank, Connected Components, Collaborative Filtering
 * Return the correspnding enum
 * @param app_str
 * @return
 */
app_code match_str(std::string const &app_str) {
  if (app_str == "pr") return ePageRank;
  if (app_str == "cc") return eConnectedComponents;
  if (app_str == "cf") return eCollaborativeFiltering;
  if (app_str == "none") return eNone;
  return eDefault;
}


/**
 * Reads the in, out-csr of a graph whose vertices have been reordered using
 * an isomorphism map
 * Constructs the Rhubarb representation of the input graph using Recursive
 * Hilbert Blocking
 * If an algorithm string is supplied, computes either:
 * 1. PageRank
 * 2. Connected Components
 * 3. Collaborative Filtering
 *
 * Records the time to complete a repetition of the above algorithms.
 * If `rhubarb_main` is called with `likwid-perfctr`, the execution of each
 * edge-centric algorithm will be profiled.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {

  bool bipartite;
#if WEIGHTED_EDGE == 1
  bipartite = true;
  using EdgeType = WeightedEdge;
#else
  bipartite = false;
  using EdgeType = UnweightedEdge;
#endif

  opterr = 0;
  int opt;
  uint32_t n = 0;               // n vertices
  uint32_t n1, n2 = 0;          // if a bipartite graph, the number of vertices
                                // in the first vertex set
  uint64_t m = 0;               // n edges
  uint num_expts = 0;           // number of repetitions
  uint num_threads = 0;         // number of cores to use
  uint num_iterations = 0;      // number of iterations for pr, cf
  std::string graph_name = "";  // the label of the graph
  std::string data_dir = "";    // graph dir ==  {data_dir}/graphs/{graph_name}
  uint32_t max_num_edges = 0;   // rhubarb hyperparameter: m
  bool debug = false;           // whether to output rhb block metadata
  bool verif = false;           // verify algo result against precomputed
                                // ground truth
  bool rand_init = false;       // randomly initialize latent values in Collaborative Filtering
                                // benchmark
  std::string order_label = ""; // label of vertex ordering
  std::string algo_str = "";    // one of "pr", "cc", "cf"
  std::vector<uint> thread_nums;// vector that stores all core-numbers configs
  uint dynamic_group_size = 1;  // rhubarb hyperparameter: b
  uint16_t K = 20;
  topology_init();
  auto topo = get_cpuTopology();
  double step_size = 0.00000035;

  // get sizes of caches in bytes
  uint64_t L3_SIZE = topo->cacheLevels[2].size;
  // todo use L2 Size to replace m hyperparameter
  uint64_t L2_SIze = topo->cacheLevels[1].size;
  while ((opt = getopt(argc, argv, "bvrg:d:n:m:o:t:e:y:a:i:k:")) != -1) {
    switch (opt) {
      case 'r':
        rand_init = !rand_init;
        break;
      case 'b':
        debug = !debug;
        break;
      case 'v':
        verif = !verif;
        break;
      case 'g':
        graph_name = optarg;
        break;
      case 'd':
        data_dir = optarg;
        break;
      case 'm':
        max_num_edges = atoi(optarg);
        break;
      case 'n':
        n1 = atoi(optarg);
        break;
      case 'o':
        order_label = optarg;
        break;
      case 'e':
        num_expts = atoi(optarg);
        break;
      case 't':
        thread_nums.push_back(atoi(optarg));
        break;
      case 'y':
        dynamic_group_size = atoi(optarg);
        break;
      case 'a':
        algo_str = optarg;
        break;
      case 'i':
        num_iterations = atoi(optarg);
        break;
      case 'k':
        K = atoi(optarg);
        break;
      case 's':
        step_size = atof(optarg);
        break;
    }
  }
  fmt::print("thread_nums: {}\n", thread_nums);
  num_threads = thread_nums[thread_nums.size() - 1];

  std::string graphs_dir = fmt::format("{}/{}", data_dir, "graphs");
  std::string graph_dir = fmt::format("{}/{}", graphs_dir, graph_name);
  fmt::print("debug? {}\n", debug);
  std::string binary_order_path =
          fmt::format("{}/{}.bin", graph_dir, order_label);
  std::string binary_edge_list_path =
          fmt::format("{}/{}", graph_dir, "comp.bin");

  std::string out_csr_path;
  std::string in_csr_path;
  out_csr_path = fmt::format("{}/{}", graph_dir, "out_csr.bin");
  in_csr_path = fmt::format("{}/{}", graph_dir, "in_csr.bin");
  CSR out_csr;
  CSR in_csr;
  if (bipartite) {
    out_csr_path = fmt::format("{}/{}", graph_dir, "out_csr.bin");
    in_csr_path = fmt::format("{}/{}", graph_dir, "in_csr.bin");
    out_csr = CSR(out_csr_path, false, true);
    in_csr = CSR(in_csr_path, true, true);

    fmt::print("k: {}, step_size: {}\n", K, step_size);
  } else {
    out_csr_path = fmt::format("{}/{}_{}", graph_dir, order_label, "out_csr.bin");
    in_csr_path = fmt::format("{}/{}_{}", graph_dir, order_label, "in_csr.bin");
    // read csrs for this isomorphism
    out_csr = CSR(out_csr_path, false);
    in_csr = CSR(in_csr_path, true);
  }

  uint32_t max_in_deg = in_csr.max_degree();
  uint32_t max_out_deg = out_csr.max_degree();

  fmt::print("Max In-Degree: {}\n", max_in_deg);
  fmt::print("Max Out-Degree: {}\n", max_out_deg);
  // return 0;
  fmt::print("BWIDTH: {}\n", BWIDTH);
  n = out_csr.num_nodes();
  m = out_csr.num_edges();

  if (bipartite)
    n2 = n - n1;

  // init isomorphism map (needed for translation of ground truth pr, cc values)
  std::vector<uint32_t> iso_map;
  if (order_label == "orig") {
    iso_map.resize(n);
#pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i)
      iso_map[i] = i;
  } else {
    iso_map = read_vector_as_bin<uint32_t>(binary_order_path);
  }

  // check that spray block width does not exceed the number of vertices in the
  // graph
  uint32_t max_bwidth_thresh = hyperceiling(n / 2);
  // the maximum number of edges per block should be at most 1/16 * |E|
  uint32_t max_num_edges_thresh = hyperceiling((1.0f / 16) * m);
  check_valid_block_width(n);

  if (max_num_edges > max_num_edges_thresh) {
    fmt::print("max_num_edges: {} > {}\n", max_num_edges, max_num_edges_thresh);
    fmt::print("Please use a smaller number of edges per block.\n");
    return 1;
  }
  if (bipartite) {
    fmt::print("n1, n2, m: {} {} {} \n", n1, n2, m);
  } else {
    fmt::print("n, m: {} {} \n", n, m);
  }
  std::vector<uint32_t> out_degs(n);
  std::vector<uint32_t> in_degs(n);
  get_degs_from_csrs(in_degs, in_csr, out_degs, out_csr);

  fmt::print("num_threads: {}\n", num_threads);
  fmt::print("num_expts: {}\n", num_expts);
  fmt::print("max_num_edges: {}\n", max_num_edges);

  // flattened edgelist of the graph, sorted in ascending hierarchical hilbert
  uint32_t *block_edges = nullptr;
  // the start + end point of each recursive hilbert block
  uint32_t *block_index = nullptr;
  // if weighted, the weight associated with each edge
  double *block_weights = nullptr;



  Rhubarb rhb = Rhubarb<EdgeType>(in_csr, num_threads, max_num_edges,
                                  block_edges, block_index, block_weights);
  rhb.info(graph_name);
  uint32_t n_non_emptyblocks = rhb.n_nonempty_blocks;
  std::vector<uint32_t> top_k;
  if (debug) {
    // before deleting the csrs, get the id's of the top highest degree nodes

    if (bipartite) {
      // get the id's of the top highest in-degree nodes
      top_k = in_csr.top_k_hubs(10);
      fmt::print("top_k: {}\n", top_k);
    }
  }

  // now that csrs are no longer needed, delete them
  // in_csr.release_resources();
  // out_csr.release_resources();

  if (debug) {
    std::string rec_bs_path =
            fmt::format("{}/{}-{}-rec_bs", graph_dir, order_label, max_num_edges);
    std::string sorted_edges_path = fmt::format(
            "{}/{}-{}-hilbert_edges", graph_dir, order_label, max_num_edges);
    std::string block_edges_path = fmt::format(
            "{}/{}-{}-block_edges", graph_dir, order_label, max_num_edges);
    std::string block_index_path = fmt::format(
            "{}/{}-{}-block_index", graph_dir, order_label, max_num_edges);

    rhb.write_blocks_metadata(rec_bs_path);

    uint32_t debug_thresh = 1'000'000;
    if (m <= debug_thresh) {
      // rhb.write_edges(sorted_edges_path);
      // only write graphs with a relatively small number of edges
      rhb.write_block_edges(block_edges_path, block_index_path, block_edges,
                            block_index, rhb.n_nonempty_blocks);
    }
  }

  // clean up rhubarb now that edges are reordered, flattened and ready to be processed
  rhb.release_resources();
  // likwid initialization
  LIKWID_MARKER_INIT;

#pragma omp parallel num_threads(num_threads)
  {
    LIKWID_MARKER_THREADINIT;
  }

#pragma omp parallel num_threads(num_threads)
  {
    LIKWID_MARKER_REGISTER("pr");
    LIKWID_MARKER_REGISTER("cc");
    LIKWID_MARKER_REGISTER("cf");
  }

  // result file to output runtimes to
  std::string results_path =
          fmt::format("{}/{}.results.csv", graph_dir, algo_str);

  std::vector<std::string> results_columns = {
          "graph_name", "order_label", "algo_str", "max_num_edges",
          "dynamic_group_size", "block_width", "runtime", "num_iterations",
          "num_cores", "expt_num"};
  std::ofstream results_file(results_path, std::ios_base::app);

  // write csv header
  for (const auto &col: results_columns)
    results_file << col << ",";
  results_file << "\n";

  for (auto num_threads: thread_nums) {
    switch (match_str(algo_str)) {
      case ePageRank: {
        auto *scores = new ScoreT[n]();
        for (uint expt_num = 0; expt_num < num_expts; ++expt_num) {
          fmt::print("expt_num: {}\n", expt_num);
          uint64_t runtime = pr_compute(out_degs, n, n_non_emptyblocks,
                                        num_iterations, block_edges, block_index,
                                        dynamic_group_size, num_threads, scores);

          fmt::print("{}-{} iterations of PageRank: {} ms\n", graph_name,
                     num_iterations, runtime);
          // append to results.csv
          results_file << fmt::format("{},{},{},{},{},{},{},{},{},{}\n", graph_name,
                                      order_label, algo_str, max_num_edges,
                                      dynamic_group_size, BWIDTH, runtime,
                                      num_iterations, num_threads, expt_num);
        }
      }

        //       case eConnectedComponents: {
        //         std::vector<uint32_t> labels;
        //         uint64_t runtime;
        //         uint32_t num_cc_iters;
        //         for (uint expt_num = 0; expt_num < num_expts; ++expt_num) {
        //           std::tie(labels, runtime, num_cc_iters) =
        //                   sv_cc_compute(n, block_edges, block_index, dynamic_group_size,
        //                                 num_threads, n_non_emptyblocks);
        //           // append to results.csv
        //           results_file << fmt::format("{},{},{},{},{},{},{},{},{},{}\n", graph_name,
        //                                       order_label, algo_str, max_num_edges,
        //                                       dynamic_group_size, BWIDTH, runtime,
        //                                       num_cc_iters, num_threads, expt_num);
        //         }

        //         std::map<uint32_t, uint32_t> tp_vcs =
        //                 par_value_count<uint32_t>(num_threads, labels);

        //         if (verif) {
        //           // read the groundtruth connected components vec
        //           std::string cc_path = fmt::format("{}/{}", graph_dir, "cc");
        //           std::vector<uint32_t> cc = read_vector_as_bin<uint32_t>(cc_path);
        //           std::map<uint32_t, uint32_t> gt_vcs =
        //                   par_value_count<uint32_t>(num_threads, cc);

        //           fmt::print("Rhubarb Component-Size Value Counts: {}\n", tp_vcs);
        //           fmt::print("Ground Truth Component-Size Value Counts: {}\n", gt_vcs);
        //           bool valid = std::equal(tp_vcs.begin(), tp_vcs.end(), gt_vcs.begin());
        //           fmt::print("Valid Connected Components?: {}\n", valid);
        //         }

        //         break;
        //       }

        //       case eCollaborativeFiltering: {
        //         for (uint expt_num = 0; expt_num < num_expts; ++expt_num) {
        //           if (!bipartite) {
        //             block_weights = new double[m]();
        // #pragma omp parallel for
        //             for (uint32_t i = 0; i < m; ++i) {
        //               block_weights[i] = 1.0;
        //             }
        //           }
        //           double *latent_curr{nullptr};
        //           posix_memalign((void **) &latent_curr, 64,
        //                          sizeof(double) * K * n);
        //           assert(latent_curr != nullptr && ((uintptr_t) latent_curr % 64 == 0) &&
        //                  "App Malloc Failure\n");
        //           uint64_t runtime = cf_compute(
        //                   n, n1, n2, n_non_emptyblocks, num_iterations, latent_curr,
        //                   block_edges, block_index, dynamic_group_size, num_threads,
        //                   block_weights, bipartite, verif, rand_init, K, step_size);
        //           results_file << fmt::format("{},{},{},{},{},{},{},{},{},{}\n", graph_name,
        //                                       order_label, algo_str, max_num_edges,
        //                                       dynamic_group_size, BWIDTH, runtime,
        //                                       num_iterations, num_threads, expt_num);
        //           // print out the highest rated items' latent vectors
        //           for (const auto &vid: top_k) {
        //             std::vector<double> latent_vals(K);
        //             for (uint k = 0; k < K; ++k) {
        //               latent_vals[k] = latent_curr[K * vid + k];
        //             }
        //             std::string vertex_identifier = fmt::format("Vertex {}:", vid);
        //             fmt::print("{:<20} {:>12.3f}\n", vertex_identifier, fmt::join(latent_vals, ", "));
        //           }
        //           free(latent_curr);
        //         }

      case eNone: {
        break;
      }
      case eDefault: {
        break;
      }
    }
  }


  // clean up
  results_file.close();
  delete[] block_edges;
  delete[] block_index;
  delete[] block_weights;


  LIKWID_MARKER_CLOSE;
  return 0;
}