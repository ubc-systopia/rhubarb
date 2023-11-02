//
// Optimized edge list reader
// Adapted from
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//


#include "io.h"

template<typename Edge>
class EdgeListReader {
  public:
  EdgeListReader(const std::string &_path) {
    path = _path;
  }
  std::vector<Edge> read() {
    const file_desc fd(path);
    const mmapped_file mm(fd);
    const int nthread = omp_get_max_threads();
    const size_t zchunk = 1024 * 1024 * 64;// 64MiB
    const size_t nchunk = mm.size / zchunk + (mm.size % zchunk > 0);

    //
    // For load balancing, partition the file into small chunks (whose size is
    // defined as `zchunk`) and dynamically assign the chunks into threads
    //

    std::vector<std::deque<Edge>> eparts(nthread);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < nchunk; ++i) {
      const char *p = mm.data + zchunk * i;
      const char *q = mm.data + std::min(zchunk * (i + 1), mm.size);

      // Advance pointer `p` to the end of a line because it is possibly at the
      // middle of the line
      if (i > 0) p = std::find(p, q, '\n');

      if (p < q) {                                // If `p == q`, do nothing
        q = std::find(q, mm.data + mm.size, '\n');// Advance `q` likewise
        EdgeParser(p, q)(std::back_inserter(eparts[omp_get_thread_num()]));
      }
    }

    // Compute indices to copy each element of `eparts` to
    std::vector<size_t> eheads(nthread + 1);
    for (int t = 0; t < nthread; ++t)
      eheads[t + 1] = eheads[t] + eparts[t].size();

    // Gather the edges read by each thread to a single array
    std::vector<Edge> edges(eheads.back());
#pragma omp parallel for schedule(guided, 1)
    for (int t = 0; t < nthread; ++t)
      std::copy(eparts[t].begin(), eparts[t].end(), edges.begin() + eheads[t]);

    return edges;
  }

  private:
  std::string path;

  class EdgeParser {
public:
    const char *const strfirst;
    const char *const strlast;
    const char *crr;

    EdgeParser(const char *const first, const char *const last)
        : strfirst(first), strlast(last), crr(first) {}

    template<typename OutputIt>
    void operator()(OutputIt dst) {
      while (crr < strlast) {
        eat_empty_lines();
        if (crr < strlast)
          *dst++ = eat_edge();
      }
    }

    Edge eat_edge() {
      const uint32_t s = eat_id();
      eat_separator();
      const uint32_t t = eat_id();

      // weighted edge list; todo only weights supported are positive integers
      if constexpr (std::is_same_v<Edge, WeightedEdge>) {
        eat_separator();
        const uint32_t w = eat_id();
        return Edge{s, t, w};
      } else {
        return Edge{s, t};
      }
    }

    uint32_t eat_id() {
      //
      // Naive implementation is faster than library functions such as `atoi` and
      // `strtol`
      //
      const auto _crr = crr;
      uint32_t v = 0;
      for (; crr < strlast && std::isdigit(*crr); ++crr) {
        const uint32_t _v = v * 10 + (*crr - '0');
        if (_v < v)// overflowed
          std::cerr << "Too large vertex ID at line " << crr_line();
        v = _v;
      }
      if (_crr == crr)// If any character has not been eaten
        std::cerr << "Invalid vertex ID at line " << crr_line();
      return v;
    }

    void eat_empty_lines() {
      while (crr < strlast) {
        if (*crr == '\r') ++crr;// Empty line
        else if (*crr == '\n')
          ++crr;// Empty line
        else if (*crr == '#')
          crr = std::find(crr, strlast, '\n');// Comment
        else
          break;
      }
    }

    void eat_separator() {
      while (crr < strlast &&
             (*crr == '\t' || *crr == ',' || *crr == ' ' || *crr == ';'))
        ++crr;
    }

    // Only for error messages
    size_t crr_line() {
      return std::count(strfirst, crr, '\n');
    }
  };
};

/**
 * @brief 
 * 
 * wrapper for a function that mmaps a text edge list and populates a vector of 
 * edges in parallel. Time to read the edgelist is displayed in seconds.
 * 
 * @tparam Edge weighted vs. unweighted edge
 * @param path 
 * @return std::vector<Edge> 
 */
template<typename Edge>
std::vector<Edge> read_edge_list(std::string &path) {
  EdgeListReader<Edge> elr(path);

  using EdgeList = std::vector<Edge>;
  using TimeUnit = std::chrono::milliseconds;

  std::tuple<EdgeList, TimeUnit> res = time_function_invocation<TimeUnit>(
          &EdgeListReader<Edge>::read, &elr);

  auto es = std::get<0>(res);
  auto read_runtime_ms = std::get<1>(res);
  auto read_time_ms = (double) read_runtime_ms.count() / 1'000;
  fmt::print("Read {} edges in {} s\n", es.size(), read_time_ms);
  return es;
}
