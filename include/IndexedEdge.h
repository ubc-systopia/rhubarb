#include "WeightedEdge.h"
#include <cstdint>
#include <fmt/format.h>
#ifndef RHUBARB_INDEXEDEDGE_H
#define RHUBARB_INDEXEDEDGE_H

/**
 * A struct that associates edges with an index
 * Used to assign edges a Hilbert index
 */
template<typename EdgeType>
class IndexedEdge {
  public:
  EdgeType e;
  uint64_t h_idx;

  auto src() { return e.src(); }
  auto dest() { return e.dest(); }
};

template<typename EdgeType>
struct fmt::formatter<IndexedEdge<EdgeType>> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(const IndexedEdge<EdgeType> &edge, FormatContext &ctx) const {

    if constexpr (std::is_same_v<EdgeType, WeightedEdge>) {
      return format_to(ctx.out(),
                       "#{} ({}, {}, {})",
                       edge.h_idx, edge.e.src, edge.e.dest, edge.e.weight);
    }

    return format_to(ctx.out(), "#{} ({}, {})",
                     edge.h_idx, edge.e.src, edge.e.dest);
  }
};

#endif