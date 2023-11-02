//
// Created by atrostan on 02/04/23.
//

#ifndef RHUBARB_RECURSIVEBLOCK_H
#define RHUBARB_RECURSIVEBLOCK_H

#include "IndexedEdge.h"
#include <cstdint>
#include <vector>
template<typename EdgeType>
class RecursiveBlock {
  public:
  uint32_t x;
  uint32_t y;
  uint32_t side_len;
  uint32_t nnz;
  std::vector<IndexedEdge<EdgeType>> es;
  uint64_t idx;

  RecursiveBlock() {
    x = 0;
    y = 0;
    side_len = 0;
    nnz = 0;
    es.resize(0);
    idx = 0;
  }

  ~RecursiveBlock() {}
};

template<typename EdgeType>
struct fmt::formatter<RecursiveBlock<EdgeType>> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(const RecursiveBlock<EdgeType> &b, FormatContext &ctx) const {

    return format_to(ctx.out(), 
                    "#{} (x:{}, y:{}) (s:{}, n:{})\n{}",
                     b.idx, b.x, b.y, b.side_len, b.nnz,b.es);

  }
};


#endif// RHUBARB_RECURSIVEBLOCK_H
