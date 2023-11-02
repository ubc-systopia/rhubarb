#include<cstdint>

#ifndef RHUBARB_WEIGHTEDEDGE_H
#define RHUBARB_WEIGHTEDEDGE_H

struct WeightedEdge {
  uint32_t src;
  uint32_t dest;
  double weight;
  WeightedEdge(uint32_t _src, uint32_t _dest, double _weight)
      : src(_src), dest(_dest), weight(_weight) {}
  WeightedEdge() {
    src = -1;
    dest = -1;
    weight = 0;
  }
};

struct UnweightedEdge {
  uint32_t src;
  uint32_t dest;
  UnweightedEdge(uint32_t _src, uint32_t _dest)
      : src(_src), dest(_dest) {}
  UnweightedEdge() {
    src = -1;
    dest = -1;
  }
};

#endif