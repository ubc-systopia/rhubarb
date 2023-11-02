#include <cstdint>
template<typename TData>
class EdgeBase {
  public:
  uint32_t src;
  uint32_t dest;
  
  void foo() {
    static_cast<TData *>(this)->doFoo();
  }
};

class Edge : public EdgeBase<Edge> {
  friend EdgeBase<Edge>;
  void doFoo() { cout << "A::foo()\n"; }
};

class WeightedEdge : public EdgeBase<WeightedEdge> {
  friend EdgeBase<WeightedEdge>;
  int someDataMember;
  void doFoo() { cout << "AData::foo()\n"; /*... use someDataMember ... */ }
};