#ifndef _repeat_iterator_h_
#define _repeat_iterator_h_
#include <thrust/iterator/iterator_adaptor.h>

template<typename Iterator>
class repeat_iterator : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>
{
public:
  typedef thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> super_t;
  repeat_iterator(const Iterator& x, int n) : super_t(x), begin(x), n(n) {}
  friend class thrust::iterator_core_access;
private:
  unsigned int n;
  const Iterator begin;
  __host__ __device__
  typename super_t::reference derefernce() const
  {
    return *(begin + (this->base() - begin)/n);
  }
};
#endif
