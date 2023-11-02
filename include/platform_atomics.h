//
// Created by atrostan on 04/04/23.
//

#ifndef RHUBARB_PLATFORM_ATOMICS_H
#define RHUBARB_PLATFORM_ATOMICS_H

template<typename T, typename U>
T fetch_and_add(T &x, U inc) {
  return __sync_fetch_and_add(&x, inc);
}

template<typename T>
bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
  return __sync_bool_compare_and_swap(&x, old_val, new_val);
}

#endif // RHUBARB_PLATFORM_ATOMICS_H
