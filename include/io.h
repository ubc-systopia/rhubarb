//
// Created by atrostan on 29/03/23.
//

#include "WeightedEdge.h"
#include "fmt/core.h"
#include "util.h"
#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#ifndef RHUBARB_IO_H
#define RHUBARB_IO_H

// rabbit edgelist reader
off_t file_size(const int fd);
struct file_desc {
  int fd;
  file_desc(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
      std::cerr << "open(2): " << strerror(errno);
  }
  ~file_desc() {
    if (close(fd) != 0)
      std::cerr << "close(2): " << strerror(errno);
  }
};

struct mmapped_file {
  size_t size;
  const char *data;
  mmapped_file(const file_desc &fd) {
    size = file_size(fd.fd);
    data = static_cast<char *>(
            mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd.fd, 0));
    if (data == NULL)
      std::cerr << "mmap(2): " << strerror(errno);
  }
  ~mmapped_file() {
    if (munmap(const_cast<char *>(data), size) != 0)
      std::cerr << "munmap(2): " << strerror(errno);
  }
};



bool is_comment(std::string line);

/**
 * Given a path to an edge list and a reference to a vector of pairs,
 * populate the edgelist vector
 * @param path
 * @param edge_list
 */
template<typename Edge>
void read_text_edge_list(
        std::string path, std::vector<Edge> &edge_list) {
  std::string line;
  uint64_t i = 0;

  std::ifstream input_file(path);
  if (input_file.is_open()) {
    while (getline(input_file, line)) {
      std::stringstream linestream(line);
      if (is_comment(line))
        continue;
      uint32_t src, dest;

      if constexpr (std::is_same_v<Edge, WeightedEdge>) {
        double wt;
        linestream >> src >> dest >> wt;
        if (wt == 0)
          wt = 1;
        edge_list[i] = {src, dest, wt};

      } else {
        linestream >> src >> dest;
        edge_list[i].first = src;
        edge_list[i].second = dest;
      }

      ++i;
    }
    input_file.close();
  } else
    std::cout << "Unable to open file";
  // resize to actual value of number of edges seen
  edge_list.resize(i);
}


/**
 * Write an edgelist to a text file
 * @param path
 * @param es
 */
template<typename Edge>
void write_text_edge_list(std::string path,
                          std::vector<Edge> &es) {
  std::string line;
  // flat edges' size has been preallocated and initialized
  std::ofstream output_file(path);
  if (output_file.is_open()) {
    for (uint32_t i = 0; i < es.size(); ++i) {
      if constexpr (std::is_same_v<Edge, WeightedEdge>) {
        output_file << es[i].src << " " << es[i].dst << " " << es[i].wt << "\n";
      } else {
        output_file << es[i].first << " " << es[i].second << "\n";
      }
    }
    output_file.close();
  } else
    std::cout << "Unable to open file";
}

void read_text_vertex_ordering(std::string in_path, std::vector<uint32_t> &vertex_ordering);



/**
 * Write a vector<T> to out_path as binary
 * First, writes the size of the vector - .size().
 * Second, writes the vector's data - .data().
 * @tparam T
 * @param out_path
 * @param v
 */
template<typename T>
void write_vector_as_bin(std::string out_path, std::vector<T> &v) {
  std::ofstream out(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
  fmt::print("Writing vector to {}\n", out_path);
  uint64_t size = v.size();
  out.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
  out.write(reinterpret_cast<const char *>(v.data()), size * sizeof(T));
  out.close();
}

/**
 * Reads the size of a vector<T> (size represented using uint64_t)
 * Fills a dynamic T array[size] with the values read from a binary file
 * Constructs a vector<T> from the array, and returns that vector
 * @tparam T
 * @param in_path
 * @return
 */
template<typename T>
std::vector<T> read_vector_as_bin(std::string in_path) {
  std::ifstream in(in_path, std::ios::binary | std::ios::in);
  if (!in.is_open()) {
    fmt::print("No file at: {} -  Returning empty vector.\n", in_path);
    std::vector<T> empty;
    return empty;
  }
  fmt::print("Reading vector from {}\n", in_path);

  uint64_t size = 0;
  in.read(reinterpret_cast<char *>(&size), sizeof(uint64_t));
  T *arr = new T[size]();
  in.read(reinterpret_cast<char *>(arr), size * sizeof(T));

  std::vector<T> v(arr, arr + size);
  in.close();

  // deallocate before returning
  delete[] arr;
  return v;
}


#endif//RHUBARB_IO_H
