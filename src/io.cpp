//
// Created by atrostan on 29/03/23.
//

#include "io.h"

off_t file_size(const int fd) {
  struct stat st;
  if (fstat(fd, &st) != 0)
    std::cerr << "stat(2): " << strerror(errno);
  return st.st_size;
}

/**
 * @brief 
 * 
 * @param line 
 * @return true, if the line contains a comment char
 * @return false, otherwise
 */
bool is_comment(std::string line) {
  return (line.find("%") != std::string::npos) ||
         (line.find("#") != std::string::npos);
}


void read_text_vertex_ordering(std::string in_path,
                               std::vector<uint32_t> &vertex_ordering) {
  // since vertex ids form the range [0, num_vertices - 1), it's safe to use vertex_ordering
  // as a vector to map between original vertex ids to the new vertex ids
  std::ifstream input_file(in_path);
  std::string line;

  // skip the first two lines - they show the number of nodes + edges in the
  // edgelist
  getline(input_file, line);
  getline(input_file, line);
  if (input_file.is_open()) {
    while (getline(input_file, line)) {
      uint32_t orig_id;
      uint32_t mapped_id;
      std::stringstream linestream(line);
      linestream >> orig_id >> mapped_id;
      vertex_ordering[orig_id] = mapped_id;
    }
  }
}