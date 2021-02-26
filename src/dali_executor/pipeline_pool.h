// The MIT License (MIT)
//
// Copyright (c) 2020 NVIDIA CORPORATION
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef DALI_BACKEND_DALI_EXECUTOR_PIPELINE_POOL_H_
#define DALI_BACKEND_DALI_EXECUTOR_PIPELINE_POOL_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/dali_executor/dali_pipeline.h"

namespace triton { namespace backend { namespace dali {

struct PipelineDescr {
  size_t serialized_pipeline_hash;
  int max_batch_size, device_id, bytes_per_sample_hint, num_threads, seed;


  PipelineDescr(
      const std::string& serialized_pipeline, int max_batch_size,
      int device_id = -1, int bytes_per_sample_hint = 0, int num_threads = -1,
      int seed = -1)
      : serialized_pipeline_hash(std::hash<std::string>{}(serialized_pipeline)),
        max_batch_size(max_batch_size), device_id(device_id),
        bytes_per_sample_hint(bytes_per_sample_hint), num_threads(num_threads),
        seed(seed)
  {
  }


  bool operator==(const PipelineDescr& other) const noexcept
  {
    return serialized_pipeline_hash == other.serialized_pipeline_hash &&
           max_batch_size == other.max_batch_size &&
           device_id == other.device_id &&
           bytes_per_sample_hint == other.bytes_per_sample_hint &&
           num_threads == other.num_threads && seed == other.seed;
  }
};

}}}  // namespace triton::backend::dali

namespace std {
template <>
struct hash<triton::backend::dali::PipelineDescr> {
  size_t operator()(
      triton::backend::dali::PipelineDescr const& pk) const noexcept
  {
    return pk.serialized_pipeline_hash ^ static_cast<size_t>(pk.max_batch_size);
  }
};
}  // namespace std

namespace triton { namespace backend { namespace dali {

class PipelinePool {
 public:
  /**
   * If needed, registers DaliPipeline in the Pool and returns reference to it.
   *
   * @param serialized_pipeline Serialized DALI pipeline
   * @param args Remaining parameters to DaliPipeline ctor
   * @return Reference to newly added DaliPipeline, or the old one,
   *         if given DaliPipeline already existed
   */
  template <typename... Args>
  DaliPipeline& Get(
      const std::string& serialized_pipeline, int max_batch_size,
      const Args&... args)
  {
    auto key = PipelineDescr(serialized_pipeline, max_batch_size, args...);
    if (pool_.find(key) == pool_.end()) {
      DaliPipeline pipeline(serialized_pipeline, max_batch_size, args...);
      ++created_pipelines_;
      return pool_.insert({key, std::move(pipeline)}).first->second;
    } else {
      return pool_[key];
    }
  }

  template <typename... Args>
  void Remove(
      const std::string& serialized_pipeline, int max_batch_size,
      const Args&... args)
  {
    pool_.erase(PipelineDescr(serialized_pipeline, max_batch_size, args...));
  }

  size_t NumCreatedPipelines() { return created_pipelines_; }

  std::unordered_map<PipelineDescr, DaliPipeline> pool_;
  size_t created_pipelines_ = 0;
};

}}}  // namespace triton::backend::dali


#endif  // DALI_BACKEND_DALI_EXECUTOR_PIPELINE_POOL_H_
