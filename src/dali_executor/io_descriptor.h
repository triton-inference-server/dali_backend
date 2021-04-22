// The MIT License (MIT)
//
// Copyright (c) 2021 NVIDIA CORPORATION
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

#ifndef TRITONDALIBACKEND_IO_DESCRIPTOR_H
#define TRITONDALIBACKEND_IO_DESCRIPTOR_H

#include "src/dali_executor/utils/dali.h"
#include "src/error_handling.h"

namespace triton { namespace backend { namespace dali {

struct StorageCpu {};
struct StorageGpu {};

namespace detail {


template<typename StorageBackend = StorageCpu>
void allocate(void** ptr, size_t size) {
  *ptr = malloc(size);
  if (!*ptr) {
    throw std::bad_alloc();
  }
}


template<>
inline void allocate<StorageGpu>(void** ptr, size_t size) {
  CUDA_CALL_GUARD(cudaMalloc(ptr, size));
}


template<typename StorageBackend = StorageCpu>
void deallocate(void* ptr) {
  free(ptr);
}


template<>
inline void deallocate<StorageGpu>(void* ptr) {
  CUDA_CALL_GUARD(cudaFree(ptr));
}


template<typename StorageBackend = StorageCpu>
void copy(void* dst, const void* src, size_t count) {
  memcpy(dst, src, count);
}


template<>
inline void copy<StorageGpu>(void* dst, const void* src, size_t count) {
  CUDA_CALL_GUARD(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
}
}  // namespace detail

template<typename T>
class IODescriptorBase {
 public:
  std::string name;
  dali_data_type_t type;
  device_type_t device;
  int device_id;
  TensorListShape<> shape;
  span<T> buffer;
};

static_assert(std::is_move_constructible<IODescriptorBase<char>>::value &&
              std::is_move_assignable<IODescriptorBase<char>>::value);

template<typename T, typename StorageBackend, bool owns_memory = false>
class IODescriptor : public IODescriptorBase<T> {
 public:
  using IODescriptorBase<T>::name;
  using IODescriptorBase<T>::type;
  using IODescriptorBase<T>::device;
  using IODescriptorBase<T>::device_id;
  using IODescriptorBase<T>::shape;
  using IODescriptorBase<T>::buffer;
};


template<typename T, typename StorageBackend>
class IODescriptor<T, StorageBackend, true> : public IODescriptorBase<T> {
 public:
  IODescriptor() = default;

  explicit IODescriptor(size_t cap) : owned_mem_capacity(cap) {
    alloc();
  }

  ~IODescriptor() {
    dealloc();
  }

  IODescriptor(const IODescriptor&) = delete;

  IODescriptor(IODescriptor&& other) noexcept :
      owned_mem(nullptr), owned_mem_size(0), owned_mem_capacity(0) {
    *this = std::move(other);
  }

  IODescriptor& operator=(const IODescriptor&) = delete;

  IODescriptor& operator=(IODescriptor&& other) {
    if (this != &other) {
      dealloc();

      IODescriptorBase<T>::operator=(std::move(other));

      owned_mem = other.owned_mem;
      owned_mem_size = other.owned_mem_size;
      owned_mem_capacity = other.owned_mem_capacity;

      other.owned_mem = nullptr;
      other.owned_mem_size = 0;
      other.owned_mem_capacity = 0;
    }
    return *this;
  }

  void append(span<const T> buffer) {
    append(buffer.data(), buffer.size());
  }


  void append(const T* buffer, size_t size) {
    assert(size + owned_mem_size <= owned_mem_capacity);
    detail::copy<StorageBackend>(reinterpret_cast<uint8_t*>(owned_mem) + owned_mem_size, buffer,
                                 size * sizeof(T));
    owned_mem_size += size;
    regenerate_buffer();
  }

  size_t size() const noexcept {
    return owned_mem_size;
  }
  size_t capacity() const noexcept {
    return owned_mem_capacity;
  }


  using IODescriptorBase<T>::name;
  using IODescriptorBase<T>::type;
  using IODescriptorBase<T>::device;
  using IODescriptorBase<T>::device_id;
  using IODescriptorBase<T>::shape;
  using IODescriptorBase<T>::buffer;

 protected:
  void* owned_mem = nullptr;
  size_t owned_mem_size = 0;      /// In bytes
  size_t owned_mem_capacity = 0;  /// In bytes

 private:
  void regenerate_buffer() {
    assert(owned_mem_size % sizeof(T) == 0);
    buffer = make_span(reinterpret_cast<T*>(owned_mem), owned_mem_size / sizeof(T));
  }


  void alloc() {
    detail::allocate<StorageBackend>(&owned_mem, owned_mem_capacity);
  }


  void dealloc() {
    detail::deallocate<StorageBackend>(owned_mem);
  }
};

template<bool owns_memory>
using IODescr = IODescriptor<char, StorageCpu, owns_memory>;


}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_IO_DESCRIPTOR_H
