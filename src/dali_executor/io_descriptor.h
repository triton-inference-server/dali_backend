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

namespace triton { namespace backend { namespace dali {

struct StorageCpu {
};
struct StorageGpu {
};


template <typename StorageBackend = StorageCpu>
void
allocate(void** ptr, size_t size)
{
  *ptr = malloc(size);
}


template <>
void
allocate<StorageGpu>(void** ptr, size_t size)
{
  // TODO
}


template <typename StorageBackend = StorageCpu>
void
deallocate(void* ptr)
{
  free(ptr);
}


template <>
void
deallocate<StorageGpu>(void* ptr)
{
  // TODO
}


template <typename StorageBackend = StorageCpu>
void
copy(void* dst, const void* src, size_t count)
{
  memcpy(dst, src, count);
}


template <>
void
copy<StorageGpu>(void* dst, const void* src, size_t count)
{
  // TODO
}


template <typename T, typename StorageBackend = StorageCpu>
struct IODescriptor {
 public:
  IODescriptor() = default;


  IODescriptor(size_t cap) : owns_memory(true), owned_mem_capacity(cap)
  {
    alloc();
  }


  ~IODescriptor() { dealloc(); }


  IODescriptor(const IODescriptor&) = delete;


  IODescriptor(IODescriptor&& other) noexcept
      : owns_memory(true), name(""), type(static_cast<dali_data_type_t>(0)),
        device(static_cast<device_type_t>(0)), shape({}), buffer({}),
        owned_mem(nullptr), owned_mem_size(0), owned_mem_capacity(0)
  {
    *this = std::move(other);
  }


  IODescriptor& operator=(const IODescriptor&) = delete;


  IODescriptor& operator=(IODescriptor&& other)
  {
    if (this != &other) {
      delete[] owned_mem;

      owns_memory = other.owns_memory;
      name = other.name;
      type = other.type;
      device = other.device;
      shape = other.shape;
      buffer = other.buffer;
      owned_mem = other.owned_mem;
      owned_mem_size = other.owned_mem_size;
      owned_mem_capacity = other.owned_mem_capacity;

      other.owns_memory = true;
      other.name = "";
      other.type = static_cast<dali_data_type_t>(0);
      other.device = static_cast<device_type_t>(0);
      other.shape = {};
      other.buffer = {};
      other.owned_mem = nullptr;
      other.owned_mem_size = 0;
      other.owned_mem_capacity = 0;
    }
    return *this;
  }


  bool owns_memory = false;
  std::string name;
  dali_data_type_t type;
  device_type_t device;
  TensorListShape<> shape;
  span<T> buffer;


  void append(span<const T> buffer) { append(buffer.data(), buffer.size()); }


  void append(const T* buffer, size_t size)
  {
    assert(owns_memory);
    assert(size + owned_mem_size <= owned_mem_capacity);
    copy<StorageBackend>(owned_mem + owned_mem_size, buffer, size * sizeof(T));
    regenerate_buffer();
  }


 protected:
  using U = typename std::remove_cv<T>::type;
  U* owned_mem;
  size_t owned_mem_size, owned_mem_capacity;

 private:
  void regenerate_buffer() { buffer = make_span(owned_mem, owned_mem_size); }


  void alloc()
  {
    allocate<StorageBackend>(
        reinterpret_cast<void**>(&owned_mem), owned_mem_capacity);
  }


  void dealloc()
  {
    deallocate<StorageBackend>(reinterpret_cast<void**>(&owned_mem));
  }
};

using InputDescriptor = IODescriptor<const char>;
using OutputDescriptor = IODescriptor<char>;

}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_IO_DESCRIPTOR_H
