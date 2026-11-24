#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tinyinfer {

// RAII read-only memory-mapped file for zero-copy weight access.
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool is_open() const noexcept { return data_ != nullptr; }

private:
    void close_mapping() noexcept;

    int fd_ = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace tinyinfer
