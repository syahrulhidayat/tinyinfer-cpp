#include "tinyinfer/mmap_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace tinyinfer {

MappedFile::MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error(
            "MappedFile: failed to open '" + path + "': " + std::strerror(errno)
        );
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        const int err = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "MappedFile: fstat failed for '" + path + "': " + std::strerror(err)
        );
    }

    if (st.st_size <= 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("MappedFile: file is empty: '" + path + "'");
    }

    size_ = static_cast<std::size_t>(st.st_size);
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        ::close(fd_);
        fd_ = -1;
        size_ = 0;
        throw std::runtime_error(
            "MappedFile: mmap failed for '" + path + "': " + std::strerror(err)
        );
    }

#ifdef MADV_SEQUENTIAL
    ::madvise(mapped, size_, MADV_SEQUENTIAL);
#endif
    data_ = static_cast<const std::byte*>(mapped);
}

MappedFile::~MappedFile() {
    close_mapping();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), data_(other.data_), size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close_mapping();
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MappedFile::close_mapping() noexcept {
    if (data_ != nullptr && size_ > 0) {
        ::munmap(const_cast<std::byte*>(data_), size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

}  // namespace tinyinfer
