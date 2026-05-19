#include "llama-mmap.h"
#include "llama-impl.h"
#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/resource.h>

// Portable ftell/fseek helpers
#define llama_mmap_ftell ftello
#define llama_mmap_fseek fseeko

// Linux-only implementation of llama_file
struct llama_file::impl {
    FILE *fp = nullptr;
    bool owns_fp = true;
    std::string fname;
    size_t size = 0;

    impl(const char *fname_, const char *mode, bool /*use_direct_io*/ = false) {
        fname = fname_;
        fp = fopen(fname_, mode);
        if (!fp) {
            throw std::runtime_error("failed to open " + std::string(fname_) + ": " + strerror(errno));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE *file) : fp(file), owns_fp(false), fname("(FILE*)") {
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        off_t off = llama_mmap_ftell(fp);
        if (off == -1) throw std::runtime_error("ftell error: " + std::string(strerror(errno)));
        return static_cast<size_t>(off);
    }

    void seek(size_t offset, int whence) const {
        if (llama_mmap_fseek(fp, static_cast<off_t>(offset), whence) != 0) {
            throw std::runtime_error("seek error: " + std::string(strerror(errno)));
        }
    }

    void read_raw_unsafe(void *ptr, size_t len) {
        if (len == 0) return;
        size_t cur = tell();
        size_t to_read = std::min(len, size - cur);
        if (to_read && std::fread(ptr, to_read, 1, fp) != 1) {
            throw std::runtime_error("read error: " + std::string(strerror(errno)));
        }
    }

    void write_raw_unsafe(const void *ptr, size_t len) const {
        if (len == 0) return;
        size_t cur = tell();
        size_t to_write = std::min(len, size - cur);
        if (to_write && std::fwrite(ptr, to_write, 1, fp) != 1) {
            throw std::runtime_error("write error: " + std::string(strerror(errno)));
        }
    }

    void flush() {
        if (fflush(fp) != 0) throw std::runtime_error("fflush error: " + std::string(strerror(errno)));
    }

    void *map(void *addr, size_t len, bool write) const {
        int prot = PROT_READ | (write ? PROT_WRITE : 0);
        void *p = mmap(addr, len, prot, MAP_SHARED, fileno(fp), 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap error: " + std::string(strerror(errno)));
        return p;
    }

    void unmap(void *ptr, size_t len) {
        if (munmap(ptr, len) != 0) throw std::runtime_error("munmap error: " + std::string(strerror(errno)));
    }

    ~impl() { if (fp && owns_fp) fclose(fp); }
};

// Public wrapper implementations
llama_file::llama_file(const char *fname, const char *mode, bool use_direct_io)
    : pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

llama_file::llama_file(FILE *file) : pimpl(std::make_unique<impl>(file)) {}

llama_file::~llama_file() = default;

size_t llama_file::size() const { return pimpl->size; }
size_t llama_file::tell() const { return pimpl->tell(); }
int llama_file::file_id() const { return fileno(pimpl->fp); }
void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void *ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }
void llama_file::read_raw_unsafe(void *ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }
void llama_file::read_aligned_chunk(void *dest, size_t size) { pimpl->read_raw_unsafe(dest, size); }
uint32_t llama_file::read_u32() { uint32_t v; pimpl->read_raw_unsafe(&v, sizeof(v)); return v; }
void llama_file::write_raw(const void *ptr, size_t len) const { pimpl->write_raw_unsafe(ptr, len); }
void llama_file::write_u32(uint32_t v) const { pimpl->write_raw_unsafe(&v, sizeof(v)); }
size_t llama_file::read_alignment() const { return 1; }
bool llama_file::has_direct_io() const { return false; }

// Minimal mmap and mlock implementations (Linux only)
struct llama_mmap::impl {
    const llama_file *file;
    void *addr = nullptr;
    size_t sz = 0;
    bool numa = false;
    impl(const llama_file *f, size_t /*prefetch*/, bool /*n*/) : file(f) {
        sz = f->size();
        addr = nullptr; // No actual mapping needed for CPU‑only build
    }
    ~impl() { if (addr) munmap(addr, sz); }
    size_t size() const { return sz; }
    void *addr_ptr() const { return addr; }
    void unmap_fragment(size_t, size_t) { /* no-op */ }
};

llama_mmap::llama_mmap(const llama_file *file, size_t prefetch, bool numa)
    : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}

llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size(); }
void *llama_mmap::addr() const { return pimpl->addr_ptr(); }
void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

const bool llama_mmap::SUPPORTED = true;

// Minimal mlock implementation
struct llama_mlock::impl {
    void *ptr = nullptr;
    void init(void *p) { ptr = p; }
    void grow_to(size_t) { /* no-op */ }
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;
void llama_mlock::init(void *ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t sz) { pimpl->grow_to(sz); }
const bool llama_mlock::SUPPORTED = true;

size_t llama_path_max() { return 4096; }
