#pragma once

#include <dlfcn.h>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(const fs::path & path);
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();
