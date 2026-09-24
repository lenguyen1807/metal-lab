#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gemm/matrix.h"

inline std::string read_file(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("Cannot read file");
    }

    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    file.ignore(std::numeric_limits<std::streamsize>::max());

    try {
        auto size = file.gcount();

        if (size > 0x10000)  // 64kib sanity check for shaders:
            return std::string();

        file.clear();
        file.seekg(0, std::ios_base::beg);

        std::stringstream sstr;
        sstr << file.rdbuf();
        file.close();

        return sstr.str();
    } catch (const std::ifstream::failure& e) {
        throw std::runtime_error("cannot read file: " + path + " (" + e.what() + ")");
    }
}

// https://stackoverflow.com/questions/25201131/writing-csv-files-from-c
// https://gist.github.com/rudolfovich/f250900f1a833e715260a66c87369d15
class CSVWriter {
    std::ofstream fs_;
    bool is_first_;
    const std::string separator_;
    const std::string escape_seq_;
    const std::string special_chars_;

   public:
    CSVWriter(const std::string filename, const std::string separator = ",")
        : fs_(), is_first_(true), separator_(separator), escape_seq_("\""), special_chars_("\"") {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        std::filesystem::create_directories(OUTPUTS_PATH);
        fs_.open(std::string(OUTPUTS_PATH) + filename, std::fstream::out | std::fstream::trunc);
    }

    ~CSVWriter() {
        flush();
        fs_.close();
    }

    void flush() { fs_.flush(); }

    void endrow() {
        fs_ << std::endl;
        is_first_ = true;
    }

    CSVWriter& operator<<(CSVWriter& (*val)(CSVWriter&)) { return val(*this); }

    CSVWriter& operator<<(const char* val) { return write(escape(val)); }

    CSVWriter& operator<<(const std::string& val) { return write(escape(val)); }

    template <typename T>
    CSVWriter& operator<<(const T& val) {
        return write(val);
    }

   private:
    template <typename T>
    CSVWriter& write(const T& val) {
        if (!is_first_) {
            fs_ << separator_;
        } else {
            is_first_ = false;
        }
        fs_ << val;
        return *this;
    }

    std::string escape(const std::string& val) {
        std::ostringstream result;
        result << '"';
        std::string::size_type to, from = 0u, len = val.length();
        while (from < len && std::string::npos != (to = val.find_first_of(special_chars_, from))) {
            result << val.substr(from, to - from) << escape_seq_ << val[to];
            from = to + 1;
        }
        result << val.substr(from) << '"';
        return result.str();
    }
};

inline static CSVWriter& endrow(CSVWriter& file) {
    file.endrow();
    return file;
}

inline static CSVWriter& flush(CSVWriter& file) {
    file.flush();
    return file;
}

inline double matmul_time_to_gflops(double rows, double cols, double inner_dim,
                                    double milliseconds) {
    return 2.0 * rows * cols * inner_dim / (milliseconds * 1e6);
}

inline void copy(const HostMatrix& src, DeviceMatrix& dst) {
    assert(src.rows == dst.rows);
    assert(src.cols == dst.cols);
    const size_t byte_size = src.rows * src.cols * sizeof(float);
    // On unified memory, host_data() and device_data()->contents() point to the
    // same region. A memcpy is the most direct way to express the copy
    // operation.
    std::memcpy(static_cast<float*>(dst.data()->contents()), src.data(), byte_size);
}

struct ResultComparison {
    size_t mismatches = 0;
    double max_abs_error = 0.0;

    static ResultComparison compare(const float* actual_data, const float* expected_data,
                                    size_t count) {
        ResultComparison result;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(actual_data[i]) || !std::isfinite(expected_data[i])) {
                result.max_abs_error = std::numeric_limits<double>::infinity();
                ++result.mismatches;
                continue;
            }
            const double reference = expected_data[i];
            const double error = std::abs(static_cast<double>(actual_data[i]) - reference);
            result.max_abs_error = std::max(result.max_abs_error, error);
            if (error > 1e-3 + 1e-3 * std::abs(reference)) {
                ++result.mismatches;
            }
        }
        return result;
    }

    static ResultComparison compare(const DeviceMatrix& actual, const DeviceMatrix& expected) {
        assert(actual.cols == expected.cols);
        assert(actual.rows == expected.rows);
        return compare(actual.host_data(), expected.host_data(), actual.rows * actual.cols);
    }
};
