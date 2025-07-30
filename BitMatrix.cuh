#ifndef BITMATRIX_CUH
#define BITMATRIX_CUH

#include "Common.cuh"

class BitMatrix
{
public:
    BitMatrix() = default;
    BitMatrix(const BitMatrix& other) = delete;
    BitMatrix(BitMatrix&& other) noexcept = delete;
    BitMatrix& operator=(const BitMatrix& other) = delete;
    BitMatrix& operator=(BitMatrix&& other) noexcept = delete;
    virtual ~BitMatrix() = default;

    virtual void save(std::string filename) = 0;
};

#endif
