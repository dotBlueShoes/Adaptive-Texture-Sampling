// Created 2024.05.06 by Matthew Strumiłło (dotBlueShoes)
//  LICENSE: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
//
#pragma once
#include "types.hpp"

void ReverseBits8 (u8& n) {
    n = (n >> 1) & 0x55 | (n << 1) & 0xAA;
    n = (n >> 2) & 0x33 | (n << 2) & 0xCC;
    n = (n >> 4) & 0x0F | (n << 4) & 0xF0;
}

void ReverseBits16 (u16& n) {
    n = (n >> 1) & 0x5555 | (n << 1) & 0xAAAA;
    n = (n >> 2) & 0x3333 | (n << 2) & 0xCCCC;
    n = (n >> 4) & 0x0F0F | (n << 4) & 0xF0F0;
    n = (n >> 8) | (n << 8);
}

void ReverseBits32 (u32& n) {
    n = (n >>  1) & 0x55555555 | (n << 1) & 0xAAAAAAAA;
    n = (n >>  2) & 0x33333333 | (n << 2) & 0xCCCCCCCC;
    n = (n >>  4) & 0x0F0F0F0F | (n << 4) & 0xF0F0F0F0;
    n = (n >>  8) & 0x00FF00FF | (n << 8) & 0xFF00FF00;
    n = (n >> 16) | (n << 16);
}

void ReverseBits64 (u64& n) {
    n = (n >>  1) & 0x5555555555555555ULL | (n <<  1) & 0xAAAAAAAAAAAAAAAAULL;
    n = (n >>  2) & 0x3333333333333333ULL | (n <<  2) & 0xCCCCCCCCCCCCCCCCULL;
    n = (n >>  4) & 0x0F0F0F0F0F0F0F0FULL | (n <<  4) & 0xF0F0F0F0F0F0F0F0ULL;
    n = (n >>  8) & 0x00FF00FF00FF00FFULL | (n <<  8) & 0xFF00FF00FF00FF00ULL;
    n = (n >> 16) & 0x0000FFFF0000FFFFULL | (n << 16) & 0xFFFF0000FFFF0000ULL;
    n = (n >> 32) | (n << 32);
}
