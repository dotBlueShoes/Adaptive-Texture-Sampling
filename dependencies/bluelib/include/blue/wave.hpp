// Created 2024.05.06 by Matthew Strumiłło (dotBlueShoes)
//  LICENSE: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
//
#pragma once
#include "types.hpp"
#include "lut.hpp"


//  ABOUT
// Wave types are defined in 4 different ways. Each can be accessed via preprocessor definition.
//  - (default) (avoids bitfields and braches)
//  - WAVE_LUT_OPT (uses lut)
//  - WAVE_BITWISE_SCALING_OPT (might outperform normal multiply in some embedded cases)
//  - WAVE_NO_OPT (uses the easy to understand solution without much opts)

//  TODO
// This code could be further optimized via SIMD use.


class w8 {

	private:

		union {
			u8 value;
			struct {
				u8 wave 	: 6;
				u8 phase 	: 1;
				u8 sign 	: 1;
			} bitmask;
		} base;

	public:

		w8 (u8 value) : base(value) {};

		r32 real32 () {

			union {
				r32 value;
				union {
					u32 value;
					struct {
						u32 mantissa 			: 23;
						u32 exponentWave		: 6;
						u32 exponentPhase		: 1;
						u32 exponentReserved	: 1;
						u32 sign 				: 1;
					} bitmask;
				} base;
			} real;

			const r32 STEP6 = 1.0f / 64.0f;

			#ifdef WAVE_NO_OPT
				if (base.bitmask.phase) { real.value = (~base.bitmask.wave + 64 + 1) * STEP6; } // 65 -> 127
				else { real.value = base.bitmask.wave * STEP6; } // for 0 -> 63
				real.base.bitmask.sign = base.bitmask.sign;
			#else
				const auto& value = base.value;			// Don't use bitfields.
				u8 wave 	= value & 0x3F;				// Extract wave 	(bits 0-5)
				u8 phase 	= (value >> 6) & 1;			// Extract phase 	(bit  6  )
				u8 sign 	= (value >> 7);				// Extract sign 	(bit  7  )

				#ifdef WAVE_LUT_OPT
					u8 phasedWave = phase ? (64 - wave) : wave;		// Brachless
					real.value = DIV64LUT[phasedWave];				// LUT
				#else
					u8 phasedWave = phase ? (~wave - 191) : wave; 	// Brachless
					#ifdef WAVE_BITWISE_SCALING_OPT
						// Weird. Might be faster.
						real.value = (r32)(phasedWave << 18) * 0x1p-24f;
					#else
						real.value = (r32)phasedWave * STEP6;		// u8 to r32 cast.
					#endif
				#endif

				// Set sign bit directly without the use of bitfields.
				((u32*)&real.value)[0] = (((u32*)&real.value)[0] & 0x7FFFFFFF) | (sign << 31);
			#endif

			return real.value;
		}

};
