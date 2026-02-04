// Created 2024.11.13 by Matthew Strumiłło (dotBlueShoes)
//  LICENSE: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
//
#pragma once
#include "debug.hpp"

//  'About'
// - Implements a simple wrapper around 'malloc'
// - Implements a custom 'atexit' implementation that allows sending the parameter. 
//


#ifndef MEMORY_EXIT_SIZE
	#define MEMORY_EXIT_SIZE 64
#endif


#ifndef MEMORY_TYPE
	#define MEMORY_TYPE u32
#endif


#define DEALLOC_TYPE void
#define DEALLOC_RET  


#ifdef MEMORY_TYPE_NOT_SIZED
	#define DEALLOC_ARGS INOUT void* data
#else
	#define DEALLOC_ARGS IN	MEMORY_TYPE	size, INOUT void* data
#endif


#if DDEBUG (DEBUG_FLAG_MEMORY)

	s32 allocationsCounter = 0;

	#if DDEBUG (DEBUG_FLAG_LOGGING)

		#include "log.hpp"

		#define LOGMEMORY() { \
			LOGWARN ("Missed Deallocations: %d\n", allocationsCounter); \
		}

	#else
		#define LOGMEMORY() {} // dummy
	#endif

	#define INCALLOCCO() ++allocationsCounter;
	#define DECALLOCCO() --allocationsCounter;


	template <typename Type>
	void _ALLOCATE (Type*& address, const u64& size) {
		++allocationsCounter;
		address = (Type*) malloc (size);
	}

	#define ALLOCATE(type, address, size) _ALLOCATE<type> (address, size)

	void FREE (DEALLOC_ARGS) {
		--allocationsCounter;
		free (data);
	}

#else

	#define ALLOCATE(type, address, size) address = (type*) malloc (size)
	void FREE (DEALLOC_ARGS) { free (data); }
	#define LOGMEMORY() {} // dummy
	#define INCALLOCCO() // dummy
	#define DECALLOCCO() // dummy

#endif



namespace MEMORY {

	template <typename T>
	void Construct2 (
		OUT		T*& 				newArray, 
		IN		const u32& 			aArrayLength, 
		IN		const void* const& 	aArray, 
		IN		const u32& 			bArrayLength, 
		IN		const void* const& 	bArray
	) {
		ALLOCATE (T, newArray, aArrayLength + bArrayLength);
		memcpy (newArray, aArray, aArrayLength);
		memcpy (newArray + aArrayLength, bArray, bArrayLength);
	}

	template <typename T>
	void Construct3 (
		OUT		T*& 				newArray, 
		IN		const u32& 			aArrayLength, 
		IN		const void* const& 	aArray, 
		IN		const u32& 			bArrayLength, 
		IN		const void* const& 	bArray,
		IN		const u32& 			cArrayLength, 
		IN		const void* const& 	cArray
	) {
		ALLOCATE (T, newArray, aArrayLength + bArrayLength + cArrayLength);
		memcpy (newArray, aArray, aArrayLength);
		memcpy (newArray + aArrayLength, bArray, bArrayLength);
		memcpy (newArray + aArrayLength + bArrayLength, cArray, cArrayLength);
	}

}
