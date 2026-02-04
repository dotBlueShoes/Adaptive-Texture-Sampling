// Created 2025.05.01 by Matthew Strumiłło (dotBlueShoes)
//  LICENSE: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
//
#pragma once
#include "memory.hpp"


//  ABOUT
// We're defining 2 systems. SIZED and NOT_SIZED. It's important to note that both connot be used at the same time.
//  Because of allocation order importance. Therefore as a programmer u have to decide on one and 
//  ensure reusing that system in every dealloc method.

//  ABOUT - MEMORY_TYPE_NOT_SIZED
// By default use not sized variant for memory exiting. 
// We cannot use both systems because it's important to dealloc memory in order of allocation!
// #define MEMORY_TYPE_NOT_SIZED 

#ifdef MEMORY_TYPE_NOT_SIZED

	namespace MEMORY::EXIT {

		using DeleteFunction = void (*) (DEALLOC_ARGS);

		u8 memoryCounter = 0;

		DeleteFunction functions [MEMORY_EXIT_SIZE];
		void* memories	[MEMORY_EXIT_SIZE];

		void ATEXIT () {
			for (u8 i = memoryCounter; i != 0; --i) {
				auto& func		= functions	[i - 1];
				auto& memory	= memories	[i - 1];

				auto&& _func = (DeleteFunction)func;
				_func (memory);
			}
		}

		void PUSH (
			DeleteFunction function, 
			void* memory
		) {
			functions[memoryCounter] 	= function;
			memories[memoryCounter] 	= memory;

			++memoryCounter;
		}

		void POP () {
			--memoryCounter;
		}

	}

#else

	namespace MEMORY::EXIT {

		u8 memoryCounter = 0;

		using DeleteFunction = void (*) (DEALLOC_ARGS);

		DeleteFunction functions [MEMORY_EXIT_SIZE];
		void* memories	[MEMORY_EXIT_SIZE];
		u32   sizes		[MEMORY_EXIT_SIZE];

		void ATEXIT () {
			for (u8 i = memoryCounter; i != 0; --i) {
				auto& func		= functions	[i - 1];
				auto& memory	= memories	[i - 1];
				auto& size		= sizes 	[i - 1];

				auto&& _func = (DeleteFunction)func;
				_func (size, memory);
			}
		}

		
		void PUSH (
			DeleteFunction function, 
			MEMORY_TYPE size, 
			void* memory
		) {
			functions[memoryCounter] 	= function;
			memories[memoryCounter] 	= memory;
			sizes[memoryCounter] 		= size;

			++memoryCounter;
		}

		void POP () {
			--memoryCounter;
		}

	}

#endif

#define DEALLOC(name, body) \
    DEALLOC_TYPE name (DEALLOC_ARGS) { \
        body \
        DEALLOC_RET \
    }
//#define end
