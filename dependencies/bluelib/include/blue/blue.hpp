#pragma once
#include "error.hpp"

#define BINIT(msg) { \
    TIMESTAMP_BEGIN = TIMESTAMP::GetCurrent (); \
    DEBUG (DEBUG_FLAG_LOGGING) putc ('\n', stdout); \
    LOGINFO (msg); \
}


#define BSTOP(msg) { \
    LOGMEMORY (); \
	LOGINFO (msg); \
	DEBUG (DEBUG_FLAG_LOGGING) putc ('\n', stdout); \
}
