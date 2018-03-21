#ifndef TIMER_HEADER
#define TIMER_HEADER

#include <ctime>
#ifdef _WIN32
#   include "windows.h"
#else
#   include <sys/time.h>
#endif

class WallTimer
{
private:
#ifdef _WIN32
    std::size_t start;
#else
    struct timeval start;
#endif
    
public:
    WallTimer (void);
    void reset (void);
    
    /** Returns the milli seconds since last reset. */
    std::size_t get_elapsed (void) const;
    
    /** Returns the seconds since last reset. */
    float get_elapsed_sec (void) const;
};


/* ---------------------------------------------------------------- */

inline
WallTimer::WallTimer (void)
{
    this->reset();
}

inline void
WallTimer::reset (void)
{
#ifdef _WIN32
    // FIXME: ::GetTickCount has poor precision (~10 - 16ms)
    this->start = ::GetTickCount();
#else
    ::gettimeofday(&this->start, NULL);
#endif
}

inline std::size_t
WallTimer::get_elapsed (void) const
{
#ifdef _WIN32
    return ::GetTickCount() - start;
#else
    struct timeval cur_time;
    ::gettimeofday(&cur_time, NULL);
    std::size_t ret = (cur_time.tv_sec - start.tv_sec) * 1000;
    std::size_t cur_ms = cur_time.tv_usec / 1000;
    std::size_t start_ms = start.tv_usec / 1000;
    if (cur_ms >= start_ms)
        ret += (cur_ms - start_ms);
    else
        ret -= (start_ms - cur_ms);
    return ret;
#endif
}

inline float
WallTimer::get_elapsed_sec (void) const
{
    return static_cast<float>(this->get_elapsed()) / 1000.0f;
}

#endif /* TIMER_HEADER */
