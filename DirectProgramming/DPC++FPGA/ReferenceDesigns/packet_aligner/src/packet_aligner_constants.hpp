#ifndef __PACKET_ALIGNER_CONSTANTS_HPP__
#define __PACKET_ALIGNER_CONSTANTS_HPP__

// Allow design parameters to be defined on the command line

// check is USM host allocations are enabled
#ifdef USM_HOST_ALLOCATIONS
constexpr bool kUseUSMHostAllocation = true;
#else
constexpr bool kUseUSMHostAllocation = false;
#endif


#endif  // __PACKET_ALIGNER_CONSTANTS_HPP__
