#ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
#define __PACKET_ALIGNER_PREPROCESS_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <array>
#include <vector> // TODO temp

// utility classes
//#include "UnrolledLoop.hpp"

template <unsigned min_msg_len,   // minimum message length, in bytes
          unsigned hdr_len,       // length of header, including length field
          unsigned len_start,     // lengh starts at byte len_start and goes to
                                  // the end of the header
          const std::array<bool,hdr_len> &hdr_mask, // true = compare this byte to hdr_val
          const std::array<char,hdr_len> &hdr_val   // expected value of header at each byte pos
          >
struct ProtocolBase {

  constexpr static unsigned kMinMsgLen = min_msg_len;
  constexpr static unsigned kHdrLen = hdr_len;
  constexpr static unsigned kLenStart = len_start;
  constexpr static std::array<bool,hdr_len> kHdrMask = hdr_mask;
  constexpr static std::array<char,hdr_len> kHdrVal = hdr_val;

};  // end of struct ProtocolBase




// SubmitPacketAlignerPreprocessKernel
// TODO description
template <typename PacketAlignerPreprocessKernelName,  // Name for the Kernel

          typename PacketBus,     // Struct containing all data and metadata
                                  // used as input/output for this kernel.
          typename Protocol,      // Protocol definition, must be a type
                                  // derrived from ProtocolBase.
          typename PacketInPipe,  // Receive data words bundled in a PacketBus
                                  // struct.
          typename PacketOutPipe, // Send data words bundled in a PacketBus
                                  // struct.  Feed through copy of data read
                                  // from PacketInPipe.
          typename PacketInfoOutPipe  // Send metadata about each data
                                      // word, used to ease downstream
                                      // processing
          >
sycl::event SubmitPacketAlignerPreprocessKernel(sycl::queue& q) {
  // Template parameter checking



  // kernel code
  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<PacketAlignerPreprocessKernelName>([=] {
      while (1) {

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitPacketAlignerPreprocessKernel()

#endif  // ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
