#ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
#define __PACKET_ALIGNER_PREPROCESS_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <array>

#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

// utility classes
//#include "UnrolledLoop.hpp"

template <unsigned min_msg_len,   // minimum message length, in bytes
          unsigned hdr_len,       // length of header, including length field
          unsigned len_start,     // lengh starts at byte len_start and goes to
                                  // the end of the header
          const std::array<bool,hdr_len> &hdr_mask,   // true = compare this byte to hdr_val
          const std::array<uint8_t,hdr_len> &hdr_val  // expected value of header at each byte pos
          >
struct ProtocolBase {

  constexpr static unsigned kMinMsgLen = min_msg_len;
  constexpr static unsigned kHdrLen = hdr_len;
  constexpr static unsigned kLenStart = len_start;
  constexpr static std::array<bool,hdr_len> kHdrMask = hdr_mask;
  constexpr static std::array<uint8_t,hdr_len> kHdrVal = hdr_val;

};  // end of struct ProtocolBase

template <unsigned bus_width,           // width of the data bus, in bytes
          unsigned channel_bit_width    // width of the channel field, in bits
          >
struct PacketBusBase {
  // TODO check that bus_width < 256, so we can use uint8_t

  using ChannelType = ac_int<channel_bit_width, false>;

  constexpr static unsigned kBusWidth = bus_width;

  bool valid;                 // indicates if contents are valid
  uint8_t data[bus_width];    // data bytes
  ChannelType channel;        // channel number
  bool sop;                   // this word is the start of a TCP packet
  bool eop;                   // this word is the end of a TCP packet
  uint8_t num_valid_bytes;    // number of bytes (starting at 0) that are valid
                              // only applies when eop = true

};  // end of struct PacketBusBase


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
  // TODO Template parameter checking



  // kernel code
  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<PacketAlignerPreprocessKernelName>([=] {
      while (1) {
        
        // TODO temp code for testing
        if( Protocol::kMinMsgLen > 10) {break;};
        if(PacketBus::kBusWidth < 10) {break;};

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitPacketAlignerPreprocessKernel()

#endif  // ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
