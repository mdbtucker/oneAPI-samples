#ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
#define __PACKET_ALIGNER_PREPROCESS_HPP__

// Note on 'NO-FORMAT' comments: these are present to prevent clang-format from
// messing up code formatting.  They are commonly applied to attributes, which
// clang-format does not seem to support well


#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <array>

#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

// utility classes
#include "UnrolledLoop.hpp"
#include "ParallelCopyArray.hpp"
#include "mp_math.hpp"

template <unsigned min_msg_len,   // minimum message length, in bytes
          unsigned hdr_len,       // length of header in bytes, including length
                                  // field
          unsigned len_start,     // lengh starts at byte len_start and goes to
                                  // the end of the header
          const std::array<bool,hdr_len> &hdr_mask,   // true = compare this 
                                                      // byte to hdr_val
          const std::array<uint8_t,hdr_len> &hdr_val  // expected value of 
                                                      // header at each byte
          >
struct ProtocolBase {

  constexpr static unsigned kMinMsgLen = min_msg_len;
  constexpr static unsigned kHdrLen = hdr_len;
  constexpr static unsigned kLenStart = len_start;
  constexpr static unsigned kLengthLen = hdr_len - len_start;
  constexpr static std::array<bool,hdr_len> kHdrMask = hdr_mask;
  constexpr static std::array<uint8_t,hdr_len> kHdrVal = hdr_val;

};  // end of struct ProtocolBase

template <unsigned bus_width,           // width of the data bus, in bytes
          unsigned channel_width_bits   // width of the channel field, in bits
          >
struct PacketBusBase {
  // TODO check that bus_width < 256, so we can use uint8_t

  using ChannelType = ac_int<channel_width_bits, false>;
  constexpr static unsigned kNumChannels = 0x01 << channel_width_bits;
  constexpr static unsigned kBusWidth = bus_width;
  constexpr static unsigned kBusPosTypeWidthBits = 
    hldutils::CeilLog2(bus_width);
  using BusPosType = ac_int<kBusPosTypeWidthBits, false>;

  uint8_t data[bus_width];    // data bytes
  uint8_t num_valid_bytes;    // number of bytes (starting at 0) that are valid
                              // only applies when eop = true
  bool sop;                   // this word is the start of a TCP packet
  bool eop;                   // this word is the end of a TCP packet
  ChannelType channel;        // channel number

};  // end of struct PacketBusBase

template <typename Protocol,
          typename PacketBus
          >
struct PacketInfoBase {
  bool header_match[PacketBus::kBusWidth + Protocol::kHdrLen - 1];
};


// SubmitPacketAlignerPreprocessKernel
// TODO description
template <typename PacketAlignerPreprocessKernelName,  // Name for the Kernel

          typename Protocol,      // Protocol definition, must be a type
                                  // derrived from ProtocolBase.
          typename PacketBus,     // Struct containing all data and metadata
                                  // used as input/output for this kernel.
          typename PacketInfo,    // Struct containing metadata calculated for
                                  // each packet word received
          typename PacketInPipe,  // Receive data words bundled in a PacketBus
                                  // struct.
          typename PacketOutPipe, // Send data words bundled in a PacketBus
                                  // struct.  Feed through copy of data read
                                  // from PacketInPipe.
          typename PacketInfoOutPipe  // Send metadata about each data word 
                                      // bundled in a PacketInfo struct, used to
                                      // ease downstream processing
          >
sycl::event SubmitPacketAlignerPreprocessKernel(sycl::queue& q) {
  // TODO Template parameter checking



  // kernel code
  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<PacketAlignerPreprocessKernelName>([=] {
      
      // Storage for the 'tail' from the previous packet from the same channel
      // This is used for the case where a header straddles two packets
      constexpr unsigned kTailLen = Protocol::kHdrLen - 1;
      using TailType = ParallelCopyArray<uint8_t, kTailLen>;
      TailType prev_word_tail[PacketBus::kNumChannels];

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      while (1) {

        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        PacketBus packet_in;    // current word being processed
  
        bool packet_in_valid;   // valid flag for current word
  
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        PacketInfo packet_info; // metadata calculated for each packet word

        // combine current data with tail from previous word from same channel
        constexpr unsigned kDataWithPrevTailSize = PacketBus::kBusWidth + 
          kTailLen;
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        uint8_t data_with_prev_tail[kDataWithPrevTailSize];

        // read a word from the PacketInPipe, if available
        packet_in = PacketInPipe::read(packet_in_valid);

        // retreive the 'tail' of the previous word from this channel
        TailType prev_tail = prev_word_tail[packet_in.channel];
        UnrolledLoop<kTailLen>([&](auto i) {
          data_with_prev_tail[i] = prev_tail[i];
        });

        // combine the data from this packet with the previous word tail
        UnrolledLoop<PacketBus::kBusWidth>([&](auto i) {
          data_with_prev_tail[i + kTailLen] = packet_in.data[i];
        });

        // store the tail from this packet
        TailType new_tail;
        constexpr unsigned kTailStartIndex = PacketBus::kBusWidth - kTailLen;
        UnrolledLoop<kTailLen>([&](auto i) {
          //new_tail[i] = packet_in.data[i + kTailStartIndex];
          prev_word_tail[packet_in.channel][i] = 
            packet_in.data[i + kTailStartIndex];
        });

        // check each byte position for a potentially valid header
        UnrolledLoop<kDataWithPrevTailSize>([&](auto i) {
          bool cur_pos_header_match = true;
          UnrolledLoop<Protocol::kLenStart>([&](auto j) {
            constexpr auto position = i + j;
            if constexpr ((position < kDataWithPrevTailSize) && 
                          Protocol::kHdrMask[j]) {
              cur_pos_header_match &= 
                (data_with_prev_tail[position] == Protocol::kHdrVal[j]);
            }
          });
          packet_info.header_match[i] = cur_pos_header_match;
        });

        // determine end of message position assuming each byte is the start
        // of the length field in a new message

        constexpr unsigned kLastLenStartPos = 
          PacketBus::kBusWidth - Protocol::kLengthLen;

        using MsgLenType = ac_int<Protocol::kLengthLen * 8, false>;
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        MsgLenType msg_end_word[kLastLenStartPos - Protocol::kLenStart + 1];
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        typename PacketBus::BusPosType msg_end_byte[kLastLenStartPos - 
                                                    Protocol::kLenStart + 1];

        UnrolledLoop<Protocol::kLenStart, kLastLenStartPos>([&](auto i) {
          MsgLenType length = 0;
          UnrolledLoop<Protocol::kLengthLen>([&](auto j) {
            // TODO support opposite endianness here?
            constexpr unsigned shift_bits = (Protocol::kLengthLen - j - 1) * 8;
            length.set_slc(
              shift_bits, (ac_int<8,false>)data_with_prev_tail[i+j]);
//            length |= ((MsgLenType)data_with_prev_tail[i+j]) << shift_bits;
          });
          msg_end_word[i - Protocol::kLenStart] = length >> 
            PacketBus::kBusPosTypeWidthBits;
          msg_end_byte[i - Protocol::kLenStart] = 
            length.template slc<PacketBus::kBusPosTypeWidthBits>(0);
        });

        // write the metadata and input word out for downstream processing
        if (packet_in_valid) {
          PacketInfoOutPipe::write(packet_info);
          PacketOutPipe::write(packet_in);
        }


      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitPacketAlignerPreprocessKernel()

#endif  // ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
