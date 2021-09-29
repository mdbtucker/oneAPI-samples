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
  // TODO convert this back to ac_int of exact size?
//  using BusPosType = ac_int<kBusPosTypeWidthBits, false>;
  using BusPosType = uint8_t;

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
  // Tail is 1 byte shorter than a full header, long enough to store any
  // partial header but not a full header
  constexpr static unsigned kTailLen = Protocol::kHdrLen - 1;
  constexpr static unsigned kDataWithPrevTailSize = PacketBus::kBusWidth + 
                                                    kTailLen;
  
  // No need to check for header matches for the final kTailLen bytes of a
  // packet word, as there are not enough bytes for a full header so we can
  // never have a match.
  constexpr static unsigned kHeaderMatchLen = kDataWithPrevTailSize - kTailLen;
  
  // Only need to calculate next_msg values for bytes that could be the start
  // of a new full header field that fits entirely in the current word,
  // therefore we truncate the last few bytes
  constexpr static unsigned kNextMsgSize = kDataWithPrevTailSize - kTailLen;
  
  // maximum number of messages that can fit in a word (including the tail)
  constexpr static unsigned kMaxMsgsPerWord = kDataWithPrevTailSize / 
                                              Protocol::kMinMsgLen;

  // TODO much of this is temporary to allow viewing of intermediate results
  
  // header_match[0] refers to a match at the start of the previous word tail
  // header_match[kTailLen] refers to a match at the start of the current word
  // The final kTailLen bytes of the current word are not relevant, and no
  // header_match is calculated for them.
  bool header_match[kHeaderMatchLen];
  
  unsigned next_msg_start_word[kNextMsgSize] ;
  unsigned next_msg_start_byte[kNextMsgSize];
  bool next_msg_start_same_word[kNextMsgSize];
};  // end of struct PacketInfoBase

// structure to store a word of packet data combined with the 'tail' from the
// previous word
template <typename PacketInfo>
struct PacketWithPrevTailBase {
  uint8_t data[PacketInfo::kDataWithPrevTailSize];
};  // end of struct PacketWithPrevTailBase


// SubmitPacketAlignerPreprocessKernel
// TODO description
template <typename PacketAlignerPreprocessKernelName,  // Name for the Kernel

          typename Protocol,      // Protocol definition, must be a type
                                  // derrived from ProtocolBase.
          typename PacketBus,     // Struct containing all data and metadata
                                  // used as input/output for this kernel.
          typename PacketInfo,    // Struct containing metadata calculated for
                                  // each packet word received
          typename PacketWithPrevTail,  // Struct containing a full packet word
                                        // plus the 'tail' from the previous
                                        // packet word
          typename PacketInPipe,  // Receive data words bundled in a PacketBus
                                  // struct.
          typename PacketOutPipe, // Send data words bundled in a 
                                  // PacketWithPrevTail struct.  Combines data
                                  // from PacketInPipe with end of the previous
                                  // word from the same channel
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
      using TailType = ParallelCopyArray<uint8_t, PacketInfo::kTailLen>;
      TailType prev_word_tail[PacketBus::kNumChannels];

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      while (1) {

        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        PacketBus packet_in;    // current word being processed
  
        bool packet_in_valid;   // valid flag for current word
  
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        PacketInfo packet_info; // metadata calculated for each packet word

        // combine current data with tail from previous word from same channel
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        PacketWithPrevTail packet_with_prev_tail;

        // read a word from the PacketInPipe, if available
        packet_in = PacketInPipe::read(packet_in_valid);

        // retreive the 'tail' of the previous word from this channel
        TailType prev_tail = prev_word_tail[packet_in.channel];
        UnrolledLoop<PacketInfo::kTailLen>([&](auto i) {
          packet_with_prev_tail.data[i] = prev_tail[i];
        });

        // combine the data from this packet with the previous word tail
        UnrolledLoop<PacketBus::kBusWidth>([&](auto i) {
          packet_with_prev_tail.data[i + PacketInfo::kTailLen] = 
            packet_in.data[i];
        });

        // store the tail from this packet
        TailType new_tail;
        constexpr unsigned kTailStartIndex = PacketBus::kBusWidth - 
                                             PacketInfo::kTailLen;
        UnrolledLoop<PacketInfo::kTailLen>([&](auto i) {
          //new_tail[i] = packet_in.data[i + kTailStartIndex];
          prev_word_tail[packet_in.channel][i] = 
            packet_in.data[i + kTailStartIndex];
        });

        // check each byte position for a potentially valid header
        UnrolledLoop<PacketInfo::kHeaderMatchLen>([&](auto i) {
          bool cur_pos_header_match = true;
          UnrolledLoop<Protocol::kLenStart>([&](auto j) {
            constexpr auto position = i + j;
            if constexpr (Protocol::kHdrMask[j]) {
              cur_pos_header_match &= 
                (packet_with_prev_tail.data[position] == Protocol::kHdrVal[j]);
            }
          });
          packet_info.header_match[i] = cur_pos_header_match;
        });

        // determine position of start of next message IF each byte were the 
        // start of the header field in a new message 

        using MsgLenType = ac_int<Protocol::kLengthLen * 8, false>;
        
        // Which word (relative to the current word being 0) would the next
        // message start in, if the given byte were the start of a message
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        MsgLenType next_msg_start_word[PacketInfo::kNextMsgSize];
        
        // Which byte within the data_with_tail array would the next message
        // start on, if the given byte were the start of a message
        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        typename PacketBus::BusPosType 
          next_msg_start_byte[PacketInfo::kNextMsgSize];
        
        // pre-calculate if the next message would start in the same word,
        // equivalent to next_msg_start_word == 0
        bool next_msg_start_same_word[PacketInfo::kNextMsgSize];

        [[intel::fpga_register]]  // NO-FORMAT: Attribute
        typename PacketBus::BusPosType 
          next_word_pos_chain[PacketInfo::kNextMsgSize]
                             [PacketInfo::kMaxMsgsPerWord];


        UnrolledLoop<PacketInfo::kNextMsgSize>([&](auto i) {
          
          // determine the message lengh, if the current byte were the start
          // of a header
          MsgLenType length = 0;
          UnrolledLoop<Protocol::kLengthLen>([&](auto j) {
            // TODO support opposite endianness here?
            constexpr unsigned shift_bits = (Protocol::kLengthLen - j - 1) * 8;
            constexpr auto position = i + j + Protocol::kLenStart;
            length.set_slc(
              shift_bits, 
              (ac_int<8,false>)packet_with_prev_tail.data[position]);
          });
          ac_int<(Protocol::kLengthLen * 8) + 1, false> next_start_pos;
          
          // Calculate the position within the data_with_tail array where the
          // next message would start, if the current byte were the start of
          // a message.
          next_start_pos = length + i;
          next_msg_start_word[i] = next_start_pos >> 
                                   PacketBus::kBusPosTypeWidthBits;
          next_msg_start_same_word[i] = next_msg_start_word[i] == 0;
          next_msg_start_byte[i] = 
            next_start_pos.template slc<PacketBus::kBusPosTypeWidthBits>(0);



          // TODO: Temporary, store all intermediate values in packet_info, for validation
          packet_info.next_msg_start_word[i] = next_msg_start_word[i];
          packet_info.next_msg_start_byte[i] = next_msg_start_byte[i];
          packet_info.next_msg_start_same_word[i] = next_msg_start_same_word[i];
        });

        // write the metadata and input word out for downstream processing
        if (packet_in_valid) {
          PacketInfoOutPipe::write(packet_info);
          PacketOutPipe::write(packet_with_prev_tail);
        }


      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitPacketAlignerPreprocessKernel()

#endif  // ifndef __PACKET_ALIGNER_PREPROCESS_HPP__
