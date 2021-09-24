//#include <chrono>
//#include <cstring>
//#include <iomanip>
//#include <vector>
#include <iostream>
#include <iomanip>
#include <stdlib.h>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "packet_aligner_constants.hpp"
#include "packet_aligner_preprocess.hpp"
#include "FakeIOPipes.hpp"

// We will have the producer and consumer use USM host allocations
// if they are enabled, otherwise they use device allocations
template <typename Id, typename T, size_t min_capacity = 0>
using MyProducer = Producer<Id, T, kUseUSMHostAllocation, min_capacity>;
template <typename Id, typename T, size_t min_capacity = 0>
using MyConsumer = Consumer<Id, T, kUseUSMHostAllocation, min_capacity>;

// Forward declare the kernel names to reduce name mangling
class PacketProducerID;
class PacketConsumerID;
class PacketInfoConsumerID;
class PacketAlignerPreprocessID;

// define the protocol and packet bus interfaces
// TODO define multiple protocols in a header file to easily select one with
// a CMAKE define.
/*
constexpr unsigned kHeaderLength = 4;
constexpr unsigned kLenStart = kHeaderLength - 2;
constexpr static std::array<bool, kHeaderLength> kHeaderMask = 
  {true,true,false,false};
constexpr static std::array<uint8_t, kHeaderLength> kHeaderVal = 
  {0x45,0x32,0x00,0x00};
using MyProtocol = 
  ProtocolBase<kHeaderLength, kHeaderLength, kLenStart, kHeaderMask, kHeaderVal>;
*/
constexpr unsigned kHeaderLength = 6;
constexpr unsigned kLenStart = kHeaderLength - 3;
constexpr static std::array<bool, kHeaderLength> kHeaderMask = 
  {true,true,true,false,false,false};
constexpr static std::array<uint8_t, kHeaderLength> kHeaderVal = 
  {0x45,0x32,0x11,0xaa,0xcc,0x00};
using MyProtocol = 
  ProtocolBase<kHeaderLength, kHeaderLength, kLenStart, kHeaderMask, kHeaderVal>;

// TODO allow packet bus parameters to be defined by CMAKE
constexpr unsigned kPacketBusWidth = 16;
constexpr unsigned kPacketChannelBitWidth = 5;
using MyPacketBus = PacketBusBase<kPacketBusWidth, kPacketChannelBitWidth>;
using MyPacketInfo = PacketInfoBase<MyProtocol, MyPacketBus>;

// Fake IO Pipes
using PacketProducer =
  MyProducer<PacketProducerID, MyPacketBus, 512>;
using PacketInPipe = PacketProducer::Pipe;
using PacketConsumer =
  MyConsumer<PacketConsumerID, MyPacketBus, 512>;
using PacketOutPipe = PacketConsumer::Pipe;
using PacketInfoConsumer = 
  MyConsumer<PacketInfoConsumerID, MyPacketInfo, 512>;
using PacketInfoPipe = PacketInfoConsumer::Pipe;

// Command Line Arguments
/*
bool ParseArgs(int argc, char *argv[], int &num_matrix_copies,
               std::string &in_dir, std::string &out_dir, UDPArgs *udp_args);
void PrintUsage();
*/

template<typename Protocol, typename PacketBus>
int GenerateRandomMsgData(PacketBus *packets, 
                          unsigned num_packets,
                          unsigned max_msg_len);

template<typename Protocol, typename PacketBus>
void PrintMsgData(PacketBus *packets,
                  unsigned num_packets);

template<typename Protocol, typename PacketBus, typename PacketInfo>
void PrintPacketInfo(PacketInfo *info,
                     unsigned num_packets);


// the main function
int main(int argc, char *argv[]) {

/*
  // parse the command line arguments
  if (!ParseArgs(argc, argv, num_matrix_copies, in_dir, out_dir, &udp_args)) {
    PrintUsage();
    std::terminate();
  }
*/

  bool passed = true;


  try {
    // device selector
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector selector;
#else
    sycl::ext::intel::fpga_selector selector;
#endif

    // input and output data
    unsigned num_input_packets = 4;  // TODO parameterize this from command line
    std::vector<MyPacketBus> input_packets(num_input_packets);
    std::vector<MyPacketBus> preprocess_output_packets(num_input_packets);
    std::vector<MyPacketInfo> preprocess_packet_info(num_input_packets);

    GenerateRandomMsgData<MyProtocol, MyPacketBus>(input_packets.data(), 
                                                   num_input_packets, 20);

    std::cout << "Packet Bus Width = " << kPacketBusWidth << std::endl;
    std::cout << "Num Packets = " << num_input_packets << std::endl;
    std::cout << "Input packet data:" << std::endl;
    PrintMsgData<MyProtocol, MyPacketBus>(input_packets.data(), 
                                          num_input_packets);

    // create the device queue
    sycl::queue q(selector, dpc_common::exception_handler);

    // initialize the producers and consumers
    PacketProducer::Init(q, num_input_packets);
    PacketConsumer::Init(q, num_input_packets);
    PacketInfoConsumer::Init(q, num_input_packets);

    // copy the input data to the producer fake IO pipe buffer
    std::copy_n(input_packets.data(), num_input_packets, PacketProducer::Data());

    // start the Packet Aligner kernels
    // TODO put all packet aligner kernels into a single call
    auto my_event = SubmitPacketAlignerPreprocessKernel<
      PacketAlignerPreprocessID,
      MyProtocol,
      MyPacketBus,
      MyPacketInfo,
      PacketInPipe,
      PacketOutPipe,
      PacketInfoPipe
      >(q);

    // start consumers and producers
/*    sycl::event packet_producer_dma_event;
    sycl::event packet_producer_kernel_event;
    sycl::event packet_consumer_dma_event;
    sycl::event packet_consumer_kernel_event;
    sycl::event packet_info_consumer_dma_event;
    sycl::event packet_info_consumer_kernel_event;
*/
    auto [packet_producer_dma_event, packet_producer_kernel_event] =
      PacketProducer::Start(q, num_input_packets);
    auto [packet_consumer_dma_event, packet_consumer_kernel_event] =
      PacketConsumer::Start(q, num_input_packets);
    auto [packet_info_consumer_dma_event, packet_info_consumer_kernel_event] =
      PacketInfoConsumer::Start(q, num_input_packets);

    // wait for all data transfers to finish
    packet_producer_dma_event.wait();
    packet_producer_kernel_event.wait();
    packet_consumer_kernel_event.wait();
    packet_consumer_dma_event.wait();
    packet_info_consumer_kernel_event.wait();
    packet_info_consumer_dma_event.wait();

    // copy the output back from the consumer
    std::copy_n(PacketInfoConsumer::Data(), num_input_packets,
                preprocess_packet_info.data());


    std::cout << "Preprocess packet_info:" << std::endl;
    PrintPacketInfo<MyProtocol, MyPacketBus, MyPacketInfo>(
      preprocess_packet_info.data(), num_input_packets);

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// generate random messages
template<typename Protocol, typename PacketBus>
int GenerateRandomMsgData(PacketBus *packets, 
                          unsigned num_packets,
                          unsigned max_msg_len) {

  int num_msg = 0;
  int msg_start_byte_pos;
  int msg_start_packet_num;
  int msg_len;
  bool start_new_msg = true;

  for (int packet_num = 0; packet_num < num_packets; packet_num++) {
    for (int byte_pos = 0; byte_pos < PacketBus::kBusWidth; byte_pos++) {
      
      // check if this byte is the start of a new message
      if (start_new_msg) {

        msg_len = (std::rand() % max_msg_len) + 1;
        
        // prevent messages shorter than min message length
        if (msg_len < Protocol::kMinMsgLen) {
          msg_len = Protocol::kMinMsgLen;
        }
        
        // make sure we have space for another message after this one, otherwise
        // round the message length up to the end of the last packet
        if ((packet_num * PacketBus::kBusWidth + byte_pos + msg_len) >
            ((num_packets * PacketBus::kBusWidth) - Protocol::kMinMsgLen)) {
          msg_len = (num_packets * PacketBus::kBusWidth) - 
                    (packet_num * PacketBus::kBusWidth + byte_pos);
        }

        msg_start_byte_pos = byte_pos;
        msg_start_packet_num = packet_num;
        start_new_msg = false;
        num_msg++;
      }

      // calculate our position within the current message
      int msg_pos = 
        ((packet_num - msg_start_packet_num) * PacketBus::kBusWidth) +
        (byte_pos - msg_start_byte_pos);

      // set the value of the current byte based on our position in the msg
      if (msg_pos < Protocol::kLenStart) {
        packets[packet_num].data[byte_pos] = 
          Protocol::kHdrVal[msg_pos];
      } else if (msg_pos < Protocol::kHdrLen ) {
        packets[packet_num].data[byte_pos] = (msg_len >> ((Protocol::kLengthLen - (msg_pos - Protocol::kLenStart) - 1) * 8)) & 0xff;
      } else {
        packets[packet_num].data[byte_pos] = std::rand();
      }

      // determine if next byte is start of a new message
      if (msg_pos == msg_len - 1) {
        start_new_msg = true;
      }

    }
  }

  return num_msg;

}

template<typename Protocol, typename PacketBus>
void PrintMsgData(PacketBus *packets,
                  unsigned num_packets) {

  for (int packet_num = 0; packet_num < num_packets; packet_num++) {
    std::cout << std::setfill('0') << std::setw(4) << std::hex << packet_num << 
      ": ";
    for (int byte_pos = 0; byte_pos < PacketBus::kBusWidth; byte_pos++) {
      std::cout << std::setfill('0') << std::setw(2)  << 
        (unsigned)packets[packet_num].data[byte_pos] << " ";
    }
    std::cout << std::endl;
  }

}


template<typename Protocol, typename PacketBus, typename PacketInfo>
void PrintPacketInfo(PacketInfo *info,
                     unsigned num_packets) {

  std::cout << "header_match:" << std::endl;
  for (int packet_num = 0; packet_num < num_packets; packet_num++) {
    std::cout << std::setfill('0') << std::setw(4) << std::hex << packet_num << 
      ": ";
    for (int i = 0; i < PacketInfo::kHeaderMatchLen; i++) {
      if (info[packet_num].header_match[i]) {
        std::cout << " T ";
      } else {
        std::cout << " F ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "next_msg_start_word:" << std::endl;
  for (int packet_num = 0; packet_num < num_packets; packet_num++) {
    std::cout << std::setfill('0') << std::setw(4) << std::hex << packet_num << 
      ": ";
    for (int i = 0; i < PacketInfo::kNextMsgSize; i++) {
      std::cout << std::setfill('0') << std::setw(4) << std::hex 
                << info[packet_num].next_msg_start_word[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "next_msg_start_byte:" << std::endl;
  for (int packet_num = 0; packet_num < num_packets; packet_num++) {
    std::cout << std::setfill('0') << std::setw(4) << std::hex << packet_num << 
      ": ";
    for (int i = 0; i < PacketInfo::kNextMsgSize; i++) {
      std::cout << std::setfill('0') << std::setw(4) << std::hex 
                << info[packet_num].next_msg_start_byte[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

}
