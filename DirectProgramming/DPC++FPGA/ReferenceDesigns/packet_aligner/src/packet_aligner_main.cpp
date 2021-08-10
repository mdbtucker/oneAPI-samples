//#include <chrono>
//#include <cstring>
//#include <iomanip>
//#include <vector>
#include <iostream>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

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
constexpr unsigned kHeaderLength = 4;
constexpr static std::array<bool, kHeaderLength> kHeaderMask = 
  {true,true,false,false};
constexpr static std::array<uint8_t, kHeaderLength> kHeaderVal = 
  {0x45,0x32,0x00,0x00};
using MyProtocol = 
  ProtocolBase<kHeaderLength, kHeaderLength, 2, kHeaderMask, kHeaderVal>;
// TODO allow packet bus parameters to be defined by CMAKE
constexpr unsigned kPacketBusWidth = 16;
constexpr unsigned kPacketChannelBitWidth = 5;
using MyPacketBus = PacketBusBase<kPacketBusWidth, kPacketChannelBitWidth>;
using MyPacketInfo = PacketInfoBase<kPacketBusWidth, kHeaderLength>;

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
    sycl::INTEL::fpga_emulator_selector selector;
#else
    sycl::INTEL::fpga_selector selector;
#endif

    // create the device queue
    sycl::queue q(selector, dpc_common::exception_handler);

    // initialize the producers and consumers
    constexpr unsigned kNumPackets = 1024;  // TODO parameterize this from command line
    PacketProducer::Init(q, kNumPackets);
    PacketConsumer::Init(q, kNumPackets);
    PacketInfoConsumer::Init(q, kNumPackets);

    std::cout << "Packet Bus Width = " << kPacketBusWidth << std::endl;
    std::cout << "Num Packets = " << kNumPackets << std::endl;

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
    sycl::event packet_producer_dma_event;
    sycl::event packet_producer_kernel_event;
    sycl::event packet_consumer_dma_event;
    sycl::event packet_consumer_kernel_event;
    sycl::event packet_info_consumer_dma_event;
    sycl::event packet_info_consumer_kernel_event;
    std::tie(packet_producer_dma_event, packet_producer_kernel_event) =
        PacketProducer::Start(q, kNumPackets);
    std::tie(packet_consumer_dma_event, packet_consumer_kernel_event) =
        PacketConsumer::Start(q, kNumPackets);
    std::tie(packet_info_consumer_dma_event, packet_info_consumer_kernel_event)
      = PacketInfoConsumer::Start(q, kNumPackets);

    // TODO wait on appropriate events

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
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

