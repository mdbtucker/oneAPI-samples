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
class DataProducerID;
class DataConsumerID;

// Fake IO Pipes
/*
using DataProducer =
    MyProducer<DataProducerID, TODO type, TODO pipe depth>;
using DataInPipe = DataProducer::Pipe;
using DataConsumer =
    MyConsumer<DataConsumerID, TODO type, TODO pipe depth>;
using DataOutPipe = DataConsumer::Pipe;
*/
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
/*
    DataProducer::Init(q, kInputDataSize * num_matrix_copies);
    DataOutConsumer::Init(q, kDataOutSize * num_matrix_copies);
    SinThetaProducer::Init(q, kNumSteer);
*/

    constexpr static std::array<bool,4> mask = {true,true,false,false};
    constexpr static std::array<uint8_t,4> val = {0x45,0x32,0x00,0x00};
    using MyProtocol = ProtocolBase<4,4,2,mask,val>;
    MyProtocol my_protocol;

    std::cout << my_protocol.kLenStart << " " << my_protocol.kHdrMask[1] << " " << (int)my_protocol.kHdrVal[0] << std::endl;

    auto my_event = SubmitPacketAlignerPreprocessKernel<
      class MyName,
      class MyPacketBus,
      MyProtocol,
      class Pipe1,
      class Pipe2,
      class Pipe3
      >(q);

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

