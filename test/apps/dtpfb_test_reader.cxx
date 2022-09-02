
#include <iostream>

#include "dtpfeedbacktools/RawFileReader.hpp"

using namespace dunedaq::dtpfeedbacktools;


int main(int argc, char const *argv[])
{
    /* code */
    std::cout << "AAAARGH" << std::endl;

    auto rfr = RawFileReader("./raw_record_15285/output_0_4.out");
    std::cout << "File size " << rfr.get_size() << std::endl;

    auto blk = rfr.read_block(1024);

    std::cout << blk->size() << std::endl;

        return 0;
}
