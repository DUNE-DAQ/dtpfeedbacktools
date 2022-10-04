#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <iostream>
#include <fmt/core.h>


#include "dtpfeedbacktools/RawFileReader.hpp"
#include "dtpfeedbacktools/fwtp_unpack_utils.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace dunedaq::dtpfeedbacktools;


int main(int argc, char const *argv[])
{
    /* code */
    CLI::App app{"App description"};

    std::string filename = "default";
    bool verbose;

    app.add_option("-f,--file", filename, "block file name")->required();
    app.add_flag("-v", verbose, "Enable verbose output");

    CLI11_PARSE(app, argc, argv);

    
    auto rfr = RawFileReader(filename);
    std::cout << "File size " << rfr.get_size() << std::endl;

    auto blk = rfr.read_block(1024*1024*1024);

    fmt::print("Block size : {}\n", blk->size());
    // auto fwtp_vector = unpack_fwtps(blk->data(), blk->size()/24);
    auto fwtp_vector = unpack_fwtps_2g(blk->data(), blk->size()/12);
    fmt::print("Unpacked {} TPs\n", fwtp_vector.size());
    return 0;
}
