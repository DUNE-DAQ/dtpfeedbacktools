#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <iostream>

#include "dtpfeedbacktools/RawFileReader.hpp"
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

    std::cout << blk->size() << std::endl;

    uint64_t tp_size = blk->size()/2;
    uint16_t* tp_data = reinterpret_cast<uint16_t*>(blk->data());

    uint16_t marker_1 = 0xf00d;
    uint16_t marker_2 = 0xfeed;
    size_t last_marker = 0;
    std::map<uint8_t, size_t> counts;
    size_t i(0);
    for( ; i<tp_size; ++i) {

        if (tp_data[i] == marker_1 && tp_data[i+1] == marker_2 ){
        
            uint64_t dist = i - last_marker;
        
            if ((((dist % 6) != 0) or dist < 18) and last_marker != 0) {
                std::cout << "Found TP trailer at the wrong distance from previous one: " << std::dec << i << " dist" << dist << " (last=" << last_marker << ")" <<std::endl;
                for ( size_t j(last_marker-6); j<=i+6; ++j) {
                    std::cout << std::dec << j << "   " << std::hex << std::setw(4) << std::setfill('0') << tp_data[j] << std::endl;
                }
        
            } else {
        
                ++counts[(dist-12)/6];
        
            }
            last_marker = i;
        }

    }

    std::cout << "Scanned " << i << " words" << std::endl;

    json report;
    for( auto const & [k, v] : counts) {
        report[std::to_string(k)] = v;
    }

    std::cout << std::setw(2) << report << std::endl;
    return 0;
}
