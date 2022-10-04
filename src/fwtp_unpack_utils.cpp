#include "dtpfeedbacktools/fwtp_unpack_utils.hpp"

#include <cstddef>
#include <iostream>
#include <vector>
#include "fmt/core.h"

namespace dunedaq {
namespace dtpfeedbacktools {

template<typename T, typename U> constexpr size_t offset_of(U T::*member)
{
    return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

//-----------------------------------------------------------------------------
std::vector<FWTP> check_fwtps(void *buf, size_t n_blocks, bool safe_mode){

}


//-----------------------------------------------------------------------------
std::vector<FWTP> unpack_fwtps(void *buf, size_t n_blocks, bool safe_mode){

    constexpr uint16_t trl_magic_1 = 0xf00d;
    constexpr uint16_t trl_magic_2 = 0xfeed;

    std::vector<FWTP> fwtps;
    fwtps.reserve(n_blocks/3);


    size_t offset(0), offset_pad1(0), offset_trl(0);

    // Search for the first trailer
    // discard block before it

    // Work in bytes
    auto buf_u8 = static_cast<uint8_t*>(buf);
    size_t buf_size = n_blocks * sizeof(FWTPHeader);

    // Find the first occurence of f00d
    for (; offset_pad1<buf_size; ++offset_pad1) {

      if (((buf_u8[offset_pad1+1]<<8)+buf_u8[offset_pad1]) != trl_magic_1) {
        continue;
      }

      if (offset_pad1 < offset_of(&FWTPTrailer::padding_1)) {
        continue;
      }

      break;
    }

    // work back the trailer offset
    offset_trl= offset_pad1-offset_of(&FWTPTrailer::padding_1);

    // calculate the binary to mem block phase
    offset = offset_trl % sizeof(FWTPHeader);

    size_t i_trl_0((offset_trl-offset)/sizeof(FWTPHeader));

    FWTPHeader *hdr = static_cast<FWTPHeader *>(buf+offset);
    FWTPTrailer *trl = static_cast<FWTPTrailer *>(buf+offset);

    // Initialize indexes
    size_t i = 0;


    // In safe mode start from the block after the first trailer
    if ( safe_mode ) {
      i = i_trl_0+1;
    }

  // std::cout << "starting from block " << i << " (byte offset " ")" << offset << std::endl;

    size_t i_hdr(i), i_trl(i);

    //std::cout << "n blocks " << n_blocks << std::endl;

    for (; i<n_blocks; ++i) {

      if (trl[i].padding_1 != trl_magic_1) {
        //std::cout << "Something went terribly wrong..." << std::endl;
        //std::cout << "trl.padding_1 " << trl[i].padding_1 << std::endl;
        continue;
      }
      //std::cout << "iblock " << i << std::endl;
      //std::cout << "trl.padding_1 " << trl[i].padding_1 << std::endl;

      i_trl = i;

      int64_t n_hits = ((int64_t)i_trl-((int64_t)i_hdr+1));
      if (  n_hits < 1 ) {
        std::cout << "Block " << i << " -> " << n_hits << " ~ " <<i_trl << " " << i_hdr << std::endl;
      } else {
        fwtps.push_back(FWTP((void*)(hdr+i_hdr), i_trl-(i_hdr+1)));
      }

      i_hdr = i+1;
    }

    return fwtps;
}


//-----------------------------------------------------------------------------
std::vector<FWTP> unpack_fwtps_2g(void *buf, size_t n_blocks, bool safe_mode){

    std::vector<FWTP> fwtps;
    fwtps.reserve(n_blocks/3);

    constexpr uint16_t trl_marker_1 = 0xf00d;
    constexpr uint16_t trl_marker_2 = 0xfeed;
    constexpr uint16_t trl_marker_3 = 0xbeef;
    constexpr uint16_t trl_marker_4 = 0xdead;

    // There are 6 16b words per TP block
    // The trailer markers start at offset 2 from the block
    constexpr uint8_t words_per_block = 6;
    constexpr uint8_t marker_offset = 2;

    // 1 block = 6 16b words
    uint16_t* data = reinterpret_cast<uint16_t*>(buf);
    
    size_t n_words = n_blocks*words_per_block;
    size_t tp_offset = 0;
    size_t i(0);
    
    // i+3 corresponds to the offset of the last word of the trailer.
    for(; (i+3)<n_words; ++i) {
        // fmt::print("{:08d} 0x{:04x} 0x{:04x} 0x{:04x} 0x{:04x}\n", i, data[i],  data[i+1],  data[i+2],  data[i+3]);
        // fmt::print("{:08d} 0x{:04x}\n", i, data[i]);

      if (
        data[i  ] == trl_marker_1 && 
        data[i+1] == trl_marker_2 &&
        data[i+2] == trl_marker_3 && 
        data[i+3] == trl_marker_4 
      ){
        uint64_t delta = i - (tp_offset-(words_per_block-marker_offset));

        if ( safe_mode && tp_offset == 0 ) {
          tp_offset = i+words_per_block-marker_offset;
          continue;
        }
        
        // fmt::print("Found TP trailer at {} (tp_offset={})\n", i, tp_offset);
        if (
          ((delta % words_per_block != 0) or (delta < 3*words_per_block)) and 
            tp_offset != 0
          ) {
            fmt::print("Found TP trailer at the wrong distance from previous one: {} delta={} (tp_offset={})\n", i, delta, tp_offset);
            continue;
        }

        // for ( size_t j(tp_offset-2*words_per_block); j<=i+words_per_block; ++j) {
        //   fmt::print(">> {} 0x{:04x}\n", j, data[j]);
        // }

        size_t n_hits = (delta/words_per_block)-2;
        if ( n_hits < 1 ) { 
            fmt::print("Empty TP: {} delta={} (tp_offset={})\n", i, delta, tp_offset);
        }
        fwtps.emplace_back((void*)(data+tp_offset), n_hits);
      
        tp_offset = i+words_per_block-marker_offset;
      
      }

    }
    return fwtps;
  
}

} // namespace dtpfeedbacktools
} // namespace dunedaq

