#include "dtpfeedbacktools/RawFileReader.hpp"

#include <iostream>

namespace dunedaq
{
  namespace dtpfeedbacktools
  {

    RawFileReader::RawFileReader(const std::string &path) : m_path(path)
    {
        m_file.open(m_path, std::ios::binary);
        if ( !m_file.is_open() ) {
          // throw something here
        }

        m_file.seekg(0, std::ios_base::end);
        m_size = m_file.tellg();
        m_file.seekg(0, std::ios_base::beg);

        std::cout << "Opened " << m_path << " size " << m_size << std::endl;
    }

    RawFileReader::~RawFileReader() {
      if ( m_file.is_open()) 
        m_file.close();
    }

    std::unique_ptr<RawDataBlock>
    RawFileReader::read_block(size_t size, size_t offset) {

      if (offset + size > m_size) {
        // throw something
        std::cout << "AAARGH" << std::endl;
        
        return std::unique_ptr<RawDataBlock>();
      }

      m_file.seekg(offset);

      auto block = std::make_unique<RawDataBlock>(size);
      m_file.read(block->m_data, block->m_size);

      return block;
    }

  } // namespace dtpfeedbacktools
} // namespace dunedaq
