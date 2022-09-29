#include "dtpfeedbacktools/RawFileReader.hpp"

#include <iostream>

namespace dunedaq
{
  namespace dtpfeedbacktools
  {

    RawFileReader::RawFileReader(const std::string &path) : m_path(path)
    {
     
        // std::cout << "Opening " << m_path << std::endl;

        m_file.open(m_path, std::ios::binary);
        if ( !m_file.is_open() ) {
          // throw something here
          throw std::runtime_error("failed to open file");
        }

        m_file.seekg(0, std::ios_base::end);
        m_size = m_file.tellg();
        m_file.seekg(0, std::ios_base::beg);

        // std::cout << "Opened " << m_path << " size " << m_size << std::endl;
    }

    RawFileReader::~RawFileReader() {
      if ( m_file.is_open()) 
        m_file.close();
    }

    std::unique_ptr<RawDataBlock>
    RawFileReader::read_block(size_t size, size_t offset) {

      if (offset > m_size) {
        // throw something
        std::cout << "AAARGH" << " file size=" <<  m_size << " offset=" << offset << std::endl;
        
        return std::unique_ptr<RawDataBlock>();
      }

      if (offset+size > m_size) {

        std::cout << "WARNING: file shorter than requested size. file size=" <<  m_size << " offset+size=" << offset+size << std::endl;
        std::cout << "WARNING: file shorter than requested size. New size: " << m_size-offset << std::endl;
        size = m_size-offset;
        // throw something
      }


      m_file.seekg(offset);

      auto block = std::make_unique<RawDataBlock>(size);
      m_file.read(block->m_data, block->m_size);

      return block;
    }

  } // namespace dtpfeedbacktools
} // namespace dunedaq
