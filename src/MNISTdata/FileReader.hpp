#ifndef FILEREADER_HPP
#define FILEREADER_HPP
#include<string>
#include<vector>
#include<iostream>
#include <cstdint>
class FileReader{
    private:
        std::string filePathImages;
        std::string filePathLabels;
        std::vector<std::vector<uint8_t>> images;  //uint8_t is 1 byte
        std::vector<uint8_t> labels;

    public:
        FileReader(std::string filePathImages, std::string filePathLabels);
        void read_mnist();
        const std::vector<std::vector<uint8_t>>& get_images() const;
        void read_labels(const std::string& labelFilePath);
        const std::vector<uint8_t>& get_labels() const;
    private:
        int reverseInt(int i);
};

#endif