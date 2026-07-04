#include "FileReader.hpp"
#include <fstream>
#include <vector>
#include <iostream>

FileReader::FileReader(std::string filePathImages, std::string filePathLabels) : filePathImages(filePathImages), filePathLabels(filePathLabels) {}

//https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int FileReader::reverseInt(int i){
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void FileReader::read_mnist(){
    std::ifstream file(this->filePathImages, std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        
        int image_size = n_rows * n_cols;
        images.resize(number_of_images, std::vector<uint8_t>(image_size));

        for (int i = 0; i < number_of_images; ++i) {
            file.read(reinterpret_cast<char*>(images[i].data()), image_size);
        }


    std::ifstream labelFile(this->filePathLabels, std::ios::binary);
    if (labelFile.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        labelFile.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        labelFile.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        labels.resize(number_of_labels);

        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            labelFile.read((char*)&temp, 1);
            labels[i] = temp;
        }
    }
    }
}

const std::vector<std::vector<uint8_t>>& FileReader::get_images() const {
    return images;
}

const std::vector<uint8_t>& FileReader::get_labels() const {
    return labels;
}