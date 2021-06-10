#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "/content/drive/My Drive/Colab Notebooks/LibreriaSTBIMAGE/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/content/drive/My Drive/Colab Notebooks/LibreriaSTBIMAGE/stb_image_write.h"

using namespace std;

__constant__ int BLUR_SIZE = 50;

__global__
void convertToGrayScaleKernel(unsigned char* Pbw, const unsigned char* Pin, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < width && row < height) {
    int grey_loc = row*width + col;
    int rgb_loc = 3*grey_loc;

    unsigned char r = Pin[rgb_loc  ];
    unsigned char g = Pin[rgb_loc+1];
    unsigned char b = Pin[rgb_loc+2];
    Pbw[grey_loc] = 0.21f*r + 0.71f*g + 0.07f*b;
  }
}

/**
 * Blur the BW image (Kernel)
 */
__global__
void blurKernel(unsigned char* Pout, const unsigned char* Pbw, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(col < width && row < height) {
    int loc = row*width + col;

    int pixval = 0;
    int pixels = 0;
    for(int i = -BLUR_SIZE; i < BLUR_SIZE+1; ++i) {
      for(int j = -BLUR_SIZE; j < BLUR_SIZE+1; ++j) {
        int blurRow = row + i;
        int blurCol = col + j;
        if(blurRow > -1 && blurRow < height && blurCol > -1 && blurCol < width) {
          pixval += Pbw[blurRow*width + blurCol];
          pixels++;
        }
      }
    }
    Pout[loc] = (unsigned char)(pixval/pixels);
  }
}
int main(int argc, char* argv[])
{
    int width, height, channels;
    unsigned char* imageIn = stbi_load("/content/drive/My Drive/Colab Notebooks/LibreriaSTBIMAGE/image1.jpg", &width, &height, &channels, 0);
    cout << "Imported image " << "image1.jpg" << " (" << width << " x " << height << ") with " << channels << " channels" << endl;
    int size = width*height;
    unsigned char* out_image1 = new unsigned char[size];
    unsigned char* out_image2 = new unsigned char[size];
 
    unsigned char* d_Pin;
    cudaMalloc((void**)&d_Pin, size*3);
    cudaMemcpy((void*)d_Pin, (void*)imageIn, size*3, cudaMemcpyHostToDevice);
 
    unsigned char* d_Pin2;
    cudaMalloc((void**)&d_Pin2, size*3);
    cudaMemcpy((void*)d_Pin2, (void*)imageIn, size*3, cudaMemcpyHostToDevice);
 
    unsigned char* d_Pin3;
    cudaMalloc((void**)&d_Pin3, size);

    unsigned char* grayscaleout;
    unsigned char* blurout;
    cudaMalloc((void**)&grayscaleout, size);
    cudaMalloc((void**)&blurout, size);

    dim3 dimGrid(ceil(width/16.0f),ceil(height/16.0f),1);
    dim3 dimBlock(16,16,1);
    cout << "Launching a (" << dimGrid.x << " x " << dimGrid.y << " x " << dimGrid.z << ") grid." << endl;
    cout << "Total number of threads: " << dimGrid.x*dimGrid.y*dimGrid.z*16*16 << endl;
    cout << "Number of pixels: " << width*height << endl;
 
    convertToGrayScaleKernel<<<dimGrid,dimBlock>>>(grayscaleout, d_Pin, width, height);
    blurKernel<<<dimGrid,dimBlock>>>(blurout, d_Pin2, width, height);

    cudaMemcpy((void*)out_image1, (void*)grayscaleout, size, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)out_image2, (void*)blurout, size, cudaMemcpyDeviceToHost);
 
    stbi_write_bmp("imageGrayScale_out.bmp", width, height, 1, (void*)out_image1);
    stbi_write_bmp("imageBlur_out.bmp", width, height, 1, (void*)out_image2);

    stbi_image_free(imageIn);
    delete [] out_image1;
    delete [] out_image2;
    cudaFree(d_Pin);
    cudaFree(d_Pin2);
    cudaFree(d_Pin3);
    //cudaFree(d_Pbw);
    //cudaFree(d_Pout);
 }