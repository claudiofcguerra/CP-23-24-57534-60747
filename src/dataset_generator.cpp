#include "wb.h"


static unsigned char *generate_data(const unsigned int y,
                                    const unsigned int x) {
    /* raster of y rows
       R, then G, then B pixel
       if maxVal < 256, each channel is 1 byte
       else, each channel is 2 bytes
    */
    unsigned int i;

    const int maxVal    = 255;
    unsigned char *data = (unsigned char *)malloc(y * x * 3);

    unsigned char *p = data;
    for (i = 0; i < y * x; ++i) {
        unsigned short r = rand() % maxVal;
        unsigned short g = rand() % maxVal;
        unsigned short b = rand() % maxVal;
        *p++             = r;
        *p++             = g;
        *p++             = b;
    }
    return data;
}

static void write_data(char *file_name, unsigned char *data,
                       unsigned int width, unsigned int height,
                       unsigned int channels) {
    FILE *handle = fopen(file_name, "w");
    if (channels == 1) {
        fprintf(handle, "P5\n");
    } else {
        fprintf(handle, "P6\n");
    }
    fprintf(handle, "#Created by %s\n", __FILE__);
    fprintf(handle, "%d %d\n", width, height);
    fprintf(handle, "255\n");

    fwrite(data, width * channels * sizeof(unsigned char), height, handle);

    fflush(handle);
    fclose(handle);
}



int main(int argc, char* argv[]) {

    if (argc != 4) {
        printf("usage %s path_to_image height width\n", argv[0]);
        return 1;
    }

    const auto x = strtol(argv[3], nullptr, 10);
    const auto y = strtol(argv[2], nullptr, 10);;
    char* input_file_name = wbPath_join(wbDirectory_current(), argv[1]);
    printf("Generating image %s with size %dx%d\n", argv[1], y, x);

    unsigned char *input_data = generate_data(y, x);
    write_data(input_file_name, input_data, x, y, 3);
    free(input_data);
    return 0;
}