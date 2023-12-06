#ifndef __WH_BITARRAY_H_
#define __WH_BITARRAY_H_

#include <cstddef>
#include <string.h>
#include <exception>
#include <stdexcept>

namespace pretty_tools
{
    /**
        在map的position位置写入bit
    **/
    bool writeBit(unsigned char *, int, bool);

    /**
       读取map的position位置的bit数据
    **/
    bool readBit(unsigned char *, int);

    /* 三个用于 BitArray 的静态常量 */
    static const size_t c_initBitsCapacity = 80U;
    static const double c_increaseCapacity = 1.5;
    static const double c_maxAllowedOutOfBound = 3.0;

    class BitArray;

    class Bit
    {
    private:
        BitArray *m_bits;
        int m_position;

    public:
        Bit() : m_bits(nullptr), m_position(0) {}
        Bit(BitArray *bits, int position);
        Bit &operator=(bool bit);
        operator bool();
    };

    /**
        BitArray 可以对比特位进行直接操作，通过构造方法或者setData()传入一个字符指针之后就可以将BitArray视作一个由比特组成的数组
        set() 以及 get() 方法封装了对位进行操作的两个最主要的函数
    **/
    class BitArray
    {
        // 类型，常量定义区
        /*    无符号字符类型 */
        typedef unsigned char uchar;
        inline size_t BitsToBytes(size_t bits) { return (bits - 1) / 8 + 1; }
        inline size_t BytesToBits(size_t bytes) { return 8 * bytes; }

    public:
        /**
        默认构造函数，创建一个默认大小 c_initBitsCapacity 的比特数组
        **/
        BitArray();
        BitArray &operator=(const BitArray &bits);
        BitArray &operator=(BitArray &&bits);
        BitArray(const BitArray &bits);
        BitArray(BitArray &&bits);
        /**
        创建一个长为bitsLength，最大容量为bitsCapacity的比特数组
        **/
        BitArray(size_t bitsLength, size_t bitsCapacity = 0U);
        /**
        根据现有字符数组创建一个比特数组
            data: 现有的字符数组
            bitsLength: 该字符数组有效的比特位长度，创建之后的最大容量为 8*((bitsLength-1)/8+1)
            isClear: 该字符数组是否进行清零
            isOwns: 是否允许比特数组获得对该字符数组的控制权，若为true则在析构或其他恰当时机将会进行内存释放
        **/
        BitArray(uchar *data, int bitsLength, bool isClear = false, bool isOwns = true);
        ~BitArray();
        bool operator==(BitArray &bits);
        /**
        获得position位置的真值，
        **/
        Bit operator[](int position) { return Bit(this, position); }
        Bit at(int position) { return Bit(this, position); }
        /**
        获得position位置的真值，有效范围为 [-(int)getBitSize(),getBitSize()),超出将抛出异常
        **/
        bool get(int position);
        /**
        获得比特数组的底层字节数据，该数组的有效长度可由 getBitSize()/getByteSize() 得到
        **/
        uchar *getData() { return m_data; }
        /**
        根据现有字符数组更新比特数组，原有的数据将根据m_owns的真值来决定是否释放
            data: 现有的字符数组
            bitsLength: 该字符数组有效的比特位长度，创建之后的最大容量为 8*((bitsLength-1)/8+1)
            isClear: 该字符数组是否进行清零
            isOwns: 是否允许比特数组获得对该字符数组的控制权，若为true则在析构或其他恰当时机将会进行内存释放
        **/
        void setData(uchar *data, int bitsLength, bool isClear = false, bool isOwns = true);
        /**
        对比特数组的position位置进行数据更新
            position: 访问位置，以0为起点，合法范围为 [-(int)getBitSize(),getBitSize())，超出将可能抛出异常
            bit: 将要更新的真值
            isAllowOutOfRange: 是否允许在适当时机进行数据扩增，并且最大扩充倍数为c_maxAllowOutOfRange，默认是不允许的
            isAllowOutOfSize: 是否允许当超出当前长度，但是并未超出容量时进行自动扩张,默认是允许的
            isAllowToInfinite： 是否允许大小无限大，默认是不允许的
        **/
        bool set(int position, bool bit, bool isAllowOutOfRange = false, bool isAllowOutofSize = true, bool isAllowToInfinite = false);
        /**
        设置比特数组的有效长度，单位：比特
        如果超出容量，将会进行扩容，扩增后的容量为 c_increaseCapacity*newBitsLength
        **/
        size_t setBitSize(size_t newBitsLength);
        /**
        设置比特数组的最大容量，单位：比特，但是将会以8为基本单位对齐
        只要底层数据的字节数与新容量的占用字节数不同，就将重新分配内存，并且获得对新内存的支配权
        **/
        size_t setBitCapacity(size_t newBitsCapacity);
        /**
        获得比特数组的有效比特长度，单位为：比特
        **/
        size_t getBitSize() { return m_bitsLength; }
        /**
        获得比特数组的最大比特容量，单位为：比特
        **/
        size_t getBitCapacity() { return m_bitsCapacity; }
        /**
        获得比特数组的有效字节长度，单位为：字节
        **/
        size_t getByteSize() { return BitsToBytes(m_bitsLength); }
        /**
        获得比特数组的最大字节容量，单位为：字节
        **/
        size_t getByteCapacity() { return BitsToBytes(m_bitsCapacity); }
        /**
        判断是否拥有对底层数组的控制权
        **/
        bool isOwns() { return m_owns; }
        /**
        设置是否拥有对底层数组的控制权
        **/
        bool setOwns(bool owns);

    private:
        /* 底层数据数组    */
        uchar *m_data;
        /* 比特数组的有效长度 */
        size_t m_bitsLength;
        /* 比特数组的最大比特位容量，该值将永远是8的倍数 */
        size_t m_bitsCapacity;
        /* 代表比特数组是否拥有对m_data的控制权，拥有控制权则将在适当时机对其进行释放 */
        bool m_owns;
    };

}

#endif