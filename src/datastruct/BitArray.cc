#include "BitArray.h"

namespace pretty_tools
{

    BitArray::BitArray()
    {
        m_owns = true;
        m_bitsLength = 0;
        if (c_initBitsCapacity == 0)
        {
            m_data = nullptr;
            m_bitsCapacity = 0;
        }
        else
        {
            size_t t_bytesLength = BitsToBytes(c_initBitsCapacity);
            m_data = new uchar[t_bytesLength];
            memset(m_data, 0, t_bytesLength);
            if (!m_data)
            {
                // 内存分配失败逻辑
                throw std::bad_alloc(); //("can't allow memory!");
            }
            m_bitsCapacity = 8 * t_bytesLength;
        }
    }

    BitArray::BitArray(BitArray &&bits)
    {
        m_data = bits.m_data;
        m_owns = true;
        m_bitsCapacity = bits.m_bitsCapacity;
        m_bitsLength = bits.m_bitsLength;
        bits.m_owns = false;
        bits.m_data = nullptr;
    }

    BitArray::BitArray(const BitArray &bits)
    {
        *this = bits;
    }

    BitArray &BitArray::operator=(const BitArray &bits)
    {
        m_data = bits.m_data;
        m_owns = true;
        m_bitsCapacity = bits.m_bitsCapacity;
        m_bitsLength = bits.m_bitsLength;
        uchar *t_data = new uchar[BitsToBytes(m_bitsCapacity)];
        memcpy(t_data, m_data, BitsToBytes(m_bitsCapacity));
        m_data = t_data;
        return *this;
    }

    BitArray &BitArray::operator=(BitArray &&bits)
    {
        m_data = bits.m_data;
        m_owns = true;
        m_bitsCapacity = bits.m_bitsCapacity;
        m_bitsLength = bits.m_bitsLength;
        bits.m_owns = false;
        bits.m_data = nullptr;
        return *this;
    }

    BitArray::BitArray(size_t bitsLength, size_t bitsCapacity)
    {
        /**
            整体思路：如果 bitsCapacity==0 ，那么默认容量将以 bitsLength乘以默认系数扩增
        **/
        //
        m_bitsLength = bitsLength;
        m_owns = true;
        size_t t_fact_bitsCapacity = bitsCapacity;
        if (t_fact_bitsCapacity < bitsLength)
        {
            t_fact_bitsCapacity = size_t(c_increaseCapacity * bitsLength);
        }
        size_t t_fact_bytesCapacity = BitsToBytes(t_fact_bitsCapacity);
        m_bitsCapacity = 8 * t_fact_bytesCapacity;
        m_data = new uchar[t_fact_bytesCapacity];
        if (!m_data)
        {
            // 内存分配失败逻辑
            throw std::bad_alloc(); //("can't allow memory!");
        }
        memset(m_data, 0, t_fact_bytesCapacity);
    }

    BitArray::~BitArray()
    {
        if (m_owns && m_data != nullptr)
            delete[] m_data;
    }

    bool BitArray::operator==(BitArray &bits)
    {
        if (m_bitsLength != bits.m_bitsLength)
            return false;
        for (int i = 0; i < m_bitsLength; i++)
        {
            if (get(i) != bits.get(i))
            {
                return false;
            }
        }
        return true;
    }

    BitArray::BitArray(unsigned char *data, int bitsLength, bool isClear, bool isOwns)
    {
        m_data = data;
        m_bitsLength = bitsLength;
        m_bitsCapacity = 8 * BitsToBytes(m_bitsLength);
        m_owns = isOwns;
        size_t t_bytesLength = BitsToBytes(m_bitsLength);
        if (isClear)
            memset(m_data, 0, t_bytesLength);
    }

    void BitArray::setData(unsigned char *data, int bitsLength, bool isClear, bool isOwns)
    {
        if (m_owns && m_data != nullptr)
            delete[] m_data;
        m_data = data;
        m_bitsLength = bitsLength;
        m_bitsCapacity = 8 * BitsToBytes(m_bitsLength);
        m_owns = isOwns;
        size_t t_bytesLength = BitsToBytes(m_bitsLength);
        if (isClear)
            memset(m_data, 0, t_bytesLength);
    }

    bool BitArray::set(int position, bool bit, bool isAllowOutOfRange, bool isAllowOutOfSize, bool isAllowToInfinite)
    {
        /**
            整体思路：将position分为六个区间，(-INF,-m_len),[-m_len,0),
                [0,m_len),[m_len,m_cap),[m_cap,c_max*m_cap),[c_max*m_cap,INF)
            一定越界的范围:(-INF,-m_len)
            越界与否取决于isAllowToInfinite：[c_max*m_cap,INF)
            越界与否取决于isAllowedOutOfRange:[m_cap,c_max*m_cap)  及 isAllowToInfinite
            越界与否取决于isAllowOutOfSize:[m_len,m_cap)    及 isAllowToInfinite
            合法访问范围：[-m_len,0),[0,m_len),
        **/
        // position比 -(int)m_bitsLength 还小，或者需要扩张的倍数超出c_maxAllowedOutOfBound,此时一定越界
        if (position < -(int)m_bitsLength || (position >= size_t(c_maxAllowedOutOfBound * m_bitsCapacity) && !isAllowToInfinite))
        {
            throw std::out_of_range("Out of range , This position is too larger!");
        }
        // 注意 isAllowToInfinite ， 如果这个值为 true，那么其他的条件开关将被忽略
        // 如果不允许进行自动扩张，而访问位置超出 m_bitsCapacity
        if (!isAllowOutOfRange && position >= m_bitsCapacity && !isAllowToInfinite)
        {
            throw std::out_of_range("Out of range , You are not allowed to automatically expanded memory!");
        }
        if (!isAllowOutOfSize && position >= m_bitsLength && !isAllowToInfinite)
        {
            throw std::out_of_range("Out of range , You are not allowed to amplification size automatically!");
        }
        // 以负数进行访问，修正position的实际位置,使得 [-m_len,0) -> [0,m_len)
        if (position < 0)
        {
            position += m_bitsLength;
        }
        if (position < m_bitsLength)
        {
            // 访问位置没有超出目前的长度
            return writeBit(m_data, position, bit);
        }
        else if (position >= m_bitsLength && position < m_bitsCapacity)
        {
            // 访问的位置已经超出了目前的长度，但是并没有超出实际的容量
            m_bitsLength = position + 1;
            return writeBit(m_data, position, bit);
        }
        else
        {
            size_t t_new_bitsLength = position + 1;
            size_t t_new_bytesCapacity = BitsToBytes(c_increaseCapacity * t_new_bitsLength);
            size_t t_new_bitsCapacity = 8 * t_new_bytesCapacity;
            uchar *t_data = new uchar[t_new_bytesCapacity];
            if (!t_data)
            {
                // 内存分配失败逻辑
                throw std::bad_alloc(); //("can't allow memory!");
            }
            memset(t_data, 0, t_new_bytesCapacity);
            memcpy(t_data, m_data, BitsToBytes(m_bitsCapacity));
            if (m_owns)
            {
                delete[] m_data;
            }
            m_data = t_data;
            m_bitsCapacity = t_new_bitsCapacity;
            m_bitsLength = t_new_bitsLength;
            m_owns = true;
            return writeBit(m_data, position, bit);
        }
    }

    bool BitArray::get(int position)
    {
        if (position >= m_bitsLength || position < -(int)m_bitsLength)
        {
            // 访问越界,抛出异常
            throw std::out_of_range("The location of the access is illegal!");
        }
        if (position < 0 && position >= -(int)m_bitsLength)
        {
            // 以负数进行访问，修正position的实际位置
            position += m_bitsLength;
        }
        return readBit(m_data, position);
    }

    size_t BitArray::setBitSize(size_t newBitsLength)
    {
        size_t origin_bitsLength = m_bitsLength;
        if (newBitsLength <= m_bitsCapacity)
        {
            m_bitsLength = newBitsLength;
        }
        else
        {
            // 既然需要将大小扩充至newBitsLength，那么在newBitsLength-1 处赋false即可完成该功能
            set(newBitsLength - 1, false, true, true, true);
        }
        return origin_bitsLength;
    }

    size_t BitArray::setBitCapacity(size_t newBitsCapacity)
    {
        /**
            整体思路：无论新的容量是多少，显然当与原来大小不一样时是需要进行扩容的
            但是如果新容量比原来的长度还小，那么长度必须进行修改
        **/
        // 原来的容量必定为8的倍数
        size_t origin_bytesCapacity = BitsToBytes(m_bitsCapacity);
        size_t new_bytesCapacity = BitsToBytes(newBitsCapacity);
        if (origin_bytesCapacity != new_bytesCapacity)
        {
            uchar *t_data = new uchar[new_bytesCapacity];
            if (!t_data)
            {
                // 内存分配失败逻辑
                throw std::bad_alloc(); //("can't allow memory!");
            }
            memset(t_data, 0, new_bytesCapacity);
            if (origin_bytesCapacity < new_bytesCapacity)
            {
                // 如果新容量比原来的容量大，那么全部复制
                memcpy(t_data, m_data, origin_bytesCapacity);
            }
            else
            {
                // 如果新容量比原来的容量小，那么仅复制一部分
                memcpy(t_data, m_data, new_bytesCapacity);
            }
            if (m_owns)
            {
                delete[] m_data;
            }
            m_data = t_data;
            m_bitsCapacity = BytesToBits(new_bytesCapacity);
            if (m_bitsLength > m_bitsCapacity)
            {
                m_bitsLength = m_bitsCapacity;
            }
            m_owns = true;
        }
        return BytesToBits(origin_bytesCapacity);
    }

    bool BitArray::setOwns(bool owns)
    {
        bool r = m_owns;
        m_owns = owns;
        return r;
    }

    /*
        在map的position位置写入bit
    */
    bool writeBit(unsigned char *map, int position, bool bit)
    {
        // sub表示在szMap中的下标，pos表示在该位置中相应的比特位
        int sub = (position) / 8;
        int pos = 7 - (position) % 8;
        if (bit)
        {
            map[sub] |= 1 << pos; // 打开位开关
        }
        else
        {
            map[sub] &= ~(1 << pos); // 关闭位开关
        }
        return true;
    }

    /*
        读取map的position位置的bit数据
    */
    bool readBit(unsigned char *map, int position)
    {
        // sub 代表 szMap中对应的下标，范围是[0,bitmapLength) ；pos为相应的bit位置，范围是[0,8)
        int sub = (position) / 8;
        int pos = 7 - (position) % 8;
        return bool((map[sub] >> pos) & 1);
    }

    Bit::Bit(BitArray *bits, int position)
    {
        m_bits = bits;
        m_position = position;
    }

    Bit &Bit::operator=(bool bit)
    {
        m_bits->set(m_position, bit, true);
        return *this;
    }

    Bit::operator bool()
    {
        return m_bits->get(m_position);
    }
}