import struct
#Process in each small block
def read_in_chunks(file_path, chunk_size=128*128): 
    with open(file_path, 'rb') as file:
#第一个区块编号为0
        chunk_number = 0
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
#处理完一个再处理下一个
            chunk_number += 1
#可以去掉
            print(f"Chunk {chunk_number}:")
            print(chunk)
            print("-" * 40)

read_in_chunks('image file/ecc_dti.dpf')
