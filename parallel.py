from multiprocessing import Pool
import os

def process_image(name):
    return name + 1

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-1)                 # Create a multiprocessing Pool
    data_inputs = [1,2,3,4,5,6,7,8,9,10,2145,14,512,35,1235,1,4,624,51,25,312,4,2134,32,4]
    k = pool.map(process_image, data_inputs)  # proces data_inputs iterable with pool
    pool.close()
    pool.join()
    print(k)