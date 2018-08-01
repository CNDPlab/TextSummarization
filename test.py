from pathos.multiprocessing import ProcessingPool as Pool
import pyhanlp
from tqdm import tqdm

a = ['是佛我为日哦哦 i 我'] * 2000


import time

start = time.time()
res = []
for i in tqdm(a):
    res.append([i.word for i in list(pyhanlp.HanLP.segment(i))])

end = time.time()
print(end-start)


import time

start = time.time()
def segment(input):
    from pyhanlp import HanLP
    res = list(HanLP.segment(input))
    return [i.word for i in res]

pool = Pool(4)
res = pool.map(segment,a)
end = time.time()
print(end-start)



from pathos.multiprocessing import ProcessingPool as Pool
a = ['是佛我为日哦哦 i 我'] * 100
import pyhanlp

with Pool(4) as exe:
    result = exe.map(lambda x: [i.word for i in list(pyhanlp.HanLP.segment(x))], a)
for i in tqdm(result):
    print(i)


from multiprocessing_on_dill.pool import Pool
a = ['是佛我为日哦哦 i 我'] * 100
import pyhanlp

with Pool(4) as exe:
    result = exe.map(lambda x: [i.word for i in list(pyhanlp.HanLP.segment(x))], a)
for i in tqdm(result):
    print(i)
