"""
Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import time
import os
from multiprocessing import Pool

def multi_process(func, repeat_args, fnames, cpus=os.cpu_count()):
    """
    Use multiprocess pool to exec func.
    repeat args are used every time.
    A different f in fnames will be passed on
    each execution of func
    """
    print('calling', func.__name__, 'on', len(fnames), 'images')
    start = time.time()
    pool = Pool(cpus)
    async_results = []

    for fname in fnames:
        res = pool.apply_async(func, args=list(repeat_args) + [fname])
        async_results.append(res)
    pool.close()
    pool.join()

    results = [res.get() for res in async_results]

    print(func.__name__, 'on', len(fnames), 'images took', time.time() - start)
    return results
