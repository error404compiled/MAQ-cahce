import logging

import maqcache

FORMAT = '%(asctime)s - %(thread)d - %(filename)s-%(module)s:%(lineno)s - %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)

maqcache_log = logging.getLogger(f'maqcache:{maqcache.__version__}')
