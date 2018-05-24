import time
import os
import os.path as path
from os.path import join as pj
import json
from pprint import pprint
from pprint import pformat
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

import var_config as cf
import dataPreProcess as preProcess
from EvalDataset import EvalDataset
import cnn