import csv, gc, gzip, os, pickle, shutil, sys, warnings, yaml, io, subprocess
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import abc, collections, hashlib, itertools, json, operator, pathlib
import mimetypes, inspect, typing, functools, importlib, weakref
import html, re, requests, tarfile, numbers, tempfile, bz2

from abc import abstractmethod, abstractproperty
from collections import Counter, defaultdict, namedtuple, OrderedDict
from collections.abc import Iterable
import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field, InitVar
from enum import Enum, IntEnum
from functools import partial, reduce
from pdb import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from warnings import warn
from contextlib import contextmanager

from matplotlib.patches import Patch
from pandas import Series, DataFrame
from io import BufferedWriter, BytesIO

###################
import numpy as np #for  numerical analytics
import pandas as pd #for data analytics
import category_encoders as ce #encoding
import h2o #h2o api
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OIsolationForestEstimator

from sklearn.preprocessing import LabelEncoder #label encoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_curve, precision_recall_curve, auc,roc_auc_score
from sklearn.metrics import confusion_matrix,fbeta_score
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, style
import itertools

from tqdm import tqdm_notebook
import time

#gaussian mixture model
from sklearn.mixture import GaussianMixture

from numbers import Integral
#################################################

########### db imports ###########
import pyhdb
import pymysql
import warnings


#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None
