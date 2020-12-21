from imports import *

__all__ = [ 'ItemList','MultiColumnLabelEncoder', 'ItemList', 'select_columns', 'GridSearch' ]

#########################custom datatype
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#AnnealFunc = Callable[[Number,Number,float], Number]
ArgStar = Collection[Any]
BatchSamples = Collection[Tuple[Collection[int], int]]
DataFrameOrChunks = Union[DataFrame, pd.io.parsers.TextFileReader]
FilePathList = Collection[Path]
Floats = Union[float, Collection[float]]
IntsOrStrs = Union[int, Collection[int], str, Collection[str]]
KeyFunc = Callable[[int], int]
KWArgs = Dict[str,Any]
ListOrItem = Union[Collection[Any],int,float,str]
ListRules = Collection[Callable[[str],str]]
ListSizes = Collection[Tuple[int,int]]
NPArrayableList = Collection[Union[np.ndarray, list]]
NPArrayList = Collection[np.ndarray]
NPArrayMask = np.ndarray
OptDataFrame = Optional[DataFrame]
OptListOrItem = Optional[ListOrItem]
OptRange = Optional[Tuple[float,float]]
OptStrTuple = Optional[Tuple[str,str]]
OptStats = Optional[Tuple[np.ndarray, np.ndarray]]
PathOrStr = Union[Path,str]
PathLikeOrBinaryStream = Union[PathOrStr, BufferedWriter, BytesIO]
Point=Tuple[float,float]
Points=Collection[Point]
Sizes = List[List[int]]
SplitArrayList = List[Tuple[np.ndarray,np.ndarray]]
StartOptEnd=Union[float,Tuple[float,float]]
StrList = Collection[str]
Tokens = Collection[Collection[str]]
OptStrList = Optional[StrList]
np.set_printoptions(precision=6, threshold=50, edgeitems=4, linewidth=120)
#########################custom datatype


#################utility function#################################
def num_cpus()->int:
    "Get number of cpus"
    try:                   return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count()

_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus, cmap='viridis', return_fig=False, silent=False)

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def is_tuple(x:Any)->bool: return isinstance(x, tuple)
def is_dict(x:Any)->bool: return isinstance(x, dict)
def is_pathlike(x:Any)->bool: return isinstance(x, (str,Path))
def noop(x): return x


#quote
def quote(item):
    return "\'" + item + "\'"


#convenience
def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame

def get_auc(labels, scores):
    roc_auc_score
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def get_aucpr(labels, scores):
    precision, recall, th = precision_recall_curve(labels, scores)
    aucpr_score = np.trapz(recall, precision)
    return precision, recall, aucpr_score


def plot_metric(ax, x, y, x_label, y_label, plot_label, style="-"):
    ax.plot(x, y, style, label=plot_label)
    ax.legend()
    
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)


def prediction_summary(labels, predicted_score, info, plot_baseline=True, axes=None):
    if axes is None:
        axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
        

    fpr, tpr, auc_score = get_auc(labels, predicted_score)
    plot_metric(axes[0], fpr, tpr, "False positive rate",
                "True positive rate", "{} AUC = {:.4f}".format(info, auc_score))
    if plot_baseline:
        plot_metric(axes[0], [0, 1], [0, 1], "False positive rate",
                "True positive rate", "baseline AUC = 0.5", "r--")

    precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
    plot_metric(axes[1], recall, precision, "Recall",
                "Precision", "{} AUCPR = {:.4f}".format(info, aucpr_score))
    if plot_baseline:
        thr = sum(labels)/len(labels)
        plot_metric(axes[1], [0, 1], [thr, thr], "Recall",
                "Precision", "baseline AUCPR = {:.4f}".format(thr), "r--")

    plt.show()
    return axes


def figure():
    fig_size = 4.5#2.5
    f = plt.figure()
    f.set_figheight(fig_size)
    f.set_figwidth(fig_size*2)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()
    
def stability_check(train_predict_fn, x, y, ntimes=20):
    scores = ["AUC", "AUCPR"]
    scores = {key: [] for key in scores}
    seeds = [14234, 13235,16244,15236,17254,12254]
    for seed in tqdm_notebook(seeds):
        predictions = train_predict_fn(x, int(seed))
        _, _, auc_score = get_auc(y, predictions)
        _, _, aucpr_score = get_aucpr(y, predictions)

        scores["AUC"].append(auc_score)
        scores["AUCPR"].append(aucpr_score)

    return pd.DataFrame(scores)


####################h2o helper function#############
def iso_forests_h2o(data, seed):
    
    ntrees = 110
    mtries=12
    max_depth=16
    sample_rate = 0.7


    isoforest = h2o.estimators.H2OIsolationForestEstimator(
        ntrees=ntrees, seed=seed, mtries=mtries, max_depth=max_depth, sample_rate=sample_rate)
    isoforest.train(x=data.col_names, training_frame=data)
    preds = isoforest.predict(data)
    return preds.as_data_frame()["predict"]
    
def create_table(con, drop_tableg, create_table): 
 
    # call open cursor ***CHECK
    with con.cursor() as MemSQLcursor:
        MemSQLcursor.execute(drop_table)
        MemSQLcursor.execute(create_table)
        con.commit()
        con.close()

def populate_table(con, df, populate_table): 

    with con.cursor() as MemSQLcursor:
        MemSQLcursor.executemany(populate_table, df.values.tolist())
        con.commit()
        con.close()


def close_connection(con, query):
    df = pd.read_sql(query, con)
    con.commit()
    con.close()
    df.info(verbose=True, null_counts=True)
    print(df.shape)
    
    return df
    
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
def _check_file(fname):
    size = os.path.getsize(fname)
    return size

def _maybe_squeeze(arr): return (np.array(arr).ravel() if np.array(arr).shape == () else np.squeeze(arr))

def df_names_to_idx(names:IntsOrStrs, df:DataFrame):
    "Return the column indexes of `names` in `df`."
    if not is_listy(names): names = [names]
    if isinstance(names[0], int): return names
    return [df.columns.get_loc(c) for c in names]

def is1d(a:Collection)->bool:
    "Return `True` if `a` is one-dimensional"
    return len(a.shape) == 1 if hasattr(a, 'shape') else len(np.array(a).shape) == 1

class H2OHelper():
    def __init__(self, df:DataFrame):
        self.df = df
        h2o.init()
        
    def getH2OFrame(self, df:DataFrame, types):
        return h2o.H2OFrame(df, column_types=types)
    
    def fillNA(self, df):
        return df.fillna(method="forward", axis=0,maxlen=1)

    def setColumnNames(self,df, cols):
        df.set_names(cols)

####################helper config class############
class Config():
    # Path 
    DEFAULT_CONFIG_LOCATION = os.path.expanduser(path)
    @classmethod
    def get_key(cls, key):
        return cls.get().get(key, cls.DEFAULT_CONFIG.get(key, None))
    

    @classmethod
    def get_path(cls, path):
        "Get the `path` in the config file."
        return _expand_path(cls.get_key(path))

    @classmethod
    def data_path(cls):
        "Get the path to data in the config file."
        return cls.get_path('data_path')

    @classmethod
    def model_path(cls):
        "Get the path to fastai pretrained models in the config file."
        return cls.get_path('model_path')

    @classmethod
    def get(cls, fpath=None, create_missing=True):
        "Retrieve the `Config` in `fpath`."
        fpath = _expand_path(fpath or cls.DEFAULT_CONFIG_PATH)
        if not fpath.exists() and create_missing: cls.create(fpath)
        assert fpath.exists(), f'Could not find config at: {fpath}. Please create'
        with open(fpath, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)

    @classmethod
    def create(cls, fpath):
        "Creates a `Config` from `fpath`."
        fpath = _expand_path(fpath)
        assert(fpath.suffix == '.yml')
        if fpath.exists(): return
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w') as yaml_file:
            yaml.dump(cls.DEFAULT_CONFIG, yaml_file, default_flow_style=False)
            
    @classmethod
    def get_datafile(cls, filename):
        "Creates a `Config` from `fpath`."
        return datapath4file(filename)

def _expand_path(fpath): return Path(fpath).expanduser()

class GridSearch():
    def __init__(self,hyper_params):
        self.models = []
        self.fscores = []
        self.hyper_params = hyper_params
        self.df = pd.DataFrame(hyper_params, columns=['seed','ntrees','mtries','max_depth', 'sample_rate'])
        

    def build_train_model(self, x, training_frame):
        n,m = self.df.shape #row,col
        #t = time.process_time()
        for i in range(n):
            isoforest_models = h2o.estimators.H2OIsolationForestEstimator(
            ntrees=(int)(self.df['ntrees'][i]), seed=(int)(self.df['seed'][i]), mtries=(int)(self.df['mtries'][i]), max_depth=(int)(self.df['max_depth'][i]),            
                 sample_rate=(float)(self.df['sample_rate'][i]))
            isoforest_models.train(x=x,training_frame=training_frame)
            self.models.append(isoforest_models)
        #print(self.models)
        return self.models
    

    def optimal_model(self, df):
        
        for isoforest in self.models:
            maxfscore = 0
            idx = 0
            t = time.process_time()
            predictions_test = isoforest.predict(df)
            elapsed_time = time.process_time() - t
            #print("test process time: ",elapsed_time)
            quantile = 0.95
            quantile_frame_test = predictions_test.quantile([quantile])
            threshold_test = quantile_frame_test[0, "predictQuantiles"]
            predictions_test["predicted_class"] = predictions_test["predict"] > threshold_test
            predictions_test["class"] = df['EV']
            predictions_test.shape
            #for test
            h2o_predictions_test = predictions_test.as_data_frame()

            #from sklearn.metrics import confusion_matrix,fbeta_score
            fscore = fbeta_score(y_pred=h2o_predictions_test["predicted_class"], y_true=h2o_predictions_test["class"],beta=2)
            #print("f2-score",fscore)
            self.fscores.append(fscore)
            #print("f2-score",fscore)
        array = np.asarray(self.fscores)
        idx = array.argmax()
        print("index",idx)
        maxfscore = array[idx]
        
        
        return self.models[idx], maxfscore
        
 
