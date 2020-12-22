from core import *
from imports import *

__all__ = [ 'Database', 'get_con_memsql' , 'read_from_hana', 'create_table', 'populate_table', 'close_connection']

class Database:
    ', '
    """Database connection class."""
    def __init__(self, host, port, user, password, database=None):
        self.host = host
        self.username = user
        self.password = password
        self.port = port
        self.dbname = database
        self.conn = None

    def open_connection(self):
        """Connect to MySQL Database."""
        try:
            if self.conn is None:
                self.conn = pymysql.connect(self.host,
                                            user=self.username,
                                            passwd=self.password,
                                            db=self.dbname,
                                            connect_timeout=5)
        except pymysql.MySQLError as e:
            logging.error(e)
            sys.exit()
        finally:
            logging.info('Connection opened successfully.')

    def run_select_query(self, con, query):
        """Execute SQL query."""
        try:
            #self.open_connection()
            cur = con.cursor()
            if 'SELECT' in query:
                #records = []
                cur.execute(query)
                des = cur.description
                field_names = [field[0] for field in des]
                data = cur.fetchall()
                result = pd.DataFrame(list(data),columns=field_names)
                cur.close()
                return result
        except pymysql.MySQLError as e:
            print(e)
        finally:
            if con:
                con.close()
                con = None
                logging.info('Database connection closed.')
    def get_conn(self):
        return self.conn
    
#dc connection 
from core import quote
from core import Config


# connect
def get_con_memsql():
    db_memsql = Config.get_key('memsql')
    return  Database( host=db_memsql["connection"]["host"],
                port=db_memsql["connection"]["port"],
                user=db_memsql["connection"]["user"],
                password=db_memsql["connection"]["password"],
                database=db_memsql["connection"]["db"])


# DB connect
def get_con_hana():
    db_hana = Config.get_key('hana')
    return  Database( host=db_hana["connection"]["host"],
                      port=db_hana["connection"]["port"],
                      user=db_hana["connection"]["user"],
                      password=db_hana["connection"]["password"])

    
def read_from_hana(query, con):
    df = pd.read_sql(query, con)
    df.info(verbose=True, null_counts=True)
    
    df = df.drop_duplicates(subset=['contract_account'])
    df['contract_account'] = df['contract_account'].astype(np.int64)
    print(" shape of df",df.shape)
    return df
  
def create_table(con, df, drop_tableg, create_table): 
 
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
    
