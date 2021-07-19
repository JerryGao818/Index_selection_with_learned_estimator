import psycopg2
import os
import json
import pickle

database = 'imdb'
user = 'postgres'
password = 'postgres'
host = 'localhost'
port = '5432'

#Delete all indexes except primary keys
def clear_index():
    delete_count = 0
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    search_index = '''select relname from pg_class where relkind='i' and relname !~ '^(pg_|sql_)';'''
    cur.execute(search_index)
    current_index = cur.fetchall()
    cur.close()
    conn.close()
    print('current indexes are: ', current_index)
    print('below will be removed: ')
    for index in current_index:
        if not index[0].endswith('_pkey'):
            print(index[0])
            delete_count = delete_count + 1
            delete_statement = 'drop index ' + index[0] + ';'
            conn2 = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
            cur2 = conn2.cursor()
            cur2.execute(delete_statement)
            conn2.commit()
            cur2.close()
            conn2.close()
    print(delete_count, ' indexes removed.')


# get index name
def get_index_name(index):
    point = index[0].find('.')
    index_name = index[0][:point]
    for item in index:
        point = item.find('.')
        attr = item[point + 1:]
        index_name = index_name + '_' + attr
    return index_name


# create given index
def create_index(indexes):
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    index_dic = json.load(open('./data_imdb/IndexInformation/index_dic.json', 'r'))
    for index in indexes:
        if index in index_dic:
            name = index_dic[index]
        else:
            i = 0
            while os.path.exists('./data_imdb/IndexInformation/idx' + str(i) + '.json'):
                i += 1
            name = 'idx' + str(i)
            index_dic[index] = name
            json.dump(index_dic, open('./data_imdb/IndexInformation/index_dic.json', 'w'))
        schema = index.split("#")
        # name = schema[0] + '_' + '_'.join(schema[1].split(','))
        sql = 'CREATE INDEX ' + name + ' ON ' + schema[0] + "(" + schema[1] + ');'
        print(sql)
        try:
            cur.execute(sql)
        except Exception as err:
            print(index,' create error: ', err)
        conn.commit()
        get_index_information(name, index)


# # delete given index
def delete_index(index):
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    index_dic = json.load(open('./data_imdb/IndexInformation/index_dic.json', 'r'))
    name = index_dic[index]
    sql = 'drop index ' + name
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
# pg_class    oid
# relpages 表的存储空间大小
# reltuples 表的行数
# relnatts 表的列数

# pg_attribute
# attrelid pg_class 的oid
# atttypid 属性数据类型
# attlen 属性长度
# attnotnull 是否非空

# pg_index
# indexrelid pg_class的oid
# indrelid 索引所在表的oid
# indisunique 唯一索引
# indkey 建立的字段
# get the information of an index for training
# structure: [ index name, table name,  is index unique, number of pages to store the index,
# number of tuples in the index, number of attributes in the index, bitmap of the attributes to index]
#
def get_index_information(idx_name, index):
    filename = './data_imdb/IndexInformation/' + idx_name + '.json'
    if os.path.exists(filename):
        # print('index information existed')
        return
    idx = [idx_name]
    search = '''select c1.relname as indname, c2.relname as tablename, c2.reltuples, c1.reltuples, indisunique, c1.relpages,  indnatts, indkey 
    from pg_class as c1, pg_class as c2, pg_index 
    where indexrelid = c1.oid and c2.oid = indrelid and c1.relname = %s;'''
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    cur.execute(search, idx)
    result = cur.fetchall()
    # print(result)
    # for item in result[0]:
    #     print(item, ' ', type(item))
    file = open(filename, 'w')
    json.dump([index, result[0]], file)
    print('Index information has been saved to: ', filename)
    cur.close()
    conn.close()


# [表名，行数， 存储所用页数，字段数，后面接上所有字段的信息（ 字段名，字段类型，类型长度，是否允许为空）]
# structure of a single table data: [ table name, number of tuples in the table,
# number of pages to store the table, number of attributes in the table, information of all the attributes( attribute name,
# type, length, allowed to be null) ]
def get_table_information():
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    get_tables = '''select oid, relname, reltuples, relpages, relnatts from pg_class where relkind = 'r' and  relname !~ '^(pg_|sql_)';'''
    cur.execute(get_tables)
    tables = cur.fetchall()
    print(tables)
    for table in tables:
        temp = list(table)
        get_information = '''select attname, typname, attlen, attnotnull
        from pg_attribute, pg_type
        where atttypid = oid and attrelid = %s and attnum > 0;'''
        cur.execute(get_information, [temp[0]])
        attrs = cur.fetchall()
        temp = tuple(temp + list(attrs))
        temp = temp[1:]
        print(temp)
        if not os.path.exists('./data_imdb/TableInformation/' + temp[0]):
            file = open('./data_imdb/TableInformation/' + temp[0] + '.json', 'w')
            json.dump(temp, file)
            print('Table information has been saved to: ', temp[0] + '.json')
        # break
    cur.close()
    conn.close()


if __name__ == '__main__':
    clear_index()
    # get_table_information()
    # can = pickle.load(open('./data_imdb/CandidateIndex/IMDBcandidate2.pickle', 'rb'))
    # create_index(can)
    # clear_index()

    # index_dic = json.load(open('./data_imdb/IndexInformation/index_dic.json', 'r'))
    # for i in current_index:
    #     index_dic[i[0]] = i[0]
    #     get_index_information(i[0],i[0])
    # json.dump(index_dic, open('./data_imdb/IndexInformation/index_dic.json', 'w'))

    # a = {}
    # json.dump(a, open('./data_imdb/IndexInformation/index_dic.json', 'w'))
    # clear_index()
    # create_index(['customer_demographics#cd_demo_sk', 'customer_demographics#cd_demo_sk,cd_education_status'])

    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    search_index = '''select relname from pg_class where relkind='i' and relname !~ '^(pg_|sql_)';'''
    cur.execute(search_index)
    current_index = cur.fetchall()
    for index in current_index:
        get_index_information(index[0], index[0])

    # get_table_information()

