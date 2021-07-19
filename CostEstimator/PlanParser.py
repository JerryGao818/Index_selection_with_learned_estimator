import os
import json
import re
import pandas as pd
import re
import pypred
import numpy as np
import warnings


name_column = {
        'id':0,
        'name':1,
        'imdb_index':2,
        'imdb_id':3,
        'gender':4,
        'name_pcode_cf':5,
        'name_pcode_nf':6,
        'surname_pcode':7,
        'md5sum':8
    }

role_type_column = {
        'id':0,
        'role':1
    }

title_column = {
        'id':0,
        'title':1,
        'imdb_index':2,
        'kind_id':3,
        'production_year':4,
        'imdb_id':5,
        'phonetic_code':6,
        'episode_of_id':7,
        'season_nr':8,
        'episode_nr':9,
        'series_years':10,
        'md5sum':11
    }

movie_info_column = {
        'id':0,
        'movie_id':1,
        'info_type_id':2,
        'info':3,
        'note':4
    }

person_info_column = {
        'id':0,
        'person_id':1,
        'info_type_id':2,
        'info':3,
        'note':4
    }

complete_cast_column = {
    'id': 0,
    'movie_id': 1,
    'subject_id': 2,
    'status_id': 3
}

info_type_column = {
    'id': 0,
    'info': 1
}

keyword_column = {
    'id': 0,
    'keyword': 1,
    'phonetic_code': 2
}

kind_type_column = {
    'id': 0,
    'kind': 1
}

link_type_column = {
    'id': 0,
    'link': 1
}

movie_companies_column = {
    'id': 0,
    'movie_id': 1,
    'company_id': 2,
    'company_type_id': 3,
    'note': 4
}

movie_info_idx_column = {
    'id': 0,
    'movie_id': 1,
    'info_type_id': 2,
    'info': 3,
    'note': 4
}

movie_keyword_column = {
    'id': 0,
    'movie_id': 1,
    'keyword_id': 2
}

movie_link_column = {
    'id': 0,
    'movie_id': 1,
    'linked_movie_id': 2,
    'link_type_id': 3
}

aka_title_column = {
    'id': 0,
    'movie_id': 1,
    'title': 2,
    'imdb_index': 3,
    'kind_id': 4,
    'production_year': 5,
    'phonetic_code': 6,
    'episode_of_id': 7,
    'season_nr': 8,
    'episode_nr': 9,
    'note': 10,
    'md5sum': 11
}

cast_info_column = {
    'id': 0,
    'person_id': 1,
    'movie_id': 2,
    'person_role_id': 3,
    'note': 4,
    'nr_order': 5,
    'role_id': 6
}

char_name_column = {
    'id': 0,
    'name': 1,
    'imdb_index': 2,
    'imdb_id': 3,
    'name_pcode_nf': 4,
    'surname_pcode': 5,
    'md5sum': 6
}

comp_cast_type_column = {
    'id': 0,
    'kind': 1
}

company_name_column = {
    'id': 0,
    'name': 1,
    'country_code': 2,
    'imdb_id': 3,
    'name_pcode_nf': 4,
    'name_pcode_sf': 5,
    'md5sum': 6
}

company_type_column = {
    'id': 0,
    'kind': 1
}
aka_name_column = {
        'id':0,
        'person_id':1,
        'name':2,
        'imdb_index':3,
        'name_pcode_cf':4,
        'name_pcode_nf':5,
        'surname_pcode':6,
        'md5sum':7
    }


data = {}
data["aka_name"] = aka_name_column
data["aka_title"] = aka_title_column
data["cast_info"] = cast_info_column
data["char_name"] = char_name_column
data["company_name"] = company_name_column
data["company_type"] = company_type_column
data["comp_cast_type"] = comp_cast_type_column
data["complete_cast"] = complete_cast_column
data["info_type"] = info_type_column
data["keyword"] = keyword_column
data["kind_type"] = kind_type_column
data["link_type"] = link_type_column
data["movie_companies"] = movie_companies_column
data["movie_info"] = movie_info_column
data["movie_info_idx"] = movie_info_idx_column
data["movie_keyword"] = movie_keyword_column
data["movie_link"] = movie_link_column
data["name"] = name_column
data["person_info"] = person_info_column
data["role_type"] = role_type_column
data["title"] = title_column

# print(data)
index_keywords = json.load(open('../CostEstimator/index_keyword.json', 'r'))
keyword_columns = []
keyword_columns = keyword_columns + index_keywords

for key,value in data.items():
    for key2 in value:
        keyword_columns.append(key+"."+key2)

# print(keyword_columns)

class Operator(object):
    def __init__(self, opt):
        self.op_type = 'Bool'
        self.operator = opt

    def __str__(self):
        return 'Operator: ' + self.operator


class Comparison(object):
    def __init__(self, opt, left_value, right_value):
        self.op_type = 'Compare'
        self.operator = opt
        self.left_value = left_value
        self.right_value = right_value

    def __str__(self):
        return 'Comparison: ' + self.left_value + ' ' + self.operator + ' ' + self.right_value


def remove_invalid_tokens(predicate):
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text ~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__LIKE__\2')", predicate)
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text !~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTLIKE__\2')", x)
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text <> \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTEQUAL__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) ~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__LIKE__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) !~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTLIKE__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) <> \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTEQUAL__\2')", x)
    x = re.sub(r'(\'[^\']*\')::[a-z_]+', r'\1', x)
    x = re.sub(r'\(([^\(]+)\)::[a-z_]+', r'\1', x)
    x = re.sub(r'\(([a-z_0-9A-Z\-]+) = ANY \(\'(\{.+\})\'\[\]\)\)', r"(\1 = '__ANY__\2')", x)
    # print(predicate,x)
    return x


def predicates2seq(pre_tree, alias2table, relation_name, index_name):
    current_level = -1
    current_line = 0
    sequence = []
    while current_line < len(pre_tree):
        operator_str = pre_tree[current_line]
        level = len(re.findall(r'\t', operator_str))
        operator_seq = operator_str.strip('\t').split(' ')
        operator_type = operator_seq[1]
        operator = operator_seq[0]
        if level <= current_level:
            for i in range(current_level - level + 1):
                sequence.append(None)
        current_level = level
        if operator_type == 'operator':
            sequence.append(Operator(operator))
            current_line += 1
        elif operator_type == 'comparison':
            operator = operator_seq[0]
            current_line += 1
            operator_str = pre_tree[current_line]
            operator_seq = operator_str.strip('\t').split(' ')
            left_type = operator_seq[0]
            left_value = operator_seq[1]
            current_line += 1
            operator_str = pre_tree[current_line]
            operator_seq = operator_str.strip('\t').split(' ')
            right_type = operator_seq[0]
            if right_type == 'Number':
                right_value = operator_seq[1]
            elif right_type == 'Literal':
                p = re.compile("Literal (.*) at line:")
                result = p.search(operator_str)
                right_value = result.group(1)
            elif right_type == 'Constant':
                p = re.compile("Constant (.*) at line:")
                result = p.search(operator_str)
                right_value = result.group(1)
            else:
                raise "Unsupport Value Type: " + right_type
            if re.match(r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$', left_value) is not None:
                left_relation = left_value.split('.')[0]
                left_column = left_value.split('.')[1]
                if left_relation in alias2table:
                    left_relation = alias2table[left_relation]
                left_value = left_relation + '.' + left_column
            else:
                if relation_name is None:
                    relation = index_name.replace(left_value + '_', '')
                else:
                    relation = relation_name
                left_value = relation + '.' + left_value
            if re.match(r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$', right_value) is not None:
                right_relation = right_value.split('.')[0]
                right_column = right_value.split('.')[1]
                if right_relation in alias2table:
                    right_relation = alias2table[right_relation]
                right_value = right_relation + '.' + right_column
            sequence.append(Comparison(operator, left_value, right_value.strip('\'')))
            current_line += 1
    return sequence


def pre2seq(predicates, alias2table, relation_name, index_name):
    pr = remove_invalid_tokens(predicates)
    pr = pr.replace("''", " ")
    p = pypred.Predicate(pr)
    try:
        predicates = predicates2seq(p.description().strip('\n').split('\n'), alias2table, relation_name, index_name)
    except:
        # predicates = None
        predicates = []
    return predicates


class Materialize(object):
    def __init__(self):
        self.node_type = 'Materialize'

    def __str__(self):
        return 'Materialize'


class Limit(object):
    def __init__(self):
        self.node_type = 'Limit'

    def __str__(self):
        return 'Limit'


class Unique(object):
    def __init__(self):
        self.node_type = 'Unique'

    def __str__(self):
        return 'Unique'


class WindowAgg(object):
    def __init__(self, keys):
        self.node_type = 'WindowAgg'
        self.group_keys = keys

    def __str__(self):
        return 'WindowAgg ON: ' + ','.join(self.group_keys)


class Gather(object):
    def __init__(self):
        self.node_type = 'Gather'

    def __str__(self):
        return 'Gather'


class Gather_Merge(object):
    def __init__(self):
        self.node_type = 'Gather Merge'

    def __str__(self):
        return 'Gather Merge'

class Aggregate(object):
    def __init__(self, strategy, keys):
        self.node_type = 'Aggregate'
        self.strategy = strategy
        self.group_keys = keys

    def __str__(self):
        return 'Aggregate ON: ' + ','.join(self.group_keys)


class SetOp(object):
    def __init__(self, strategy):
        self.node_type = 'SetOp'
        self.strategy = strategy

    def __str__(self):
        return 'SetOp Using: ' + self.strategy


class Group(object):
    def __init__(self, keys):
        self.node_type = 'Group'
        self.group_keys = keys

    def __str__(self):
        return 'Group ON: ' + ','.join(self.group_keys)


class Sort(object):
    def __init__(self, sort_keys):
        self.sort_keys = sort_keys
        self.node_type = 'Sort'

    def __str__(self):
        return 'Sort by: ' + ','.join(self.sort_keys)


class Hash(object):
    def __init__(self):
        self.node_type = 'Hash'

    def __str__(self):
        return 'Hash'


class Append(object):
    def __init__(self):
        self.node_type = 'Append'

    def __str__(self):
        return 'Append'


class Join(object):
    def __init__(self, node_type, condition_seq):
        self.node_type = node_type
        self.condition = condition_seq

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition])


class Scan(object):
    def __init__(self, node_type, condition_seq_filter, condition_seq_index, relation_name, index_name):
        self.node_type = node_type
        self.condition_filter = condition_seq_filter
        self.condition_index = condition_seq_index
        self.relation_name = relation_name
        self.index_name = index_name

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition_filter]) + '; ' + ','.join(
            [str(i) for i in self.condition_index])


class CTEScan(object):
    def __init__(self, CTE_name):
        self.node_type = 'CTE Scan'
        self.CTE_name = CTE_name

    def __str__(self):
        return  'CTE Scan on ' + self.CTE_name


class SubqueryScan(object):
    def __init__(self, alias):
        self.node_type = 'Subquery Scan'
        self.alias = alias

    def __str__(self):
        return 'Subquery Scan on ' + self.alias


class BitmapCombine(object):
    def __init__(self, operator):
        self.node_type = operator

    def __str__(self):
        return self.node_type


class Result(object):
    def __init__(self):
        self.node_type = 'Result'

    def __str__(self):
        return 'Result'


def class2json(instance):
    if instance == None:
        return json.dumps({})
    else:
        return json.dumps(todict(instance))


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
                     for key, value in obj.__dict__.items()
                     if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj



def change_alias2table(column, alias2table):
    # print(column, column.endswith('::text)'))
    try:
        relation_name = column.split('.')[0]
        column_name = column.split('.')[1]
        if relation_name in alias2table:
            return alias2table[relation_name] + '.' + column_name
        else:
            return column
    except Exception as e:
        return column

def extract_info_from_node(node, alias2table):
    relation_name, index_name = None, None
    if 'Relation Name' in node:
        relation_name = node['Relation Name']
    if 'Index Name' in node:
        index_name = node['Index Name']
    if node['Node Type'] == 'Materialize':
        return Materialize(), None
    elif node['Node Type'] == 'Hash':
        return Hash(), None
    elif node['Node Type'] == 'Sort':
        keys = [change_alias2table(key, alias2table) for key in node['Sort Key'] if not key.endswith('::text)')]
        return Sort(keys), None
    elif node['Node Type'] == 'BitmapAnd':
        return BitmapCombine('BitmapAnd'), None
    elif node['Node Type'] == 'BitmapOr':
        return BitmapCombine('BitmapOr'), None
    elif node['Node Type'] == 'Result':
        return Result(), None
    elif node['Node Type'] == 'Append':
        return Append(), None
    elif node['Node Type'] == 'Hash Join':
        return Join('Hash Join', pre2seq(node["Hash Cond"], alias2table, relation_name, index_name)), None
    elif node['Node Type'] == 'Merge Join':
        return Join('Merge Join', pre2seq(node["Merge Cond"], alias2table, relation_name, index_name)), None
    elif node['Node Type'] == 'Nested Loop':
        if 'Join Filter' in node:
            condition = pre2seq(node['Join Filter'], alias2table, relation_name, index_name)
        else:
            condition = []
        return Join('Nested Loop', condition), None
    elif node['Node Type'] == 'Aggregate':
        if 'Group Key' in node:
            keys = [change_alias2table(key, alias2table) for key in node['Group Key'] if not key.endswith('::text)')]
        else:
            keys = []
        return Aggregate(node['Strategy'], keys), None
    elif node['Node Type'] == 'SetOp':
        return SetOp(node['Strategy']), None
    elif node['Node Type'] == 'WindowAgg':
        if 'Group Key' in node:
            keys = [change_alias2table(key, alias2table) for key in node['Group Key'] if not key.endswith('::text)')]
        else:
            keys = []
        return WindowAgg(keys), None
    elif node['Node Type'] == 'Group':
        if 'Group Key' in node:
            keys = [change_alias2table(key, alias2table) for key in node['Group Key']]
        else:
            keys = []
        return Group(keys), None
    elif node['Node Type'] == 'Seq Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name = [], node["Relation Name"], None
        return Scan('Seq Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    elif node['Node Type'] == 'CTE Scan':
        return CTEScan(node['CTE Name']), None
    elif node['Node Type'] == 'Subquery Scan':
        return SubqueryScan(node['Alias']), None
    elif node['Node Type'] == 'Bitmap Heap Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name = [], node["Relation Name"], None
        return Scan('Bitmap Heap Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    elif node['Node Type'] == 'Index Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        relation_name, index_name = node["Relation Name"], node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name), condition_seq_index
        else:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    elif node['Node Type'] == 'Bitmap Index Scan':
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name = [], None, node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name), condition_seq_index
        else:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    elif node['Node Type'] == 'Index Only Scan':
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name = [], None, node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name), condition_seq_index
        else:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    elif node['Node Type'] == 'Gather':
        return Gather(), None
    elif node['Node Type'] == 'Gather Merge':
        return Gather_Merge(), None
    elif node['Node Type'] == 'Limit':
        return Limit(), None
    elif node['Node Type'] == 'Unique':
        return Unique(), None
    else:
        raise Exception('Unsupported Node Type: ' + node['Node Type'])
        return None, None
#


def plan2seq(root, alias2table):
    sequence = []
    join_conditions = []
    static_sequence = []
    input_tables = []
    node, join_condition = extract_info_from_node(root, alias2table)
    if join_condition != None:
        join_conditions += join_condition
    if node != None:
        sequence.append(node)
    start_cost = 0
    total_cost = 0
    plan_rows = 0
    if 'Startup Cost' in root:
        if root['Startup Cost'] > 0:
            start_cost = np.log(root['Startup Cost'])
            # start_cost = root['Startup Cost']
    if 'Total Cost' in root:
        if root['Total Cost'] > 0:
            total_cost = np.log(root['Total Cost'])
            # total_cost = root['Total Cost']
    if 'Plan Rows' in root:
        plan_rows = root['Plan Rows']
    static_sequence.append([start_cost, total_cost, plan_rows])
    if 'Plans' in root:
        for plan in root['Plans']:
            next_sequence, next_join_conditions, next_static_sequence, next_input_table = plan2seq(plan, alias2table)
            sequence += next_sequence
            static_sequence += next_static_sequence
            join_conditions += next_join_conditions
            input_tables += next_input_table
    else:
        if 'Relation Name' in root:
            input_tables.append(root['Relation Name'])
    # sequence.append(None)
    # static_sequence.append([0,0])
    return sequence, join_conditions,static_sequence, input_tables




def get_subplan(root):
    results = []
    if 'Actual Rows' in root and 'Actual Total Time' in root and root['Actual Rows'] > 0:
        results.append((root, root['Actual Total Time'], root['Actual Rows']))
    if 'Plans' in root:
        for plan in root['Plans']:
            results += get_subplan(plan)
    return results


def get_plan(root,use_cost=False, return_time=False):
    if use_cost==False:
        if return_time:
            return (root, root['Actual Total Time'], root['Actual Rows'], root['Actual Total Time'])
        else:
            return (root, root['Actual Total Time'], root['Actual Rows'])
    else:
        if return_time:
            return (root, root['Total Cost'], root['Plan Rows'],root['Actual Total Time'])
        else:
            return (root, root['Total Cost'], root['Plan Rows'])


class PlanInSeq(object):
    def __init__(self, seq, cost, cardinality, time=None):
        self.seq = seq
        self.cost = cost
        self.cardinality = cardinality
        self.time = time


def get_alias2table(root, alias2table):
    if 'Relation Name' in root and 'Alias' in root:
        alias2table[root['Alias']] = root['Relation Name']
    if 'Plans' in root:
        for child in root['Plans']:
            get_alias2table(child, alias2table)




def toseq(node_json):
    # print(type(node_json))
    if node_json is None:
        return (None,1)
    elif isinstance(node_json,list):
        res = []
        for child_json in node_json:
            child_seq = toseq(child_json)
            if isinstance(child_seq,list):
                res += child_seq
            else:
                res.append(child_seq)
        return res
    elif isinstance(node_json,dict):
        res = []
        for key in node_json:
            value = toseq(node_json[key])
            if isinstance(value,list):
                res += value
            elif isinstance(value,tuple):
                res.append(value)
            else:
                tp = 2 if key=="right_value" and value not in keyword_columns else 1
                res.append((value,tp))
        return res
    else:
        return node_json

def json2seq(root_json):
    r_d = {}
    for key,value in root_json.items():
        if key=="seq" and isinstance(value,list):
            res = []
            for v in value:
                # join_type = True if isinstance(v,dict) and 'Join' in v['node_type'] else False
                res.append(toseq(v))
            r_d[key] = res
        else:
            r_d[key] = value
    return r_d








