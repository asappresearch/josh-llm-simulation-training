import json
# Load in MultiWOZ 2.2 data
with open('data/multi-woz/data.json') as fin1:
    data = json.load(fin1)

# Build the api arguments from real MultiWOZ examples
apis = {}
for k in data.keys():
    for idx, tstt in enumerate(data[k]['log']):
        if idx % 2 == 0:
            continue
        tst = tstt['metadata']
        # assert base.keys() == tst.keys()
        for domain in ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']:
            if domain not in apis:
                apis[domain] = {}

            for key in tst[domain].keys():
                api_action = 'book' if key=='book' else 'search'
                for values in  tst[domain][key].keys():
                    if f'{api_action}_{domain}' not in apis[domain]:
                        apis[domain][f'{api_action}_{domain}'] = {'parameters': [], 'returns': []}
                    
                    if api_action == 'book' and values == 'booked':
                        booked_list = tst[domain][key][values]
                        for book in booked_list:
                            [apis[domain][f'{api_action}_{domain}']['returns'].append(k) for k in  book.keys() if k not in apis[domain][f'{api_action}_{domain}']['returns']]
                    else:
                        if values not in apis[domain][f'{api_action}_{domain}']['parameters']:
                            apis[domain][f'{api_action}_{domain}']['parameters'].append(values)  

# Build the return types for search apis by pulling return information from the databases 
import sqlite3
# These are the only domains that have databases, so they are the only ones we will use search on
domains = ['restaurant', 'hotel', 'attraction', 'train']
dbs = {}
for domain in domains:
    db = 'db/{}-dbase.db'.format(domain)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c

# Query the domains to find the column names
res = {}
for domain in domains:
    c = dbs[domain]
    # Get all table names in the DB.
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()

    # Iterate over each table.
    for table_name in tables:
        table_name = table_name[0]
        c.execute('PRAGMA TABLE_INFO({})'.format(table_name))
        # Collect the column names.
        columns = [tup[1] for tup in c.fetchall()]
        # Save the result.
        if domain not in res:
            res[domain] = {}
        res[domain][table_name] = columns

# Add returns to search apis
for domain in res.keys():
    apis[domain][f'search_{domain}']['returns'] = res[domain][domain]


with open('apis.json', 'w') as file:
    json.dump(apis, file, indent=2)