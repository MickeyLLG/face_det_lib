import json
sym=json.load(open('lnet106_112-symbol.json','r'))
nodes=sym['nodes']

eff_nodes = [node for node in nodes if node['op'] != 'null']
print(len(eff_nodes))

code=''
data='data'
for node in eff_nodes:
    name=node['name']
    op=node['op']
    try:
        attrs=node['attrs']
    except:
        attrs={}
        print(name)
    attrs['data']=data
    attrs['name']='\"'+name+'\"'
    code+='{0}=mx.symbol.{1}('.format(name,op)
    for attr in attrs:
        code+=attr+'='+attrs[attr]+','
    code=code[:-1]+')\n'
    data=name
code=code.replace('act_type=prelu','act_type=\"prelu\"')
fw=open('code.txt','w')
fw.write(code)
fw.close()