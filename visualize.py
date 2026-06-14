from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        
        data_str = ', '.join([f"{v:.4f}" for v in n.data[:2]]) + ('...' if len(n.data) > 2 else '')
        grad_str = ', '.join([f"{g:.4f}" for g in n.grad[:2]]) + ('...' if len(n.grad) > 2 else '')
        label = f"data {data_str} | grad {grad_str}"
        dot.node(name=uid, label=label, shape='record', style='filled', fillcolor='white')
        
        if n._op:
            op_uid = uid + '_op'
            dot.node(name=op_uid, label=n._op, shape='oval', style='filled', fillcolor='lightgrey')
            dot.edge(op_uid, uid)
            
            for child in n._prev:
                dot.edge(str(id(child)), op_uid)
    
    return dot